#![recursion_limit = "256"]

use std::{path::PathBuf, process::ExitCode};

use clap::Parser;
use rootcause::prelude::ResultExt;
use z_image::{
    GenerateFromTextOpts, GenerateOpts,
    modules::{ae::AutoEncoderConfig, transformer::ZImageModelConfig},
};
use qwen3_burn::{Qwen3Config, Qwen3Tokenizer};

cfg_if::cfg_if! {
    if #[cfg(any(feature = "vulkan", feature = "vulkan-fusion", feature = "metal", feature = "metal-fusion", feature = "metal-simple"))] {
        type B = burn::backend::Wgpu;
    } else if #[cfg(feature = "rocm")] {
        type B = burn::backend::Rocm;
    } else if #[cfg(feature = "cuda")] {
        type B = burn::backend::Cuda;
    } else if #[cfg(feature = "cpu")] {
        type B = burn::backend::Cpu;
    } else if #[cfg(feature = "tch")] {
        type B = burn::backend::LibTorch;
    } else if #[cfg(any(feature = "candle-metal", feature = "candle-cpu"))] {
        type B = burn::backend::Candle;
    } else {
        compile_error!("Please select a backend by enabling the respective feature");
    }
}

#[derive(Parser)]
struct Args {
    /// The text prompt to generate an image from.
    /// If provided, the text encoder will be used to embed the prompt.
    #[arg(short = 'P', long)]
    prompt: Option<String>,

    /// The path to save the resulting image to.
    #[arg(short, long, default_value = "out.png")]
    out: PathBuf,

    /// The path to a safetensors file containing an embedding of the prompt.
    /// Only used if --prompt is not provided.
    #[arg(short = 'e', long, default_value = "models/prompt.safetensors")]
    prompt_file: PathBuf,

    /// The path to the transformer model weights as a safensors or bpk file.
    #[arg(long, default_value = "models/z_image_turbo_bf16.bpk")]
    transformer: PathBuf,

    /// The path to the autoencoder model weights as a safensors or bpk file.
    #[arg(long, default_value = "models/ae.safetensors")]
    ae: PathBuf,

    /// The path to the text encoder model weights (safetensors format).
    /// Required when using --prompt.
    #[arg(long, default_value = "models/text_encoder")]
    text_encoder: PathBuf,

    /// The path to the tokenizer.json file.
    /// Required when using --prompt.
    #[arg(long, default_value = "models/tokenizer/tokenizer.json")]
    tokenizer: PathBuf,

    /// The width of the image in pixels.
    #[arg(short = 'W', long, default_value_t = 1024)]
    width: usize,

    /// The height of the image in pixels.
    #[arg(short = 'H', long, default_value_t = 1024)]
    height: usize,
}

fn main() -> ExitCode {
    let args = Args::parse();

    #[cfg(feature = "tch")]
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    #[cfg(feature = "candle-metal")]
    let device = burn::backend::candle::CandleDevice::metal(0);
    #[cfg(feature = "candle-cpu")]
    let device = burn::backend::candle::CandleDevice::Cpu;
    #[cfg(not(any(feature = "tch", feature = "candle-metal", feature = "candle-cpu")))]
    let device = Default::default();

    // Load autoencoder
    let mut autoencoder = AutoEncoderConfig::flux_ae().init(&device);
    println!("Loading autoencoder");
    if let Err(err) = autoencoder
        .load_weights(&args.ae)
        .context("Failed to load autoencoder weights")
    {
        eprintln!("{err}");
        return ExitCode::FAILURE;
    }

    // Load transformer
    let mut transformer = ZImageModelConfig::default().init::<B>(&device);
    println!("Loading transformer");
    if let Err(err) = transformer
        .load_weights(&args.transformer)
        .context("Failed to load transformer weights")
    {
        eprintln!("{err}");
        return ExitCode::FAILURE;
    }

    // Generate based on whether we have a text prompt or a pre-embedded prompt file
    let result = if let Some(prompt) = args.prompt {
        // Load tokenizer
        println!("Loading tokenizer");
        let tokenizer = match Qwen3Tokenizer::from_file(&args.tokenizer) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to load tokenizer: {e}");
                return ExitCode::FAILURE;
            }
        };

        // Load text encoder
        let mut text_encoder = Qwen3Config::default().init::<B>(&device);
        println!("Loading text encoder");
        if let Err(err) = text_encoder
            .load_weights(&args.text_encoder)
            .context("Failed to load text encoder weights")
        {
            eprintln!("{err}");
            return ExitCode::FAILURE;
        }

        println!("Generating image from prompt: \"{}\"", prompt);
        z_image::generate_from_text::<B>(
            &GenerateFromTextOpts {
                prompt,
                out_path: args.out,
                width: args.width,
                height: args.height,
            },
            &tokenizer,
            &text_encoder,
            &autoencoder,
            &transformer,
            &device,
        )
    } else {
        println!("Generating image from pre-embedded prompt");
        z_image::generate::<B>(
            &GenerateOpts {
                prompt_path: args.prompt_file,
                out_path: args.out,
                width: args.width,
                height: args.height,
            },
            &autoencoder,
            &transformer,
            &device,
        )
    };

    if let Err(err) = result {
        eprintln!("{err}");
        return ExitCode::FAILURE;
    }

    println!("Done");
    ExitCode::SUCCESS
}
