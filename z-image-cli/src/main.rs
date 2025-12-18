#![recursion_limit = "256"]

use std::{path::PathBuf, process::ExitCode};

use clap::Parser;
use rootcause::prelude::ResultExt;
use z_image::{
    GenerateOpts,
    modules::{ae::AutoEncoderConfig, transformer::ZImageModelConfig},
};

cfg_if::cfg_if! {
    if #[cfg(any(feature = "vulkan", feature = "metal"))] {
        type B = burn::backend::Wgpu;
    } else if #[cfg(feature = "rocm")] {
        type B = burn::backend::Rocm;
    } else if #[cfg(feature = "cuda")] {
        type B = burn::backend::Cuda;
    } else if #[cfg(feature = "cpu")] {
        type B = burn::backend::Cpu;
    } else if #[cfg(feature = "tch")] {
        type B = burn::backend::LibTorch;
    } else {
        compile_error!("Please select a backend by enabling the respective feature");
    }
}

#[derive(Parser)]
struct Args {
    /// The path to save the resulting image to.
    #[arg(short, long, default_value = "out.png")]
    out: PathBuf,
    /// The path to a safetensors file containing an embedding of the prompt.
    #[arg(short, long, default_value = "models/prompt.safetensors")]
    prompt_file: PathBuf,
    /// The path to the transformer model weights as a safensors or bpk file.
    #[arg(long, default_value = "models/z_image_turbo_bf16.bpk")]
    transformer: PathBuf,
    /// The path to the autoencoder model weights as a safensors or bpk file.
    #[arg(long, default_value = "models/ae.safetensors")]
    ae: PathBuf,
    /// The width of the image in pixels.
    #[arg(short = 'W', long, default_value_t = 1024)]
    width: usize,
    /// The height of the image in pixels.
    #[arg(short = 'H', long, default_value_t = 1024)]
    height: usize,
}

fn main() -> ExitCode {
    let args = Args::parse();

    #[cfg(not(feature = "tch"))]
    let device = Default::default();
    #[cfg(feature = "tch")]
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);

    let mut autoencoder = AutoEncoderConfig::flux_ae().init(&device);
    println!("Loading autoencoder");
    if let Err(err) = autoencoder
        .load_weights(args.ae)
        .context("Failed to load autoencoder weights")
    {
        eprintln!("{err}");
        return ExitCode::FAILURE;
    }

    let mut transformer = ZImageModelConfig::default().init::<B>(&device);
    println!("Loading transformer");
    if let Err(err) = transformer
        .load_weights(args.transformer)
        .context("Failed to load transformer weights")
    {
        eprintln!("{err}");
        return ExitCode::FAILURE;
    }

    println!("Generating image");
    let result = z_image::generate::<B>(
        &GenerateOpts {
            prompt_path: args.prompt_file,
            out_path: args.out,
            width: args.width,
            height: args.height,
        },
        &autoencoder,
        &transformer,
        &device,
    );
    if let Err(err) = result {
        eprintln!("{err}");
        return ExitCode::FAILURE;
    }

    println!("Done");
    ExitCode::SUCCESS
}
