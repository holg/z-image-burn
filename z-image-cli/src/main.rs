#![recursion_limit = "256"]

use std::{path::PathBuf, process::ExitCode, time::Duration};

use burn::{
    Tensor,
    prelude::Backend,
    store::{ModuleStore, SafetensorsStore},
    tensor::{DType, Distribution},
    vision::utils::{ColorDisplayOpts, ImageDimOrder, TensorDisplayOptions, save_tensor_as_image},
};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rootcause::{Report, prelude::ResultExt, report};
use z_image::{
    modules::{ae::AutoEncoderConfig, transformer::ZImageModelConfig},
    scheduler::{FlowMatchEulerDiscreteScheduler, Shift},
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
    /// The amount of sampling steps.
    #[arg(long, default_value_t = 8)]
    steps: u32,
    /// How many images to generate at once.
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    /// Whether to use dynamic shift in the scheduler.
    #[arg(long, action)]
    dynamic_shift: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    #[cfg(not(feature = "tch"))]
    let device = Default::default();
    #[cfg(feature = "tch")]
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);

    let result = generate::<B>(args, &device);
    if let Err(err) = result {
        eprintln!("{err}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

fn generate<B: Backend>(args: Args, device: &B::Device) -> Result<(), Report> {
    let batch_size = 1;
    let num_inference_steps = args.steps;
    let width = args.width;
    let height = args.height;

    let mut prompt_store = SafetensorsStore::from_file(&args.prompt_file);
    let prompt_data = prompt_store
        .get_snapshot("prompt")
        .context("Failed to read prompt file")?
        .ok_or_else(|| report!("Missing key `prompt` in prompt file"))?
        .to_data()
        .context("Failed to load prompt from prompt file")?;
    let prompt_embedding = Tensor::<B, 2>::from_data(prompt_data, device).unsqueeze_dim::<3>(0);

    let vae_scale_factor = 8;
    let vae_scale = vae_scale_factor * 2;

    if width % vae_scale != 0 || height % vae_scale != 0 {
        return Err(
            report!("Width and height must be a multiple of {vae_scale}")
                .attach(format!("width: {width}"))
                .attach(format!("height: {height}")),
        );
    }

    let latent_width = 2 * (width / vae_scale);
    let latent_height = 2 * (height / vae_scale);

    let latents_shape = [
        batch_size,
        16, // = in_channels
        latent_height,
        latent_width,
    ];
    let latents = Tensor::<B, 4>::random(latents_shape, Distribution::Normal(0., 1.), device);

    let scheduler = FlowMatchEulerDiscreteScheduler::<B>::new(
        num_inference_steps,
        1000,
        match args.dynamic_shift {
            true => Shift::dynamic(
                (latents_shape[2] / 2) * (latents_shape[3] / 2),
                Default::default(),
            ),
            false => Shift::Constant(3.0),
        },
        device,
    );

    let pb = spinner("Loading model");
    let transformer = ZImageModelConfig::default()
        .init::<B>(device)
        .with_weights(args.transformer)
        .context("Failed to load transformer weights")?;
    pb.finish();

    let pb = ProgressBar::new(num_inference_steps as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.blue.bold} {msg} {wide_bar:.blue/white.dim.dim} {pos}/{len} • {per_sec} • {eta}/{elapsed}",
        )
        .unwrap()
        .tick_strings(&[" ⠋", " ⠙", " ⠹", " ⠸", " ⠼", " ⠴", " ⠦", " ⠧", " ⠇", " ⠏", "OK"])
        .progress_chars("━╸━"),
    );
    pb.set_message("Sampling");
    let mut sampler = scheduler.sampler(&transformer, latents, prompt_embedding);
    for (i, _) in (&mut sampler).enumerate() {
        pb.set_position((i + 1) as u64);
    }
    pb.finish();
    let latents = sampler.result();

    drop(transformer);

    let pb = spinner("Loading autoencoder");
    let autoencoder = AutoEncoderConfig::flux_ae()
        .init(device)
        .with_weights(args.ae)
        .context("Failed to load autoencoder weights")?;
    pb.finish();

    let pb = spinner("Decoding image");
    let image = autoencoder.decode(latents);
    pb.finish();

    drop(autoencoder);

    let pb = spinner("Saving image");
    save_tensor_as_image(
        image.cast(DType::F32),
        TensorDisplayOptions {
            dim_order: ImageDimOrder::Nchw,
            color_opts: ColorDisplayOpts::Rgb,
            batch_opts: None,
            width_out: width,
            height_out: height,
        },
        &args.out,
    )
    .map_err(|err| report!(err.to_string()))
    .context("Failed to save image")
    .attach_with(|| format!("destination path: {}", &args.out.to_string_lossy()))?;
    pb.finish();
    Ok(())
}

fn spinner(msg: &'static str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.blue.bold} {msg}")
            .unwrap()
            .tick_strings(&[
                " ⠋", " ⠙", " ⠹", " ⠸", " ⠼", " ⠴", " ⠦", " ⠧", " ⠇", " ⠏", "OK",
            ]),
    );
    pb.enable_steady_tick(Duration::from_millis(120));
    pb.set_message(msg);
    pb
}
