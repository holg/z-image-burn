use std::path::PathBuf;

use burn::{
    Tensor,
    prelude::{Backend, ToElement},
    store::{ModuleStore, SafetensorsStore},
    tensor::{DType, Distribution, ops::FloatElem},
    vision::utils::{ColorDisplayOpts, ImageDimOrder, TensorDisplayOptions, save_tensor_as_image},
};
use rootcause::{Report, prelude::ResultExt, report};

use crate::{
    modules::{ae::AutoEncoder, transformer::ZImageModel},
    scheduler::FlowMatchEulerDiscreteScheduler,
};

pub(crate) mod compat;
mod load;
pub mod modules;
pub mod scheduler;
mod utils;

/// Options for the [generate] function.
#[derive(Debug, Clone)]
pub struct GenerateOpts {
    /// The path to a safetensors file containing the embedding of the prompt.
    pub prompt_path: PathBuf,
    /// The path to write the resulting image into.
    pub out_path: PathBuf,
    /// The width of the resulting image, in pixels.
    pub width: usize,
    /// The heihgt of the resulting image, in pixels.
    pub height: usize,
}

/// Generate an image and save it to a file.
pub fn generate<B: Backend>(
    opts: &GenerateOpts,
    autoencoder: &AutoEncoder<B>,
    transformer: &ZImageModel<B>,
    device: &B::Device,
) -> Result<(), Report> {
    let batch_size = 1;
    let num_inference_steps = 8;
    let width = opts.width;
    let height = opts.height;

    let mut prompt_store = SafetensorsStore::from_file(&opts.prompt_path);
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

    let image_seq_len = (latents_shape[2] / 2) * (latents_shape[3] / 2);
    let mu = calculate_shift(image_seq_len, 256, 4096, 0.5, 1.15);

    let mut scheduler = FlowMatchEulerDiscreteScheduler::<B>::new(1000, 3.0, false, device);
    scheduler.set_timesteps(
        Some(num_inference_steps),
        &device,
        None,
        Some(mu.into()),
        None,
    );

    let timesteps = scheduler.timesteps();
    let num_inference_steps = timesteps.dims()[0];

    let mut latents = latents;
    for (i, t) in timesteps
        .into_data()
        .as_slice::<FloatElem<B>>()
        .expect("tensor is of correct type")
        .iter()
        .enumerate()
    {
        let t = t.to_f64();
        if t == 0. && i == num_inference_steps - 1 {
            continue;
        }

        let timestep = Tensor::<B, 1>::from_floats([t], &device).expand([latents_shape[0]]);
        let timestep: Tensor<B, 1> = timestep / 1000.;

        let noise_pred = transformer.forward(latents.clone(), timestep, prompt_embedding.clone());
        latents = scheduler.step(-noise_pred, t, latents);
    }

    let image = autoencoder.decode(latents);
    save_tensor_as_image(
        image.cast(DType::F32),
        TensorDisplayOptions {
            dim_order: ImageDimOrder::Nchw,
            color_opts: ColorDisplayOpts::Rgb,
            batch_opts: None,
            width_out: width,
            height_out: height,
        },
        &opts.out_path,
    )
    .map_err(|err| report!(err.to_string()))
    .context("Failed to save image")
    .attach_with(|| format!("destination path: {}", &opts.out_path.to_string_lossy()))?;
    Ok(())
}

/// Linearly maps `image_seq_len` into the interval [`base_shift`, `max_shift`] with respect to the
/// reference range [`base_seq_len`, `max_seq_len`].
fn calculate_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f32,
    max_shift: f32,
) -> f32 {
    let m = (max_shift - base_shift) / (max_seq_len - base_seq_len) as f32;
    let b = base_shift - m * base_seq_len as f32;
    image_seq_len as f32 * m + b
}
