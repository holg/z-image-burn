use std::path::PathBuf;

use burn::{
    Tensor,
    prelude::{Backend, ToElement},
    store::{ModuleStore, SafetensorsStore},
    tensor::{Bool, DType, Distribution, Int},
    vision::utils::{ColorDisplayOpts, ImageDimOrder, TensorDisplayOptions, save_tensor_as_image},
};
use rootcause::{Report, prelude::ResultExt, report};
use qwen3_burn::{Qwen3Model, Qwen3Tokenizer};

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
    /// The height of the resulting image, in pixels.
    pub height: usize,
}

/// Options for generating from a text prompt directly.
#[derive(Debug, Clone)]
pub struct GenerateFromTextOpts {
    /// The text prompt to generate an image from.
    pub prompt: String,
    /// The path to write the resulting image into.
    pub out_path: PathBuf,
    /// The width of the resulting image, in pixels.
    pub width: usize,
    /// The height of the resulting image, in pixels.
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
        .as_slice::<f32>()
        .expect("timesteps should be F32")
        .iter()
        .enumerate()
    {
        let t = *t;
        if t == 0. && i == num_inference_steps - 1 {
            continue;
        }

        let timestep = Tensor::<B, 1>::from_floats([t], &device).expand([latents_shape[0]]);
        let timestep: Tensor<B, 1> = timestep / 1000.;

        let noise_pred = transformer.forward(latents.clone(), timestep, prompt_embedding.clone());
        latents = scheduler.step(-noise_pred, t, latents);
    }

    // Cast latents to F32 for autoencoder (which has F32 weights)
    let latents_f32 = latents.cast(DType::F32);
    let image = autoencoder.decode(latents_f32);
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

/// Generate an image from a text prompt and save it to a file.
///
/// This function uses the Qwen3 text encoder to convert the prompt to embeddings,
/// then generates an image using the Z-Image transformer.
pub fn generate_from_text<B: Backend>(
    opts: &GenerateFromTextOpts,
    tokenizer: &Qwen3Tokenizer,
    text_encoder: &Qwen3Model<B>,
    autoencoder: &AutoEncoder<B>,
    transformer: &ZImageModel<B>,
    device: &B::Device,
) -> Result<(), Report> {
    // Tokenize the prompt with chat template
    // Z-Image text encoder expects the Qwen3 chat format
    let (input_ids_vec, attention_mask_vec) = tokenizer
        .encode_prompt(&opts.prompt)
        .map_err(|e| report!("{e}"))?;

    let seq_len = input_ids_vec.len();

    // Convert to tensors
    let input_ids = Tensor::<B, 1, Int>::from_data(input_ids_vec.as_slice(), device)
        .reshape([1, seq_len]);
    let attention_mask = Tensor::<B, 1>::from_data(
        attention_mask_vec.iter().map(|&b| if b { 1.0f32 } else { 0.0f32 }).collect::<Vec<_>>().as_slice(),
        device,
    )
    .greater_elem(0.5)
    .reshape([1, seq_len]);

    // Get text embeddings from the encoder
    let prompt_embedding = text_encoder.encode(input_ids, attention_mask.clone());
    eprintln!("[z-image] Text encoder output shape: {:?}", prompt_embedding.dims());

    // Extract only valid (non-padded) tokens using the attention mask
    // The Python code does: prompt_embed[prompt_mask]
    // We'll use gather/select based on the mask
    let prompt_embedding = extract_valid_embeddings(prompt_embedding, attention_mask);
    eprintln!("[z-image] After extracting valid embeddings: {:?}", prompt_embedding.dims());


    // Now generate using the internal function
    generate_with_embedding(
        prompt_embedding,
        &opts.out_path,
        opts.width,
        opts.height,
        autoencoder,
        transformer,
        device,
    )
}

/// Extract valid (non-padded) embeddings based on attention mask.
fn extract_valid_embeddings<B: Backend>(
    embeddings: Tensor<B, 3>,
    attention_mask: Tensor<B, 2, Bool>,
) -> Tensor<B, 3> {
    // For batch size 1, we can simplify this
    let [_batch, seq_len, hidden_dim] = embeddings.dims();

    // Get mask as float and find valid positions
    let mask_float = attention_mask.float();
    let valid_count = mask_float.clone().sum().into_scalar().to_f32() as usize;

    if valid_count == seq_len {
        // All tokens are valid, return as-is
        return embeddings;
    }

    // Mask and gather valid embeddings
    // Expand mask to [batch, seq, hidden_dim]
    let mask_expanded = mask_float.unsqueeze_dim::<3>(2).repeat(&[1, 1, hidden_dim]);

    // Zero out padded positions and slice to valid count
    let masked = embeddings * mask_expanded;

    // For simplicity, just slice to valid_count (assumes padding is at the end)
    masked.slice([0..1, 0..valid_count, 0..hidden_dim])
}

/// Internal function to generate an image from a pre-computed embedding.
fn generate_with_embedding<B: Backend>(
    prompt_embedding: Tensor<B, 3>,
    out_path: &PathBuf,
    width: usize,
    height: usize,
    autoencoder: &AutoEncoder<B>,
    transformer: &ZImageModel<B>,
    device: &B::Device,
) -> Result<(), Report> {
    let batch_size = 1;
    let num_inference_steps = 8;

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
        device,
        None,
        Some(mu.into()),
        None,
    );

    let timesteps = scheduler.timesteps();
    let num_inference_steps = timesteps.dims()[0];

    let mut latents = latents;
    for (i, t) in timesteps
        .into_data()
        .as_slice::<f32>()
        .expect("timesteps should be F32")
        .iter()
        .enumerate()
    {
        let t = *t;
        if t == 0. && i == num_inference_steps - 1 {
            continue;
        }

        let timestep = Tensor::<B, 1>::from_floats([t], device).expand([latents_shape[0]]);
        let timestep: Tensor<B, 1> = timestep / 1000.;

        let noise_pred = transformer.forward(latents.clone(), timestep, prompt_embedding.clone());
        latents = scheduler.step(-noise_pred, t, latents);
    }

    // Cast latents to F32 for autoencoder (which has F32 weights)
    let latents_f32 = latents.cast(DType::F32);
    let image = autoencoder.decode(latents_f32);
    save_tensor_as_image(
        image.cast(DType::F32),
        TensorDisplayOptions {
            dim_order: ImageDimOrder::Nchw,
            color_opts: ColorDisplayOpts::Rgb,
            batch_opts: None,
            width_out: width,
            height_out: height,
        },
        out_path,
    )
    .map_err(|err| report!(err.to_string()))
    .context("Failed to save image")
    .attach_with(|| format!("destination path: {}", &out_path.to_string_lossy()))?;
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
