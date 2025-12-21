mod attention;
mod feed_forward;
mod final_layer;
mod layer_norm;
mod rope;
mod timestep_embedder;
mod transformer_block;
mod utils;

use burn::{
    Tensor,
    config::Config,
    module::{Ignored, Module, Param},
    nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::{Int, ops::PadMode, s},
};

use crate::{
    modules::transformer::{
        final_layer::{FinalLayer, FinalLayerConfig},
        rope::RopeEmbedder,
        timestep_embedder::{TimestepEmbedder, TimestepEmbedderConfig},
        transformer_block::{ZImageTransformerBlock, ZImageTransformerBlockConfig},
        utils::pad_to_patch_size,
    },
    utils::effective_dtype,
};

const ADALN_EMBED_DIM: usize = 256;

/// Configuration for [ZImageModel]. By default the standard configuration for Z-Image is used.
#[derive(Config, Debug)]
pub struct ZImageModelConfig {
    #[config(default = 16)]
    in_channels: usize,
    #[config(default = 3840)]
    dim: usize,
    #[config(default = 30)]
    n_layers: usize,
    #[config(default = 2)]
    n_refiner_layers: usize,
    #[config(default = 30)]
    n_heads: usize,
    #[config(default = 30)]
    n_kv_heads: usize,
    #[config(default = 1e-5)]
    norm_eps: f64,
    #[config(default = true)]
    qk_norm: bool,
    #[config(default = 2560)]
    cap_feat_dim: usize,
    #[config(default = 256.0)]
    rope_theta: f64,
    #[config(default = 1000.0)]
    time_scale: f64,
    #[config(default = "vec![32, 48, 48]")]
    axes_dims: Vec<usize>,
    #[config(default = "vec![1536, 512, 512]")]
    axes_lens: Vec<usize>,
}

impl Default for ZImageModelConfig {
    /// Get the default configuration used for Z-Image.
    fn default() -> Self {
        ZImageModelConfig::new()
    }
}

impl ZImageModelConfig {
    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ZImageModel<B> {
        let patch_size = 2;
        let f_patch_size = 1;

        let out_channels = self.in_channels;

        ZImageModel {
            time_scale: Ignored(self.time_scale),
            out_channels: Ignored(out_channels),
            patch_size: Ignored(patch_size),
            x_embedder: LinearConfig::new(
                f_patch_size * patch_size * patch_size * self.in_channels,
                self.dim,
            )
            .with_bias(true)
            .init(device),
            final_layer: FinalLayerConfig::new(
                self.dim,
                patch_size * patch_size * f_patch_size * out_channels,
            )
            .init(device),
            noise_refiner: (0..self.n_refiner_layers)
                .map(|layer_id| {
                    ZImageTransformerBlockConfig::new(
                        layer_id,
                        self.dim,
                        self.n_heads,
                        self.n_kv_heads,
                        self.norm_eps,
                        self.qk_norm,
                    )
                    .with_modulation(true)
                    .init(device)
                })
                .collect(),
            context_refiner: (0..self.n_refiner_layers)
                .map(|layer_id| {
                    ZImageTransformerBlockConfig::new(
                        layer_id,
                        self.dim,
                        self.n_heads,
                        self.n_kv_heads,
                        self.norm_eps,
                        self.qk_norm,
                    )
                    .with_modulation(false)
                    .init(device)
                })
                .collect(),
            t_embedder: TimestepEmbedderConfig::new(self.dim.min(ADALN_EMBED_DIM))
                .with_mid_size(Some(1024))
                .init(device),
            cap_embedder_0: RmsNormConfig::new(self.cap_feat_dim)
                .with_epsilon(self.norm_eps)
                .init(device),
            cap_embedder_1: LinearConfig::new(self.cap_feat_dim, self.dim)
                .with_bias(true)
                .init(device),

            x_pad_token: Param::from_tensor(Tensor::empty([1, self.dim], device)),
            cap_pad_token: Param::from_tensor(Tensor::empty([1, self.dim], device)),
            layers: (0..self.n_layers)
                .map(|layer_id| {
                    ZImageTransformerBlockConfig::new(
                        layer_id,
                        self.dim,
                        self.n_heads,
                        self.n_kv_heads,
                        self.norm_eps,
                        self.qk_norm,
                    )
                    .with_modulation(true)
                    .init(device)
                })
                .collect(),
            rope_embedder: RopeEmbedder::new(self.dim, self.rope_theta, self.axes_dims.clone()),
        }
    }
}

/// The main part of the Z-Image model, the 'S3-DiT'. Should be constructed using
/// [ZImageModelConfig].
#[derive(Module, Debug)]
pub struct ZImageModel<B: Backend> {
    time_scale: Ignored<f64>,
    out_channels: Ignored<usize>,
    patch_size: Ignored<usize>,

    x_embedder: Linear<B>,
    final_layer: FinalLayer<B>,

    noise_refiner: Vec<ZImageTransformerBlock<B>>,
    context_refiner: Vec<ZImageTransformerBlock<B>>,

    t_embedder: TimestepEmbedder<B>,
    cap_embedder_0: RmsNorm<B>,
    cap_embedder_1: Linear<B>,

    x_pad_token: Param<Tensor<B, 2>>,
    cap_pad_token: Param<Tensor<B, 2>>,

    layers: Vec<ZImageTransformerBlock<B>>,
    rope_embedder: RopeEmbedder<B>,
}

impl<B: Backend> ZImageModel<B> {
    /// Perform one inference step (the equivalent of denoising in diffusion models).
    ///
    /// # Arguments
    ///
    /// - `latents`: A tensor of shape [batch_size, channels, height, width].
    /// - `timestep`: A tensor of shape [batch_size] denoting the current timestep.
    /// - `cap_feats`: A tensor of shape [batch_size, n, cap_feat_dim] obtained by embedding the
    ///   prompt(s) with a text encoder.
    ///
    /// # Returns
    ///
    /// A tensor of shape [batch_size, channels, height, width] with the resulting velocity field
    /// used to push the samples closer to their targets.
    pub fn forward(
        &self,
        latents: Tensor<B, 4>,
        timestep: Tensor<B, 1>,
        cap_feats: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let output_dtype = latents.dtype();
        let model_dtype = effective_dtype(self.cap_pad_token.dtype());

        let latents = latents.cast(model_dtype);
        let timestep = timestep.cast(model_dtype);
        let cap_feats = cap_feats.cast(model_dtype);

        let t = 1.0 - timestep;
        let [_bs, _c, h, w] = latents.dims();
        let x = pad_to_patch_size(latents, [*self.patch_size, *self.patch_size]);

        let t = self.t_embedder.forward(t * self.time_scale.0);
        let adaln_input = t.clone();

        let cap_feats = self.cap_embedder_0.forward(cap_feats);
        let cap_feats = self.cap_embedder_1.forward(cap_feats);

        let (mut x, img_size, cap_size, freqs_cis) = self.patchify_and_embed(x, cap_feats, t);

        for layer in &self.layers {
            x = layer.forward(x, None, freqs_cis.clone(), Some(adaln_input.clone()));
        }

        let x = self.final_layer.forward(x, adaln_input);
        let x = self.unpatchify(x, img_size, cap_size);
        let x = x.slice(s![.., .., ..h, ..w]);
        x.cast(output_dtype)
    }

    fn patchify_and_embed(
        &self,
        x: Tensor<B, 4>,
        cap_feats: Tensor<B, 3>,
        t: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Vec<[usize; 2]>, Vec<usize>, Tensor<B, 6>) {
        let device = x.device();

        let bsz = x.dims()[0];
        let p_h = *self.patch_size;
        let p_w = *self.patch_size;

        const PAD_TOKENS_MULTIPLE: i64 = 32;
        let pad_extra = (-(cap_feats.dims()[1] as i64)).rem_euclid(PAD_TOKENS_MULTIPLE) as usize;
        // Workaround for pad_extra=0, as burn's behaviour seems to be inconsistent with pytorch
        // here.
        let cap_feats = match pad_extra {
            0 => cap_feats,
            1.. => {
                let cap_feats_dims = cap_feats.dims();
                Tensor::cat(
                    vec![
                        cap_feats,
                        (*self.cap_pad_token)
                            .clone()
                            .unsqueeze_dim::<3>(0)
                            .repeat(&[cap_feats_dims[0], pad_extra, 1]),
                    ],
                    1,
                )
            }
        };

        let cap_pos_ids = Tensor::<B, 3, Int>::zeros([bsz, cap_feats.dims()[1], 3], &device);
        let cap_pos_ids = cap_pos_ids.slice_assign(
            s![.., .., 0],
            (Tensor::arange(1..((cap_feats.dims()[1] + 1) as i64), &device))
                .unsqueeze_dim::<2>(1)
                .unsqueeze_dim::<3>(0),
        );

        let [b, c, h, w] = x.dims();
        let x = self.x_embedder.forward(
            x.reshape([b, c, h / p_h, p_h, w / p_w, p_w])
                .permute([0, 2, 4, 3, 5, 1])
                .flatten::<4>(3, -1)
                .flatten::<3>(1, 2),
        );

        let h_tokens = h / p_h;
        let w_tokens = w / p_w;

        // Note: h_scale=1.0, w_scale=1.0, h_start=0, w_start=0 are the defaults
        // We use Int tensors directly to avoid candle-metal I64 affine issues
        let x_pos_ids = Tensor::<B, 3, Int>::zeros([bsz, x.dims()[1], 3], &device)
            .slice_assign(
                s![.., .., 0],
                Tensor::full([bsz, x.dims()[1], 1], (cap_feats.dims()[1] + 1) as i64, &device),
            )
            .slice_assign(
                s![.., .., 1],
                Tensor::arange(0..(h_tokens as i64), &device)
                    .reshape([-1, 1])
                    .repeat(&[1, w_tokens])
                    .flatten::<1>(0, -1)
                    .unsqueeze_dim::<2>(1)
                    .unsqueeze::<3>(),
            )
            .slice_assign(
                s![.., .., 2],
                Tensor::arange(0..(w_tokens as i64), &device)
                    .reshape([1, -1])
                    .repeat(&[h_tokens, 1])
                    .flatten::<1>(0, -1)
                    .unsqueeze_dim::<2>(1)
                    .unsqueeze::<3>(),
            );

        let pad_extra = (-(x.dims()[1] as i64)).rem_euclid(PAD_TOKENS_MULTIPLE) as usize;
        let x_dims = x.dims();
        let x = match pad_extra {
            0 => x,
            1.. => Tensor::cat(
                vec![
                    x,
                    (*self.cap_pad_token)
                        .clone()
                        .unsqueeze_dim::<3>(0)
                        .repeat(&[x_dims[0], pad_extra, 1]),
                ],
                1,
            ),
        };
        let x_pos_ids = x_pos_ids.pad((0, 0, 0, pad_extra), PadMode::Constant(0.));

        let freqs_cis = self
            .rope_embedder
            .forward(Tensor::cat(vec![cap_pos_ids.clone(), x_pos_ids], 1))
            .movedim(1, 2);

        let mut cap_feats = cap_feats;
        let freqs_cis_slice = freqs_cis.clone().slice(s![.., ..cap_pos_ids.dims()[1]]);
        for layer in &self.context_refiner {
            cap_feats = layer.forward(cap_feats, None, freqs_cis_slice.clone(), None);
        }

        let mut x = x;
        let freqs_cis_slice = freqs_cis.clone().slice(s![.., cap_pos_ids.dims()[1]..]);
        for layer in &self.noise_refiner {
            x = layer.forward(x, None, freqs_cis_slice.clone(), Some(t.clone()));
        }

        let padded_full_embed = Tensor::cat(vec![cap_feats.clone(), x], 1);
        let img_sizes = vec![[h, w]; bsz];
        let l_effective_cap_len = vec![cap_feats.dims()[1]; bsz];

        (padded_full_embed, img_sizes, l_effective_cap_len, freqs_cis)
    }

    fn unpatchify(
        &self,
        x: Tensor<B, 3>,
        img_size: Vec<[usize; 2]>,
        cap_size: Vec<usize>,
    ) -> Tensor<B, 4> {
        let p_h = *self.patch_size;
        let p_w = *self.patch_size;

        let mut imgs = Vec::with_capacity(x.dims()[0]);
        for i in 0..(x.dims()[0]) {
            let [h, w] = img_size[i];
            let begin = cap_size[i];
            let end = begin + (h / p_h) * (w / p_w);

            imgs.push(
                x.clone()
                    .slice(s![i, begin..end])
                    .reshape([h / p_h, w / p_w, p_h, p_w, *self.out_channels])
                    .permute([4, 0, 2, 1, 3])
                    .flatten::<4>(3, 4)
                    .flatten::<3>(1, 2),
            );
        }

        Tensor::stack(imgs, 0)
    }
}
