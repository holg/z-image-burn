mod layer_norm;
mod rope;
mod utils;

use burn::{
    Tensor,
    config::Config,
    module::{Ignored, Module, Param},
    nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::{Bool, Int, activation::silu, module::attention, ops::PadMode, s},
};

use crate::modules::transformer::{
    layer_norm::LayerNormNoAffine,
    rope::{RopeEmbedder, apply_rotary_emb},
    utils::{clamp_fp16, modulate, pad_to_patch_size},
};

const ADALN_EMBED_DIM: usize = 256;

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
    axes_dims: Vec<usize>,
    axes_lens: Vec<usize>,
}

impl ZImageModelConfig {
    pub fn default() -> Self {
        ZImageModelConfig::new(vec![32, 48, 48], vec![1536, 512, 512])
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> ZImageModel<B> {
        let patch_size = 2;
        let f_patch_size = 1;

        let out_channels = self.in_channels;

        ZImageModel {
            time_scale: Ignored(self.time_scale),
            out_channels: Ignored(out_channels),
            patch_size: Ignored(patch_size),
            all_x_embedder: AllXEmbedder {
                r2_1: LinearConfig::new(
                    f_patch_size * patch_size * patch_size * self.in_channels,
                    self.dim,
                )
                .with_bias(true)
                .init(device),
            },
            all_final_layer: AllFinalLayer {
                r2_1: FinalLayerConfig::new(
                    self.dim,
                    patch_size * patch_size * f_patch_size * out_channels,
                )
                .init(device),
            },
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

#[derive(Module, Debug)]
pub struct ZImageModel<B: Backend> {
    time_scale: Ignored<f64>,
    out_channels: Ignored<usize>,
    patch_size: Ignored<usize>,

    all_x_embedder: AllXEmbedder<B>,
    all_final_layer: AllFinalLayer<B>,

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
    pub fn forward(
        &self,
        // Latents: [batch_size, channels, height, width]
        x: Tensor<B, 4>,
        timesteps: Tensor<B, 1>,
        cap_feats: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let t = 1.0 - timesteps;
        let [_bs, _c, h, w] = x.dims();
        let x = pad_to_patch_size(x, [*self.patch_size, *self.patch_size]);

        let t = self.t_embedder.forward(t * self.time_scale.0);
        let adaln_input = t.clone();

        let cap_feats = self.cap_embedder_0.forward(cap_feats);
        let cap_feats = self.cap_embedder_1.forward(cap_feats);

        let (mut x, img_size, cap_size, freqs_cis) = self.patchify_and_embed(x, cap_feats, t);

        for layer in &self.layers {
            x = layer.forward(x, None, freqs_cis.clone(), Some(adaln_input.clone()));
        }

        let x = self.all_final_layer.r2_1.forward(x, adaln_input);
        let x = self.unpatchify(x, img_size, cap_size);
        let x = x.slice(s![.., .., ..h, ..w]);
        x
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
            (Tensor::arange(0..(cap_feats.dims()[1] as i64), &device) + 1.0)
                .unsqueeze_dim::<2>(1)
                .unsqueeze_dim::<3>(0),
        );

        let [b, c, h, w] = x.dims();
        let x = self.all_x_embedder.r2_1.forward(
            x.reshape([b, c, h / p_h, p_h, w / p_w, p_w])
                .permute([0, 2, 4, 3, 5, 1])
                .flatten::<4>(3, -1)
                .flatten::<3>(1, 2),
        );

        let h_scale = 1.0;
        let w_scale = 1.0;
        let h_start = 0;
        let w_start = 0;

        let h_tokens = h / p_h;
        let w_tokens = w / p_w;

        let x_pos_ids = Tensor::zeros([bsz, x.dims()[1], 3], &device)
            .slice_fill(s![.., .., 0], (cap_feats.dims()[1] + 1) as f32)
            .slice_assign(
                s![.., .., 1],
                (Tensor::arange(0..(h_tokens as i64), &device) * h_scale + h_start)
                    .reshape([-1, 1])
                    .repeat(&[1, w_tokens])
                    .flatten::<1>(0, -1)
                    .unsqueeze_dim::<2>(1)
                    .unsqueeze::<3>(),
            )
            .slice_assign(
                s![.., .., 2],
                (Tensor::arange(0..(w_tokens as i64), &device) * w_scale + w_start)
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

#[derive(Module, Debug)]
struct AllXEmbedder<B: Backend> {
    r2_1: Linear<B>,
}

#[derive(Module, Debug)]
struct AllFinalLayer<B: Backend> {
    r2_1: FinalLayer<B>,
}

#[derive(Config, Debug)]
struct FinalLayerConfig {
    hidden_size: usize,
    out_channels: usize,
}

impl FinalLayerConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> FinalLayer<B> {
        FinalLayer {
            norm_final: LayerNormNoAffine::new(1e-6),
            linear: LinearConfig::new(self.hidden_size, self.out_channels)
                .with_bias(true)
                .init(device),
            adaln_modulation: LinearConfig::new(
                self.hidden_size.min(ADALN_EMBED_DIM),
                self.hidden_size,
            )
            .with_bias(true)
            .init(device),
        }
    }
}

#[derive(Module, Debug)]
struct FinalLayer<B: Backend> {
    norm_final: LayerNormNoAffine<B>,
    linear: Linear<B>,
    adaln_modulation: Linear<B>,
}

impl<B: Backend> FinalLayer<B> {
    fn forward(&self, x: Tensor<B, 3>, c: Tensor<B, 2>) -> Tensor<B, 3> {
        let scale = self.adaln_modulation.forward(silu(c));
        let x = modulate(self.norm_final.forward(x), scale);
        let x = self.linear.forward(x);
        x
    }
}

#[derive(Config, Debug)]
struct ZImageTransformerBlockConfig {
    layer_id: usize,
    dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    norm_eps: f64,
    qk_norm: bool,
    /// Whether to use Adaptive Layer Normalization Modulation.
    #[config(default = true)]
    modulation: bool,
}

impl ZImageTransformerBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ZImageTransformerBlock<B> {
        ZImageTransformerBlock {
            attention: ZImageAttentionConfig::new(self.dim, self.n_heads, self.n_kv_heads)
                .with_qk_norm(self.qk_norm)
                .with_eps(self.norm_eps)
                .init(device),
            feed_forward: FeedForwardConfig::new(self.dim, (self.dim as f32 / 3. * 8.) as usize)
                .init(device),
            attention_norm1: RmsNormConfig::new(self.dim)
                .with_epsilon(self.norm_eps)
                .init(device),
            ffn_norm1: RmsNormConfig::new(self.dim)
                .with_epsilon(self.norm_eps)
                .init(device),
            attention_norm2: RmsNormConfig::new(self.dim)
                .with_epsilon(self.norm_eps)
                .init(device),
            ffn_norm2: RmsNormConfig::new(self.dim)
                .with_epsilon(self.norm_eps)
                .init(device),
            adaln_modulation: self.modulation.then(|| {
                LinearConfig::new(self.dim.min(ADALN_EMBED_DIM), 4 * self.dim)
                    .with_bias(true)
                    .init(device)
            }),
        }
    }
}

#[derive(Module, Debug)]
struct ZImageTransformerBlock<B: Backend> {
    attention: ZImageAttention<B>,
    feed_forward: FeedForward<B>,
    attention_norm1: RmsNorm<B>,
    ffn_norm1: RmsNorm<B>,
    attention_norm2: RmsNorm<B>,
    ffn_norm2: RmsNorm<B>,
    adaln_modulation: Option<Linear<B>>,
}

impl<B: Backend> ZImageTransformerBlock<B> {
    fn forward(
        &self,
        x: Tensor<B, 3>,
        attn_mask: Option<Tensor<B, 2, Bool>>,
        x_freqs_cis: Tensor<B, 6>,
        adaln_input: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        if let Some(adaln_modulation) = &self.adaln_modulation {
            let adaln_input =
                adaln_input.expect("adaln_input should be Some when modulation is enabled");

            let [scale_msa, gate_msa, scale_mlp, gate_mlp] = adaln_modulation
                .forward(adaln_input)
                .chunk(4, 1)
                .try_into()
                .expect("result should be a vector of size 4");

            let x = x.clone()
                + gate_msa.unsqueeze_dim::<3>(1).tanh()
                    * self
                        .attention_norm2
                        .forward(clamp_fp16(self.attention.forward(
                            modulate(self.attention_norm1.forward(x), scale_msa),
                            attn_mask,
                            Some(x_freqs_cis),
                        )));
            let x = x.clone()
                + gate_mlp.unsqueeze_dim::<3>(1).tanh()
                    * self.ffn_norm2.forward(clamp_fp16(
                        self.feed_forward
                            .forward(modulate(self.ffn_norm1.forward(x), scale_mlp)),
                    ));
            x
        } else {
            assert!(adaln_input.is_none());

            let x = x.clone()
                + self
                    .attention_norm2
                    .forward(clamp_fp16(self.attention.forward(
                        self.attention_norm1.forward(x),
                        attn_mask,
                        Some(x_freqs_cis),
                    )));
            let x = x.clone()
                + self
                    .ffn_norm2
                    .forward(self.feed_forward.forward(self.ffn_norm1.forward(x)));
            x
        }
    }
}

#[derive(Config, Debug)]
struct ZImageAttentionConfig {
    dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    #[config(default = true)]
    qk_norm: bool,
    #[config(default = 1e-5)]
    eps: f64,
}

impl ZImageAttentionConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ZImageAttention<B> {
        let head_dim = self.dim / self.n_heads;

        ZImageAttention {
            n_heads: Ignored(self.n_heads),
            head_dim: Ignored(head_dim),
            n_kv_heads: Ignored(self.n_heads),
            qkv: LinearConfig::new(self.dim, (self.n_heads + self.n_kv_heads * 2) * head_dim)
                .with_bias(false)
                .init(device),
            to_out: LinearConfig::new(self.n_heads * head_dim, self.dim)
                .with_bias(false)
                .init(device),
            q_norm: self.qk_norm.then(|| {
                RmsNormConfig::new(head_dim)
                    .with_epsilon(self.eps)
                    .init(device)
            }),
            k_norm: self.qk_norm.then(|| {
                RmsNormConfig::new(head_dim)
                    .with_epsilon(self.eps)
                    .init(device)
            }),
        }
    }
}

#[derive(Module, Debug)]
struct ZImageAttention<B: Backend> {
    n_heads: Ignored<usize>,
    n_kv_heads: Ignored<usize>,
    head_dim: Ignored<usize>,

    qkv: Linear<B>,
    to_out: Linear<B>,

    q_norm: Option<RmsNorm<B>>,
    k_norm: Option<RmsNorm<B>>,
}

impl<B: Backend> ZImageAttention<B> {
    fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        freqs_cis: Option<Tensor<B, 6>>,
    ) -> Tensor<B, 3> {
        let [bsz, seqlen, ..] = hidden_states.dims();

        let [query, key, value] = self
            .qkv
            .forward(hidden_states)
            .split_with_sizes(
                vec![
                    self.n_heads.0 * self.head_dim.0,
                    self.n_kv_heads.0 * self.head_dim.0,
                    self.n_kv_heads.0 * self.head_dim.0,
                ],
                2,
            )
            .try_into()
            .unwrap();

        let query = query.reshape([bsz, seqlen, *self.n_heads, *self.head_dim]);
        let key = key.reshape([bsz, seqlen, *self.n_kv_heads, *self.head_dim]);
        let value = value.reshape([bsz, seqlen, *self.n_kv_heads, *self.head_dim]);

        let query = match &self.q_norm {
            Some(q_norm) => q_norm.forward(query),
            None => query,
        };
        let key = match &self.k_norm {
            Some(k_norm) => k_norm.forward(key),
            None => key,
        };

        let (query, key) = if let Some(freqs_cis) = freqs_cis {
            (
                apply_rotary_emb(query, freqs_cis.clone()),
                apply_rotary_emb(key, freqs_cis),
            )
        } else {
            (query, key)
        };

        let n_rep = *self.n_heads / *self.n_kv_heads;
        let (key, value) = if n_rep >= 1 {
            (
                key.unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
                value
                    .unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
            )
        } else {
            (key, value)
        };

        let hidden_states = attention(
            query.movedim(1, 2),
            key.movedim(1, 2),
            value.movedim(1, 2),
            attention_mask.map(|attention_mask| attention_mask.unsqueeze_dims(&[1, 2])),
        );
        let hidden_states = hidden_states.movedim(1, 2).reshape([
            bsz as i64,
            -1,
            (self.n_heads.0 * self.head_dim.0) as i64,
        ]);

        self.to_out.forward(hidden_states)
    }
}

#[derive(Config, Debug)]
struct FeedForwardConfig {
    dim: usize,
    hidden_dim: usize,
}

impl FeedForwardConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        FeedForward {
            w1: LinearConfig::new(self.dim, self.hidden_dim)
                .with_bias(false)
                .init(device),
            w2: LinearConfig::new(self.hidden_dim, self.dim)
                .with_bias(false)
                .init(device),
            w3: LinearConfig::new(self.dim, self.hidden_dim)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
struct FeedForward<B: Backend> {
    w1: Linear<B>,
    w2: Linear<B>,
    w3: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.w2
            .forward(self.forward_silu_gating(self.w1.forward(x.clone()), self.w3.forward(x)))
    }

    fn forward_silu_gating(&self, x1: Tensor<B, 3>, x3: Tensor<B, 3>) -> Tensor<B, 3> {
        clamp_fp16(silu(x1) * x3)
    }
}

#[derive(Config, Debug)]
pub struct TimestepEmbedderConfig {
    out_size: usize,
    mid_size: Option<usize>,
    #[config(default = 256)]
    frequency_embedding_size: usize,
}

impl TimestepEmbedderConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> TimestepEmbedder<B> {
        let mid_size = self.mid_size.unwrap_or(self.out_size);
        TimestepEmbedder {
            frequency_embedding_size: Ignored(self.frequency_embedding_size),
            mlp_1: LinearConfig::new(self.frequency_embedding_size, mid_size)
                .with_bias(true)
                .init(device),
            mlp_2: LinearConfig::new(mid_size, self.out_size)
                .with_bias(true)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
struct TimestepEmbedder<B: Backend> {
    frequency_embedding_size: Ignored<usize>,
    mlp_1: Linear<B>,
    mlp_2: Linear<B>,
}

impl<B: Backend> TimestepEmbedder<B> {
    fn forward(&self, t: Tensor<B, 1>) -> Tensor<B, 2> {
        let t_freq = Self::timestep_embedding(t, self.frequency_embedding_size.0, 10000);

        let t_emb = self.mlp_1.forward(t_freq);
        let t_emb = silu(t_emb);
        self.mlp_2.forward(t_emb)
    }

    fn timestep_embedding(t: Tensor<B, 1>, dim: usize, max_period: usize) -> Tensor<B, 2> {
        let device = t.device();

        let half = (dim / 2) as i64;
        let freqs = Tensor::<B, 1>::exp(
            -(max_period as f32).ln() * Tensor::arange(0..half, &device).float() / half,
        );
        let args = t.unsqueeze_dim::<2>(1) * freqs.unsqueeze_dim(0);
        let embedding = Tensor::cat(vec![args.clone().cos(), args.sin()], 1);

        if dim % 2 != 0 {
            let embedding_shape = embedding.shape();
            return Tensor::cat(
                vec![
                    embedding,
                    Tensor::zeros(embedding_shape.slice(&s![.., ..1]).unwrap(), &device),
                ],
                1,
            );
        }

        embedding
    }
}
