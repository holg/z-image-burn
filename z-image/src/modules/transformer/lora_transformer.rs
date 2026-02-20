//! LoRA-wrapped Z-Image transformer model for fine-tuning.
//!
//! This module provides LoRA variants of the Z-Image model components where
//! Linear layers in attention and feed-forward blocks are replaced with
//! [LoraLinear](crate::modules::lora::LoraLinear).

use burn::{
    Tensor,
    config::Config,
    module::{Ignored, Module, Param},
    nn::{Linear, RmsNorm},
    prelude::Backend,
    tensor::{Bool, Int, ops::PadMode, s},
};

use crate::modules::{
    lora::LoraLinear,
    transformer::{
        attention::{ZImageAttention, get_attention_slice_size, get_attention_seq_chunk_size},
        feed_forward::FeedForward,
        final_layer::FinalLayer,
        rope::{RopeEmbedder, apply_rotary_emb},
        timestep_embedder::TimestepEmbedder,
        transformer_block::ZImageTransformerBlock,
        utils::{clamp_fp16, modulate, pad_to_patch_size},
        ZImageModel,
    },
};
use burn::tensor::module::attention;

/// Configuration for which layers to apply LoRA to and with what parameters.
#[derive(Config, Debug)]
pub struct LoraConfig {
    /// LoRA rank (low-rank dimension). Higher = more capacity, more VRAM.
    #[config(default = 16)]
    pub rank: usize,
    /// LoRA alpha scaling factor. The actual scaling is alpha/rank.
    #[config(default = 16.0)]
    pub alpha: f32,
    /// Apply LoRA to attention qkv and to_out projections.
    #[config(default = true)]
    pub target_attention: bool,
    /// Apply LoRA to feed-forward w1, w2, w3 projections.
    #[config(default = true)]
    pub target_feed_forward: bool,
    /// Apply LoRA to noise and context refiner blocks.
    #[config(default = true)]
    pub target_refiners: bool,
}

// --- LoRA Attention ---

#[derive(Module, Debug)]
pub struct ZImageAttentionLora<B: Backend> {
    n_heads: Ignored<usize>,
    n_kv_heads: Ignored<usize>,
    head_dim: Ignored<usize>,
    pub qkv: LoraLinear<B>,
    pub to_out: LoraLinear<B>,
    q_norm: Option<RmsNorm<B>>,
    k_norm: Option<RmsNorm<B>>,
}

impl<B: Backend> ZImageAttentionLora<B> {
    /// Build from an existing attention module. If `apply_lora` is true,
    /// wraps qkv and to_out with LoRA; otherwise freezes them.
    pub fn from_attention(
        attn: ZImageAttention<B>,
        apply_lora: bool,
        rank: usize,
        alpha: f32,
        device: &B::Device,
    ) -> Self {
        let (qkv, to_out) = if apply_lora {
            (
                LoraLinear::from_linear(attn.qkv, rank, alpha, device),
                LoraLinear::from_linear(attn.to_out, rank, alpha, device),
            )
        } else {
            (
                LoraLinear::from_linear_frozen(attn.qkv),
                LoraLinear::from_linear_frozen(attn.to_out),
            )
        };

        ZImageAttentionLora {
            n_heads: attn.n_heads,
            n_kv_heads: attn.n_kv_heads,
            head_dim: attn.head_dim,
            qkv,
            to_out,
            q_norm: attn.q_norm.map(|n| n.no_grad()),
            k_norm: attn.k_norm.map(|n| n.no_grad()),
        }
    }

    pub fn forward(
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

        let head_slice = get_attention_slice_size();
        let seq_chunk = get_attention_seq_chunk_size();

        let query = query.movedim(1, 2);
        let key = key.movedim(1, 2);
        let value = value.movedim(1, 2);
        let mask = attention_mask.map(|m| m.unsqueeze_dims(&[1, 2]));

        // Use standard attention (slicing support can be added later for training)
        let hidden_states = if seq_chunk > 0 && seqlen > seq_chunk {
            seq_chunked_attention(query, key, value, mask, seq_chunk, head_slice, *self.n_heads)
        } else if head_slice > 0 && *self.n_heads > head_slice {
            head_sliced_attention(query, key, value, mask, head_slice)
        } else {
            attention(query, key, value, mask)
        };

        let hidden_states = hidden_states.movedim(1, 2).reshape([
            bsz as i64,
            -1,
            (self.n_heads.0 * self.head_dim.0) as i64,
        ]);

        self.to_out.forward(hidden_states)
    }
}

// --- LoRA FeedForward ---

#[derive(Module, Debug)]
pub struct FeedForwardLora<B: Backend> {
    pub w1: LoraLinear<B>,
    pub w2: LoraLinear<B>,
    pub w3: LoraLinear<B>,
}

impl<B: Backend> FeedForwardLora<B> {
    pub fn from_feed_forward(
        ff: FeedForward<B>,
        apply_lora: bool,
        rank: usize,
        alpha: f32,
        device: &B::Device,
    ) -> Self {
        let (w1, w2, w3) = if apply_lora {
            (
                LoraLinear::from_linear(ff.w1, rank, alpha, device),
                LoraLinear::from_linear(ff.w2, rank, alpha, device),
                LoraLinear::from_linear(ff.w3, rank, alpha, device),
            )
        } else {
            (
                LoraLinear::from_linear_frozen(ff.w1),
                LoraLinear::from_linear_frozen(ff.w2),
                LoraLinear::from_linear_frozen(ff.w3),
            )
        };

        FeedForwardLora { w1, w2, w3 }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x1 = self.w1.forward(x.clone());
        let x3 = self.w3.forward(x);
        self.w2.forward(clamp_fp16(burn::tensor::activation::silu(x1) * x3))
    }
}

// --- LoRA TransformerBlock ---

#[derive(Module, Debug)]
pub struct ZImageTransformerBlockLora<B: Backend> {
    pub attention: ZImageAttentionLora<B>,
    pub feed_forward: FeedForwardLora<B>,
    attention_norm1: RmsNorm<B>,
    ffn_norm1: RmsNorm<B>,
    attention_norm2: RmsNorm<B>,
    ffn_norm2: RmsNorm<B>,
    adaln_modulation: Option<Linear<B>>,
}

impl<B: Backend> ZImageTransformerBlockLora<B> {
    pub fn from_block(
        block: ZImageTransformerBlock<B>,
        config: &LoraConfig,
        device: &B::Device,
    ) -> Self {
        ZImageTransformerBlockLora {
            attention: ZImageAttentionLora::from_attention(
                block.attention,
                config.target_attention,
                config.rank,
                config.alpha,
                device,
            ),
            feed_forward: FeedForwardLora::from_feed_forward(
                block.feed_forward,
                config.target_feed_forward,
                config.rank,
                config.alpha,
                device,
            ),
            attention_norm1: block.attention_norm1.no_grad(),
            ffn_norm1: block.ffn_norm1.no_grad(),
            attention_norm2: block.attention_norm2.no_grad(),
            ffn_norm2: block.ffn_norm2.no_grad(),
            adaln_modulation: block.adaln_modulation.map(|m| m.no_grad()),
        }
    }

    pub fn forward(
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

// --- LoRA Full Model ---

/// LoRA-wrapped Z-Image model for fine-tuning.
///
/// The base model weights are frozen, and only the LoRA adapter matrices
/// (A and B in each targeted linear layer) are trainable.
#[derive(Module, Debug)]
pub struct ZImageModelLora<B: Backend> {
    time_scale: Ignored<f64>,
    out_channels: Ignored<usize>,
    patch_size: Ignored<usize>,

    x_embedder: Linear<B>,
    final_layer: FinalLayer<B>,

    noise_refiner: Vec<ZImageTransformerBlockLora<B>>,
    context_refiner: Vec<ZImageTransformerBlockLora<B>>,

    t_embedder: TimestepEmbedder<B>,
    cap_embedder_0: RmsNorm<B>,
    cap_embedder_1: Linear<B>,

    x_pad_token: Param<Tensor<B, 2>>,
    cap_pad_token: Param<Tensor<B, 2>>,

    pub layers: Vec<ZImageTransformerBlockLora<B>>,
    rope_embedder: RopeEmbedder<B>,
}

impl<B: Backend> ZImageModelLora<B> {
    /// Build a LoRA model from an existing base model.
    ///
    /// All base weights are frozen. LoRA adapters are added to the layers
    /// specified by `config`.
    pub fn from_base(base: ZImageModel<B>, config: &LoraConfig, device: &B::Device) -> Self {
        // Freeze all non-LoRA components
        let x_embedder = base.x_embedder.no_grad();
        let final_layer = base.final_layer.no_grad();
        let t_embedder = base.t_embedder.no_grad();
        let cap_embedder_0 = base.cap_embedder_0.no_grad();
        let cap_embedder_1 = base.cap_embedder_1.no_grad();

        // Freeze pad tokens
        let x_pad_token = Param::from_tensor(base.x_pad_token.val().set_require_grad(false));
        let cap_pad_token = Param::from_tensor(base.cap_pad_token.val().set_require_grad(false));

        // Build refiner config: optionally no LoRA on refiners
        let refiner_config = if config.target_refiners {
            config.clone()
        } else {
            LoraConfig {
                rank: 0,
                alpha: 0.0,
                target_attention: false,
                target_feed_forward: false,
                target_refiners: false,
            }
        };

        let noise_refiner = base
            .noise_refiner
            .into_iter()
            .map(|block| ZImageTransformerBlockLora::from_block(block, &refiner_config, device))
            .collect();

        let context_refiner = base
            .context_refiner
            .into_iter()
            .map(|block| ZImageTransformerBlockLora::from_block(block, &refiner_config, device))
            .collect();

        let layers = base
            .layers
            .into_iter()
            .map(|block| ZImageTransformerBlockLora::from_block(block, config, device))
            .collect();

        ZImageModelLora {
            time_scale: base.time_scale,
            out_channels: base.out_channels,
            patch_size: base.patch_size,
            x_embedder,
            final_layer,
            noise_refiner,
            context_refiner,
            t_embedder,
            cap_embedder_0,
            cap_embedder_1,
            x_pad_token,
            cap_pad_token,
            layers,
            rope_embedder: base.rope_embedder,
        }
    }

    /// Perform one inference step (identical logic to ZImageModel::forward).
    pub fn forward(
        &self,
        latents: Tensor<B, 4>,
        timestep: Tensor<B, 1>,
        cap_feats: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let output_dtype = latents.dtype();
        let model_dtype = crate::utils::effective_dtype(self.cap_pad_token.dtype());

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

    /// Count the number of trainable LoRA parameters.
    pub fn lora_param_count(&self) -> usize {
        let mut count = 0;
        for layer in &self.layers {
            count += count_lora_params(&layer.attention.qkv);
            count += count_lora_params(&layer.attention.to_out);
            count += count_lora_params(&layer.feed_forward.w1);
            count += count_lora_params(&layer.feed_forward.w2);
            count += count_lora_params(&layer.feed_forward.w3);
        }
        for layer in &self.noise_refiner {
            count += count_lora_params(&layer.attention.qkv);
            count += count_lora_params(&layer.attention.to_out);
            count += count_lora_params(&layer.feed_forward.w1);
            count += count_lora_params(&layer.feed_forward.w2);
            count += count_lora_params(&layer.feed_forward.w3);
        }
        for layer in &self.context_refiner {
            count += count_lora_params(&layer.attention.qkv);
            count += count_lora_params(&layer.attention.to_out);
            count += count_lora_params(&layer.feed_forward.w1);
            count += count_lora_params(&layer.feed_forward.w2);
            count += count_lora_params(&layer.feed_forward.w3);
        }
        count
    }
}

fn count_lora_params<B: Backend>(lora: &LoraLinear<B>) -> usize {
    match (&lora.lora_a, &lora.lora_b) {
        (Some(a), Some(b)) => {
            let a_dims = a.dims();
            let b_dims = b.dims();
            a_dims[0] * a_dims[1] + b_dims[0] * b_dims[1]
        }
        _ => 0,
    }
}

// --- Standalone attention helpers (mirrored from attention.rs for LoRA model) ---

fn head_sliced_attention<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    mask: Option<Tensor<B, 4, Bool>>,
    head_slice: usize,
) -> Tensor<B, 4> {
    let [bsz, n_heads, seqlen, head_dim] = query.dims();
    let mut output_slices = Vec::with_capacity((n_heads + head_slice - 1) / head_slice);

    for start in (0..n_heads).step_by(head_slice) {
        let end = (start + head_slice).min(n_heads);
        let q_slice = query.clone().slice([0..bsz, start..end, 0..seqlen, 0..head_dim]);
        let k_slice = key.clone().slice([0..bsz, start..end, 0..seqlen, 0..head_dim]);
        let v_slice = value.clone().slice([0..bsz, start..end, 0..seqlen, 0..head_dim]);
        let attn_slice = attention(q_slice, k_slice, v_slice, mask.clone());
        output_slices.push(attn_slice);
    }

    Tensor::cat(output_slices, 1)
}

fn seq_chunked_attention<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    mask: Option<Tensor<B, 4, Bool>>,
    seq_chunk: usize,
    head_slice: usize,
    n_heads: usize,
) -> Tensor<B, 4> {
    let [bsz, _n_heads, seqlen, head_dim] = query.dims();
    let use_head_slice = head_slice > 0 && n_heads > head_slice;
    let mut seq_outputs = Vec::with_capacity((seqlen + seq_chunk - 1) / seq_chunk);

    for q_start in (0..seqlen).step_by(seq_chunk) {
        let q_end = (q_start + seq_chunk).min(seqlen);
        let q_chunk = query.clone().slice([0..bsz, 0..n_heads, q_start..q_end, 0..head_dim]);

        let mask_chunk = mask.clone().map(|m| {
            let m_dims = m.dims();
            if m_dims[2] == 1 {
                m
            } else {
                m.slice([0..bsz, 0..m_dims[1], q_start..q_end, 0..seqlen])
            }
        });

        let chunk_output = if use_head_slice {
            let mut head_outputs = Vec::with_capacity((n_heads + head_slice - 1) / head_slice);
            for h_start in (0..n_heads).step_by(head_slice) {
                let h_end = (h_start + head_slice).min(n_heads);
                let chunk_len = q_end - q_start;
                let q_slice = q_chunk.clone().slice([0..bsz, h_start..h_end, 0..chunk_len, 0..head_dim]);
                let k_slice = key.clone().slice([0..bsz, h_start..h_end, 0..seqlen, 0..head_dim]);
                let v_slice = value.clone().slice([0..bsz, h_start..h_end, 0..seqlen, 0..head_dim]);
                let attn = attention(q_slice, k_slice, v_slice, mask_chunk.clone());
                head_outputs.push(attn);
            }
            Tensor::cat(head_outputs, 1)
        } else {
            attention(q_chunk, key.clone(), value.clone(), mask_chunk)
        };

        seq_outputs.push(chunk_output);
    }

    Tensor::cat(seq_outputs, 2)
}
