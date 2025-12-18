use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::Bool,
};

use crate::modules::transformer::{
    ADALN_EMBED_DIM,
    attention::{ZImageAttention, ZImageAttentionConfig},
    feed_forward::{FeedForward, FeedForwardConfig},
    utils::{clamp_fp16, modulate},
};

#[derive(Config, Debug)]
pub struct ZImageTransformerBlockConfig {
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
    pub fn init<B: Backend>(&self, device: &B::Device) -> ZImageTransformerBlock<B> {
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
pub struct ZImageTransformerBlock<B: Backend> {
    attention: ZImageAttention<B>,
    feed_forward: FeedForward<B>,
    attention_norm1: RmsNorm<B>,
    ffn_norm1: RmsNorm<B>,
    attention_norm2: RmsNorm<B>,
    ffn_norm2: RmsNorm<B>,
    adaln_modulation: Option<Linear<B>>,
}

impl<B: Backend> ZImageTransformerBlock<B> {
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
