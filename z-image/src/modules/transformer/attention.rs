use burn::{
    Tensor,
    config::Config,
    module::{Ignored, Module},
    nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::{Bool, module::attention},
};

use crate::modules::transformer::rope::apply_rotary_emb;

#[derive(Config, Debug)]
pub struct ZImageAttentionConfig {
    dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    #[config(default = true)]
    qk_norm: bool,
    #[config(default = 1e-5)]
    eps: f64,
}

impl ZImageAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ZImageAttention<B> {
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
pub struct ZImageAttention<B: Backend> {
    n_heads: Ignored<usize>,
    n_kv_heads: Ignored<usize>,
    head_dim: Ignored<usize>,

    qkv: Linear<B>,
    to_out: Linear<B>,

    q_norm: Option<RmsNorm<B>>,
    k_norm: Option<RmsNorm<B>>,
}

impl<B: Backend> ZImageAttention<B> {
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
