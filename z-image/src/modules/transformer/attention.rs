use burn::{
    Tensor,
    config::Config,
    module::{Ignored, Module},
    nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::{Bool, module::attention},
};

use crate::modules::transformer::rope::apply_rotary_emb;

/// Global attention slice size. When set to Some(n), attention is computed
/// in chunks of n heads to reduce peak memory usage. Use None for no slicing.
/// Typical values: None (fastest), Some(8), Some(4), Some(2), Some(1) (lowest memory)
static ATTENTION_SLICE_SIZE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Set the attention slice size for memory optimization.
/// - `0` means no slicing (fastest, but uses most memory)
/// - `n > 0` means process `n` heads at a time (lower = less memory, but slower)
///
/// Recommended values for different memory constraints:
/// - High memory (24GB+): 0 (no slicing)
/// - Medium memory (16GB): 8
/// - Low memory (8-12GB): 4 or 2
/// - Very low memory (<8GB): 1
pub fn set_attention_slice_size(size: usize) {
    ATTENTION_SLICE_SIZE.store(size, std::sync::atomic::Ordering::Relaxed);
    if size == 0 {
        eprintln!("[z-image] Attention slicing disabled (using full attention)");
    } else {
        eprintln!("[z-image] Attention slice size set to {} heads", size);
    }
}

/// Get the current attention slice size.
pub fn get_attention_slice_size() -> usize {
    ATTENTION_SLICE_SIZE.load(std::sync::atomic::Ordering::Relaxed)
}

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

        // Check if attention slicing is enabled
        let slice_size = get_attention_slice_size();

        let hidden_states = if slice_size > 0 && *self.n_heads > slice_size {
            // Sliced attention: process heads in chunks to reduce peak memory
            self.sliced_attention(query, key, value, attention_mask, slice_size)
        } else {
            // Standard full attention
            attention(
                query.movedim(1, 2),
                key.movedim(1, 2),
                value.movedim(1, 2),
                attention_mask.map(|attention_mask| attention_mask.unsqueeze_dims(&[1, 2])),
            )
        };

        let hidden_states = hidden_states.movedim(1, 2).reshape([
            bsz as i64,
            -1,
            (self.n_heads.0 * self.head_dim.0) as i64,
        ]);

        self.to_out.forward(hidden_states)
    }

    /// Compute attention in slices to reduce peak memory usage.
    /// This is slower but uses significantly less GPU memory.
    fn sliced_attention(
        &self,
        query: Tensor<B, 4>,  // [bsz, seqlen, n_heads, head_dim]
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        slice_size: usize,
    ) -> Tensor<B, 4> {
        let [bsz, seqlen, n_heads, head_dim] = query.dims();

        // Transpose to [bsz, n_heads, seqlen, head_dim] for attention
        let query = query.movedim(1, 2);
        let key = key.movedim(1, 2);
        let value = value.movedim(1, 2);

        let mask = attention_mask.map(|m| m.unsqueeze_dims(&[1, 2]));

        // Process attention in chunks of slice_size heads
        let mut output_slices = Vec::with_capacity((n_heads + slice_size - 1) / slice_size);

        for start in (0..n_heads).step_by(slice_size) {
            let end = (start + slice_size).min(n_heads);

            // Extract slices for this chunk of heads
            let q_slice = query.clone().slice([0..bsz, start..end, 0..seqlen, 0..head_dim]);
            let k_slice = key.clone().slice([0..bsz, start..end, 0..seqlen, 0..head_dim]);
            let v_slice = value.clone().slice([0..bsz, start..end, 0..seqlen, 0..head_dim]);

            // Compute attention for this slice
            let attn_slice = attention(
                q_slice,
                k_slice,
                v_slice,
                mask.clone(),
            );

            output_slices.push(attn_slice);
        }

        // Concatenate all slices along the heads dimension
        Tensor::cat(output_slices, 1)
    }
}
