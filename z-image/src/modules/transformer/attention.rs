use burn::{
    Tensor,
    config::Config,
    module::{Ignored, Module},
    nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::{Bool, module::attention},
};

use crate::modules::transformer::rope::apply_rotary_emb;

/// Global attention slice size for heads. When set to n > 0, attention is computed
/// in chunks of n heads to reduce peak memory usage.
static ATTENTION_HEAD_SLICE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Global attention slice size for sequence. When set to n > 0, the query sequence
/// is chunked into pieces of n tokens, each computed against full key/value.
/// This reduces peak memory from O(seqlen^2) to O(seqlen * chunk_size).
static ATTENTION_SEQ_SLICE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

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
    ATTENTION_HEAD_SLICE.store(size, std::sync::atomic::Ordering::Relaxed);
    if size == 0 {
        eprintln!("[z-image] Head slicing disabled");
    } else {
        eprintln!("[z-image] Head slice size set to {} heads", size);
    }
}

/// Set the sequence chunk size for memory optimization.
/// - `0` means no chunking (compute full seqlen×seqlen attention)
/// - `n > 0` means chunk query into pieces of n tokens
///
/// For 384x384 (2304 tokens), try 512 or 256 to reduce memory by 4-8x.
/// For 512x512 (4096 tokens), try 512 or 256.
pub fn set_attention_seq_chunk_size(size: usize) {
    ATTENTION_SEQ_SLICE.store(size, std::sync::atomic::Ordering::Relaxed);
    if size == 0 {
        eprintln!("[z-image] Sequence chunking disabled");
    } else {
        eprintln!("[z-image] Sequence chunk size set to {} tokens", size);
    }
}

/// Get the current attention slice size.
pub fn get_attention_slice_size() -> usize {
    ATTENTION_HEAD_SLICE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Get the current sequence chunk size.
pub fn get_attention_seq_chunk_size() -> usize {
    ATTENTION_SEQ_SLICE.load(std::sync::atomic::Ordering::Relaxed)
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
    pub(crate) n_heads: Ignored<usize>,
    pub(crate) n_kv_heads: Ignored<usize>,
    pub(crate) head_dim: Ignored<usize>,

    pub(crate) qkv: Linear<B>,
    pub(crate) to_out: Linear<B>,

    pub(crate) q_norm: Option<RmsNorm<B>>,
    pub(crate) k_norm: Option<RmsNorm<B>>,
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
        let head_slice = get_attention_slice_size();
        let seq_chunk = get_attention_seq_chunk_size();

        // Transpose for attention: [bsz, seqlen, n_heads, head_dim] -> [bsz, n_heads, seqlen, head_dim]
        let query = query.movedim(1, 2);
        let key = key.movedim(1, 2);
        let value = value.movedim(1, 2);
        let mask = attention_mask.map(|m| m.unsqueeze_dims(&[1, 2]));

        let hidden_states = if seq_chunk > 0 && seqlen > seq_chunk {
            // Sequence-chunked attention: reduces O(seqlen^2) to O(seqlen * chunk)
            self.seq_chunked_attention(query, key, value, mask, seq_chunk, head_slice)
        } else if head_slice > 0 && *self.n_heads > head_slice {
            // Head-sliced attention only
            self.head_sliced_attention(query, key, value, mask, head_slice)
        } else {
            // Standard full attention
            attention(query, key, value, mask)
        };

        let hidden_states = hidden_states.movedim(1, 2).reshape([
            bsz as i64,
            -1,
            (self.n_heads.0 * self.head_dim.0) as i64,
        ]);

        self.to_out.forward(hidden_states)
    }

    /// Compute attention with head slicing to reduce peak memory.
    /// Input tensors are already in [bsz, n_heads, seqlen, head_dim] format.
    fn head_sliced_attention(
        &self,
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

    /// Compute attention with sequence chunking to reduce peak memory.
    /// Chunks the query sequence and computes attention against full key/value.
    /// This reduces memory from O(seqlen^2) to O(seqlen * chunk_size).
    ///
    /// Input tensors are in [bsz, n_heads, seqlen, head_dim] format.
    fn seq_chunked_attention(
        &self,
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        mask: Option<Tensor<B, 4, Bool>>,
        seq_chunk: usize,
        head_slice: usize,
    ) -> Tensor<B, 4> {
        let [bsz, n_heads, seqlen, head_dim] = query.dims();

        // If head slicing is also enabled, apply both
        let use_head_slice = head_slice > 0 && n_heads > head_slice;

        let mut seq_outputs = Vec::with_capacity((seqlen + seq_chunk - 1) / seq_chunk);

        for q_start in (0..seqlen).step_by(seq_chunk) {
            let q_end = (q_start + seq_chunk).min(seqlen);

            // Chunk the query sequence
            let q_chunk = query.clone().slice([0..bsz, 0..n_heads, q_start..q_end, 0..head_dim]);

            // Chunk the mask if present (only rows corresponding to query chunk)
            let mask_chunk = mask.clone().map(|m| {
                // mask shape: [bsz, 1, 1, seqlen] or [bsz, 1, seqlen, seqlen]
                let m_dims = m.dims();
                if m_dims[2] == 1 {
                    // Broadcast mask, no chunking needed
                    m
                } else {
                    // Full mask, chunk the query dimension
                    m.slice([0..bsz, 0..m_dims[1], q_start..q_end, 0..seqlen])
                }
            });

            // Compute attention for this query chunk against full key/value
            let chunk_output = if use_head_slice {
                // Also apply head slicing within each sequence chunk
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

        // Concatenate along sequence dimension
        Tensor::cat(seq_outputs, 2)
    }
}
