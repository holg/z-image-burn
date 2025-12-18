use burn::{
    Tensor,
    config::Config,
    module::{Ignored, Module},
    nn::{Linear, LinearConfig},
    prelude::Backend,
    tensor::{DType, activation::silu, s},
};

use crate::modules::transformer::ADALN_EMBED_DIM;

#[derive(Config, Debug)]
pub struct TimestepEmbedderConfig {
    out_size: usize,
    mid_size: Option<usize>,
    #[config(default = "ADALN_EMBED_DIM")]
    frequency_embedding_size: usize,
}

impl TimestepEmbedderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TimestepEmbedder<B> {
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
pub struct TimestepEmbedder<B: Backend> {
    frequency_embedding_size: Ignored<usize>,
    mlp_1: Linear<B>,
    mlp_2: Linear<B>,
}

impl<B: Backend> TimestepEmbedder<B> {
    pub fn forward(&self, t: Tensor<B, 1>) -> Tensor<B, 2> {
        let t_freq = Self::timestep_embedding(t, self.frequency_embedding_size.0, 10000);

        let t_emb = self.mlp_1.forward(t_freq);
        let t_emb = silu(t_emb);
        self.mlp_2.forward(t_emb)
    }

    fn timestep_embedding(t: Tensor<B, 1>, dim: usize, max_period: usize) -> Tensor<B, 2> {
        let device = t.device();
        let original_type = t.dtype();
        let t = t.cast(DType::F32);

        let half = (dim / 2) as i64;
        let freqs = Tensor::<B, 1>::exp(
            -(max_period as f32).ln() * Tensor::arange(0..half, &device).float().cast(DType::F32)
                / half,
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

        embedding.cast(original_type)
    }
}
