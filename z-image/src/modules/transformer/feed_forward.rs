use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::Backend,
    tensor::activation::silu,
};

use crate::modules::transformer::utils::clamp_fp16;

#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    dim: usize,
    hidden_dim: usize,
}

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
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
pub struct FeedForward<B: Backend> {
    w1: Linear<B>,
    w2: Linear<B>,
    w3: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.w2
            .forward(self.forward_silu_gating(self.w1.forward(x.clone()), self.w3.forward(x)))
    }

    fn forward_silu_gating(&self, x1: Tensor<B, 3>, x3: Tensor<B, 3>) -> Tensor<B, 3> {
        clamp_fp16(silu(x1) * x3)
    }
}
