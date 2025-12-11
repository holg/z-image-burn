use std::marker::PhantomData;

use burn::{
    Tensor,
    module::{Ignored, Module},
    prelude::Backend,
};

/// A layer norm without elementwise affine (no learnable parameters).
#[derive(Module, Debug)]
pub struct LayerNormNoAffine<B: Backend> {
    backend: PhantomData<B>,
    epsilon: Ignored<f64>,
}

impl<B: Backend> LayerNormNoAffine<B> {
    pub fn new(epsilon: f64) -> Self {
        Self {
            backend: Default::default(),
            epsilon: Ignored(epsilon),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let mean = x.clone().mean_dim(D - 1);
        let var_bias = x.clone().sub(mean.clone()).powf_scalar(2.0).mean_dim(D - 1);
        let std = var_bias.add_scalar(self.epsilon.0).sqrt();
        x.sub(mean).div(std)
    }
}
