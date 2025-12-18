use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::Backend,
    tensor::activation::silu,
};

use crate::modules::transformer::{
    ADALN_EMBED_DIM, layer_norm::LayerNormNoAffine, utils::modulate,
};

#[derive(Config, Debug)]
pub struct FinalLayerConfig {
    hidden_size: usize,
    out_channels: usize,
}

impl FinalLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FinalLayer<B> {
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
pub struct FinalLayer<B: Backend> {
    norm_final: LayerNormNoAffine<B>,
    linear: Linear<B>,
    adaln_modulation: Linear<B>,
}

impl<B: Backend> FinalLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>, c: Tensor<B, 2>) -> Tensor<B, 3> {
        let scale = self.adaln_modulation.forward(silu(c));
        let x = modulate(self.norm_final.forward(x), scale);
        let x = self.linear.forward(x);
        x
    }
}
