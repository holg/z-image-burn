//! This implementation is based on:
//! https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/autoencoder.py

use burn::{
    Tensor,
    config::Config,
    module::{Ignored, Module},
    nn::{
        GroupNorm, GroupNormConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
        interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode},
    },
    prelude::Backend,
    tensor::{self, activation::sigmoid},
};

#[derive(Config, Debug)]
pub struct AutoEncoderConfig {
    resolution: usize,
    in_channels: usize,
    ch: usize,
    out_ch: usize,
    ch_mult: Vec<usize>,
    num_res_blocks: usize,
    z_channels: usize,
    scale_factor: f32,
    shift_factor: f32,
}

impl AutoEncoderConfig {
    /// Get an [AutoEncoderConfig] with the parameters used for FLUX.1-dev.
    pub fn flux_ae() -> Self {
        AutoEncoderConfig {
            resolution: 256,
            in_channels: 3,
            ch: 128,
            out_ch: 3,
            ch_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            z_channels: 16,
            scale_factor: 0.3611,
            shift_factor: 0.1159,
        }
    }

    pub fn z_image_ae() -> Self {
        AutoEncoderConfig {
            resolution: 256,
            in_channels: 3,
            ch: 128,
            out_ch: 3,
            ch_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            z_channels: 16,
            scale_factor: 0.18215,
            shift_factor: 0.0,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> AutoEncoder<B> {
        AutoEncoder {
            decoder: DecoderConfig::new(
                self.ch,
                self.out_ch,
                self.ch_mult.clone(),
                self.num_res_blocks,
                self.in_channels,
                self.resolution,
                self.z_channels,
            )
            .init(device),
            scale_factor: Ignored(self.scale_factor),
            shift_factor: Ignored(self.shift_factor),
        }
    }
}

#[derive(Module, Debug)]
pub struct AutoEncoder<B: Backend> {
    decoder: Decoder<B>,
    scale_factor: Ignored<f32>,
    shift_factor: Ignored<f32>,
}

impl<B: Backend> AutoEncoder<B> {
    /// Turn a latent space into a pixel space image.
    pub fn decode(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        let z = z / self.scale_factor.0 + self.shift_factor.0;
        self.decoder.forward(z)
    }
}

#[derive(Config, Debug)]
struct DecoderConfig {
    ch: usize,
    out_ch: usize,
    ch_mult: Vec<usize>,
    num_res_blocks: usize,
    in_channels: usize,
    resolution: usize,
    z_channels: usize,
}

impl DecoderConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Decoder<B> {
        let num_resolutions = self.ch_mult.len();
        let mut block_in = self.ch * self.ch_mult[num_resolutions - 1];

        let mut up = Vec::new();
        let mut curr_res = self.resolution / 2usize.pow(num_resolutions as u32 - 1);

        let conv_in = Conv2dConfig::new([self.z_channels, block_in], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let mid_block_1 = ResnetBlockConfig::new(block_in, block_in).init(device);
        let mid_attn_1 = AttnBlockConfig::new(block_in).init(device);
        let mid_block_2 = ResnetBlockConfig::new(block_in, block_in).init(device);

        for i_level in (0..num_resolutions).rev() {
            let mut block = Vec::new();

            let block_out = self.ch * self.ch_mult[i_level];

            for _ in 0..=self.num_res_blocks {
                block.push(ResnetBlockConfig::new(block_in, block_out).init(device));
                block_in = block_out;
            }

            let upsample = (i_level != 0).then(|| {
                let upsample = UpsampleBlockConfig::new(block_in).init(device);
                curr_res = curr_res * 2;
                upsample
            });

            up.insert(0, DecoderUpsampler { block, upsample });
        }

        Decoder {
            num_resolutions: Ignored(num_resolutions),
            num_res_blocks: Ignored(self.num_res_blocks),
            conv_in,
            mid_block_1,
            mid_attn_1,
            mid_block_2,
            up,
            norm_out: GroupNormConfig::new(32, block_in).init(device),
            conv_out: Conv2dConfig::new([block_in, self.out_ch], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
struct Decoder<B: Backend> {
    num_resolutions: Ignored<usize>,
    num_res_blocks: Ignored<usize>,

    conv_in: Conv2d<B>,

    mid_block_1: ResnetBlock<B>,
    mid_attn_1: AttnBlock<B>,
    mid_block_2: ResnetBlock<B>,

    up: Vec<DecoderUpsampler<B>>,

    norm_out: GroupNorm<B>,
    conv_out: Conv2d<B>,
}

impl<B: Backend> Decoder<B> {
    fn forward(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut h = self.conv_in.forward(z);

        h = self.mid_block_1.forward(h);
        h = self.mid_attn_1.forward(h);
        h = self.mid_block_2.forward(h);

        for i_level in (0..self.num_resolutions.0).rev() {
            for i_block in 0..=self.num_res_blocks.0 {
                h = self.up[i_level].block[i_block].forward(h);
            }

            if i_level != 0 {
                h = self.up[i_level]
                    .upsample
                    .as_ref()
                    .expect("block should exist if i_level==0")
                    .forward(h);
            }
        }

        h = self.norm_out.forward(h);
        h = swish(h);
        h = self.conv_out.forward(h);
        h
    }
}

#[derive(Module, Debug)]
struct DecoderUpsampler<B: Backend> {
    block: Vec<ResnetBlock<B>>,
    upsample: Option<UpsampleBlock<B>>,
}

#[derive(Config, Debug)]
struct UpsampleBlockConfig {
    in_channels: usize,
}

impl UpsampleBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> UpsampleBlock<B> {
        UpsampleBlock {
            interpolate: Interpolate2dConfig::new()
                .with_scale_factor(Some([2., 2.]))
                .with_mode(InterpolateMode::Nearest)
                .init(),
            conv: Conv2dConfig::new([self.in_channels, self.in_channels], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
struct UpsampleBlock<B: Backend> {
    interpolate: Interpolate2d,
    conv: Conv2d<B>,
}

impl<B: Backend> UpsampleBlock<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.interpolate.forward(x);
        let x = self.conv.forward(x);
        x
    }
}

#[derive(Config, Debug)]
struct AttnBlockConfig {
    in_channels: usize,
}

impl AttnBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> AttnBlock<B> {
        AttnBlock {
            norm: GroupNormConfig::new(32, self.in_channels)
                .with_affine(true)
                .with_epsilon(1e-6)
                .init(device),
            q: Conv2dConfig::new([self.in_channels, self.in_channels], [1, 1]).init(device),
            k: Conv2dConfig::new([self.in_channels, self.in_channels], [1, 1]).init(device),
            v: Conv2dConfig::new([self.in_channels, self.in_channels], [1, 1]).init(device),
            proj_out: Conv2dConfig::new([self.in_channels, self.in_channels], [1, 1]).init(device),
        }
    }
}

#[derive(Module, Debug)]
struct AttnBlock<B: Backend> {
    norm: GroupNorm<B>,
    q: Conv2d<B>,
    k: Conv2d<B>,
    v: Conv2d<B>,
    proj_out: Conv2d<B>,
}

impl<B: Backend> AttnBlock<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        x.clone() + self.proj_out.forward(self.attention(x))
    }

    fn attention(&self, h_: Tensor<B, 4>) -> Tensor<B, 4> {
        let h_ = self.norm.forward(h_);
        let q = self.q.forward(h_.clone());
        let k = self.k.forward(h_.clone());
        let v = self.v.forward(h_.clone());

        let [b, c, h, w] = q.shape().dims();
        let q = q.reshape([b, 1, h * w, c]);
        let k = k.reshape([b, 1, h * w, c]);
        let v = v.reshape([b, 1, h * w, c]);
        let h_ = tensor::module::attention(q, k, v, None);

        h_.reshape([b, c, h, w])
    }
}

#[derive(Config, Debug)]
struct ResnetBlockConfig {
    in_channels: usize,
    out_channels: usize,
}

impl ResnetBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ResnetBlock<B> {
        ResnetBlock {
            norm1: GroupNormConfig::new(32, self.in_channels)
                .with_epsilon(1e-6)
                .with_affine(true)
                .init(device),
            conv1: Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            norm2: GroupNormConfig::new(32, self.out_channels)
                .with_epsilon(1e-6)
                .with_affine(true)
                .init(device),
            conv2: Conv2dConfig::new([self.out_channels, self.out_channels], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            nin_shortcut: (self.in_channels != self.out_channels).then(|| {
                Conv2dConfig::new([self.in_channels, self.out_channels], [1, 1])
                    .with_stride([1, 1])
                    .with_padding(PaddingConfig2d::Explicit(0, 0))
                    .init(device)
            }),
        }
    }
}

#[derive(Module, Debug)]
struct ResnetBlock<B: Backend> {
    norm1: GroupNorm<B>,
    conv1: Conv2d<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    nin_shortcut: Option<Conv2d<B>>,
}

impl<B: Backend> ResnetBlock<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = x.clone();
        let h = self.norm1.forward(h);
        let h = swish(h);
        let h = self.conv1.forward(h);

        let h = self.norm2.forward(h);
        let h = swish(h);
        let h = self.conv2.forward(h);

        let x = if let Some(nin_shortcut) = &self.nin_shortcut {
            nin_shortcut.forward(x)
        } else {
            x
        };

        x + h
    }
}

fn swish<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    x.clone() * sigmoid(x)
}
