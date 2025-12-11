use burn::{
    Tensor,
    prelude::Backend,
    tensor::{DType, ops::PadMode},
};

pub fn pad_to_patch_size<B: Backend>(img: Tensor<B, 4>, patch_size: [usize; 2]) -> Tensor<B, 4> {
    let pad1 = (patch_size[0] - img.dims()[2].rem_euclid(patch_size[0])).rem_euclid(patch_size[0]);
    let pad2 = (patch_size[1] - img.dims()[3].rem_euclid(patch_size[1])).rem_euclid(patch_size[1]);

    let pad = (0, pad2, 0, pad1);

    img.pad(pad, PadMode::Reflect)
}

/// When the tensor's DType if f16, ensure the value is not NaN or ±inf. This hacky workaround is
/// needed for this module in f16.
pub fn clamp_fp16<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    if x.dtype() == DType::F16 {
        let nans = x.clone().is_nan();
        x.mask_fill(nans, 0.0).clamp(-66504, 66504)
    } else {
        x
    }
}

pub fn modulate<B: Backend>(x: Tensor<B, 3>, scale: Tensor<B, 2>) -> Tensor<B, 3> {
    x * (1. + scale.unsqueeze_dim(1))
}
