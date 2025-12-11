use std::marker::PhantomData;

use burn::{
    Tensor,
    module::{Ignored, Module},
    prelude::Backend,
    tensor::{DType, Int, s},
};

use crate::compat::float_vec_linspace;

#[derive(Module, Debug)]
pub struct RopeEmbedder<B: Backend> {
    b: PhantomData<B>,
    dim: Ignored<usize>,
    theta: Ignored<f64>,
    axes_dims: Ignored<Vec<usize>>,
}

impl<B: Backend> RopeEmbedder<B> {
    pub fn new(dim: usize, theta: f64, axes_dims: Vec<usize>) -> Self {
        Self {
            b: Default::default(),
            dim: Ignored(dim),
            theta: Ignored(theta),
            axes_dims: Ignored(axes_dims),
        }
    }

    pub fn forward(&self, ids: Tensor<B, 3, Int>) -> Tensor<B, 6> {
        let n_axes = ids.shape()[2];

        let x: Vec<_> = (0..n_axes)
            .map(|i| {
                rope(
                    ids.clone().slice(s![.., .., i]).squeeze_dim::<2>(2),
                    self.axes_dims[i],
                    *self.theta,
                )
            })
            .collect();
        let x_ndims = x[0].dims().len();
        let emb = Tensor::cat(x, x_ndims - 3);
        emb.unsqueeze_dim(1)
    }
}

fn rope<B: Backend>(pos: Tensor<B, 2, Int>, dim: usize, theta: f64) -> Tensor<B, 5> {
    debug_assert!(dim.rem_euclid(2) == 0);
    let device = pos.device();

    let scale = Tensor::<B, 1>::from_data_dtype(
        float_vec_linspace(0., (dim - 2) as f64 / dim as f64, dim / 2).as_slice(),
        &device,
        DType::F64,
    );
    let omega: Tensor<B, 1> = 1.0
        / (Tensor::from_floats([theta], &device)
            .cast(DType::F64)
            .powf(scale));

    let out = pos.float().cast(DType::F64).unsqueeze_dim::<3>(2) * omega.unsqueeze::<3>();
    let out = Tensor::stack(
        vec![
            out.clone().cos(),
            -out.clone().sin(),
            out.clone().sin(),
            out.cos(),
        ],
        3,
    );
    let [b, n, d, rest] = out.dims();
    assert_eq!(rest, 4);
    let out = out.reshape([b, n, d, 2, 2]);
    out.cast(DType::F32)
}

pub fn apply_rotary_emb<B: Backend>(x: Tensor<B, 4>, freqs_cis: Tensor<B, 6>) -> Tensor<B, 4> {
    let original_type = x.dtype();
    let x_shape = x.shape();
    let x_dims = x.dims();

    let x = x.cast(freqs_cis.dtype()).reshape([
        x_dims[0] as i64,
        x_dims[1] as i64,
        x_dims[2] as i64,
        -1,
        1,
        2,
    ]);

    let x_out = freqs_cis
        .clone()
        .slice(s![.., .., .., .., .., 0])
        .squeeze_dim::<5>(5)
        * x.clone()
            .slice(s![.., .., .., .., .., 0])
            .squeeze_dim::<5>(5);
    let x_out = x_out
        + freqs_cis
            .slice(s![.., .., .., .., .., 1])
            .squeeze_dim::<5>(5)
            * x.slice(s![.., .., .., .., .., 1]).squeeze_dim::<5>(5);

    x_out.reshape(x_shape).cast(original_type)
}
