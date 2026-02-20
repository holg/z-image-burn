//! LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.
//!
//! Wraps a frozen `Linear` layer with trainable low-rank A and B matrices.
//! The output is: `base(x) + scaling * (x @ A^T @ B^T)`
//! where `scaling = alpha / rank`.

use burn::{
    Tensor,
    module::{Ignored, Module, Param},
    nn::Linear,
    prelude::Backend,
    tensor::Distribution,
};

/// A linear layer with optional LoRA adaptation.
///
/// When LoRA params are present (`lora_a` and `lora_b` are `Some`), the forward pass
/// computes: `base(x) + scaling * (x @ A^T @ B^T)`.
///
/// When LoRA params are `None`, this is a simple passthrough to the base linear layer.
#[derive(Module, Debug)]
pub struct LoraLinear<B: Backend> {
    /// The base linear layer (should be frozen via `no_grad()` during training).
    pub base: Linear<B>,
    /// Down-projection: shape [rank, in_features]. Initialized with Kaiming-like scaling.
    pub lora_a: Option<Param<Tensor<B, 2>>>,
    /// Up-projection: shape [out_features, rank]. Initialized to zeros.
    pub lora_b: Option<Param<Tensor<B, 2>>>,
    /// Scaling factor: alpha / rank.
    scaling: Ignored<f32>,
}

impl<B: Backend> LoraLinear<B> {
    /// Create a LoRA-adapted layer from an existing `Linear` layer.
    ///
    /// The base layer is frozen (all params set to `no_grad()`).
    /// LoRA A is initialized from N(0, 1/sqrt(rank)) and B is initialized to zeros,
    /// so the LoRA contribution starts at zero.
    pub fn from_linear(linear: Linear<B>, rank: usize, alpha: f32, device: &B::Device) -> Self {
        let in_features = linear.weight.dims()[0];
        let out_features = linear.weight.dims()[1];

        // Freeze base weights
        let base = linear.no_grad();

        // A: [rank, in_features] initialized with scaled normal
        let std = 1.0 / (rank as f64).sqrt();
        let lora_a = Param::from_tensor(
            Tensor::random([rank, in_features], Distribution::Normal(0.0, std), device),
        );

        // B: [out_features, rank] initialized to zeros
        let lora_b = Param::from_tensor(Tensor::zeros([out_features, rank], device));

        let scaling = alpha / rank as f32;

        LoraLinear {
            base,
            lora_a: Some(lora_a),
            lora_b: Some(lora_b),
            scaling: Ignored(scaling),
        }
    }

    /// Create a frozen passthrough (no LoRA adaptation).
    ///
    /// Used for layers that should not be LoRA-adapted but still need to fit
    /// the `LoraLinear` type in the model struct.
    pub fn from_linear_frozen(linear: Linear<B>) -> Self {
        let base = linear.no_grad();
        LoraLinear {
            base,
            lora_a: None,
            lora_b: None,
            scaling: Ignored(0.0),
        }
    }

    /// Forward pass with LoRA adaptation for 3D tensors `[batch, seq, features]`.
    ///
    /// If LoRA params are present: `base(x) + scaling * x @ A^T @ B^T`
    /// If not: just `base(x)`
    ///
    /// This is specialized for 3D tensors because that's what the Z-Image transformer uses
    /// for all its linear layers (attention qkv/out and feed-forward w1/w2/w3).
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let base_out = self.base.forward(x.clone());

        match (&self.lora_a, &self.lora_b) {
            (Some(a), Some(b)) => {
                // LoRA path: compute low-rank adaptation
                // x shape: [batch, seq, in_features]
                // A shape: [rank, in_features] -> A^T: [in_features, rank]
                // B shape: [out_features, rank] -> B^T: [rank, out_features]
                // Result: [batch, seq, out_features]
                let a_val = a.val();
                let b_val = b.val();

                // Batched matmul works on 3D tensors: [batch, seq, in] @ [in, rank] -> [batch, seq, rank]
                let h = x.matmul(a_val.transpose().unsqueeze::<3>());
                // [batch, seq, rank] @ [rank, out] -> [batch, seq, out]
                let lora_out = h.matmul(b_val.transpose().unsqueeze::<3>());

                base_out + lora_out * self.scaling.0
            }
            _ => base_out,
        }
    }

    /// Forward pass for 2D tensors `[batch, features]` (used by adaln_modulation).
    pub fn forward_2d(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let base_out = self.base.forward(x.clone());

        match (&self.lora_a, &self.lora_b) {
            (Some(a), Some(b)) => {
                let a_val = a.val();
                let b_val = b.val();
                // [batch, in] @ [in, rank] -> [batch, rank]
                let h = x.matmul(a_val.transpose());
                // [batch, rank] @ [rank, out] -> [batch, out]
                let lora_out = h.matmul(b_val.transpose());
                base_out + lora_out * self.scaling.0
            }
            _ => base_out,
        }
    }

    /// Merge LoRA weights into the base linear layer.
    ///
    /// Returns a new `Linear` with `W_merged = W_base + scaling * B @ A`.
    /// The resulting linear has no LoRA overhead at inference time.
    pub fn merge(&self) -> Linear<B> {
        match (&self.lora_a, &self.lora_b) {
            (Some(a), Some(b)) => {
                let a_val = a.val(); // [rank, in_features]
                let b_val = b.val(); // [out_features, rank]

                // delta_w = B @ A -> [out_features, in_features]
                // But Linear stores weight as [in_features, out_features],
                // so we need: delta_w^T = A^T @ B^T -> [in_features, out_features]
                let delta_w = a_val.transpose().matmul(b_val.transpose()); // [in, out]

                let merged_weight = self.base.weight.val() + delta_w * self.scaling.0;

                Linear {
                    weight: Param::from_tensor(merged_weight),
                    bias: self.base.bias.clone(),
                }
            }
            _ => {
                // No LoRA, return a clone of the base
                Linear {
                    weight: Param::from_tensor(self.base.weight.val()),
                    bias: self.base.bias.clone(),
                }
            }
        }
    }

    /// Check if this layer has active LoRA adaptation.
    pub fn has_lora(&self) -> bool {
        self.lora_a.is_some() && self.lora_b.is_some()
    }

    /// Get the LoRA rank, or 0 if no LoRA is active.
    pub fn rank(&self) -> usize {
        self.lora_a
            .as_ref()
            .map(|a| a.dims()[0])
            .unwrap_or(0)
    }
}
