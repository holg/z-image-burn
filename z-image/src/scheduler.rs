use burn::{
    Tensor,
    prelude::{Backend, ToElement},
    tensor::{DType, Float, s},
};

use crate::{
    compat::{self, float_vec_linspace},
    modules::transformer::ZImageModel,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shift {
    /// Use a pre-defined shift value, the normal way this scheduler is used.
    ///
    /// The actual value usually depends on the value used during training.
    Constant(f64),
    /// Apply timestep shifting dynamically based on image resolution.
    ///
    /// This is in theory meant to improve image quality for higher resolutions. See [Self::dynamic]
    /// for the recommended way to calculate this value.
    Dynamic { mu: f64 },
}

impl Default for Shift {
    /// Initialize with a constant shift of 3.
    fn default() -> Self {
        Self::Constant(3.0)
    }
}

impl Shift {
    /// Calculate dynamic shift based on image size.
    pub fn dynamic(image_seq_len: usize, opts: DynamicShiftOptions) -> Self {
        let m = (opts.max_shift - opts.base_shift) / (opts.max_seq_len - opts.base_seq_len) as f64;
        let b = opts.base_shift - m * opts.base_seq_len as f64;
        Shift::Dynamic {
            mu: image_seq_len as f64 * m + b,
        }
    }
}

/// Helper for determining the mu value for dynamic shift.
///
/// Use [DynamicShiftConfig::Default] to initialize with recommended values.
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicShiftOptions {
    pub base_seq_len: usize,
    pub max_seq_len: usize,
    pub base_shift: f64,
    pub max_shift: f64,
}

impl Default for DynamicShiftOptions {
    fn default() -> Self {
        Self {
            base_seq_len: 256,
            max_seq_len: 4096,
            base_shift: 0.5,
            max_shift: 1.15,
        }
    }
}

pub struct FlowMatchEulerDiscreteScheduler<B: Backend> {
    timesteps: Tensor<B, 1>,
    sigmas: Tensor<B, 1>,
}

impl<B: Backend> FlowMatchEulerDiscreteScheduler<B> {
    pub fn new(
        num_inference_steps: u32,
        num_train_timesteps: u32,
        shift: Shift,
        device: &B::Device,
    ) -> Self {
        let timesteps = Tensor::<B, 1>::from_data_dtype(
            &*float_vec_linspace(1., num_train_timesteps as f64, num_train_timesteps as usize),
            device,
            DType::F64,
        )
        .slice(s![..;-1]);
        let sigmas = timesteps.clone() / num_train_timesteps as f64;

        let sigmas = match shift {
            Shift::Dynamic { .. } => sigmas,
            Shift::Constant(shift) => shift * sigmas.clone() / (1. + (shift - 1.) * sigmas),
        };
        let sigma_max = sigmas.clone().slice(s![0]).into_scalar().to_f64();
        // The implementation in the Z-Image repo sets this to zero while diffusers does the
        // following. I'm not sure why the official repo does this differently so I will follow
        // diffusers here unless I find this somehow makes the output worse.
        let sigma_min = sigmas.clone().slice(s![-1]).into_scalar().to_f64();

        let sigmas = {
            let timesteps = compat::float_vec_linspace(
                sigma_max * num_train_timesteps as f64,
                sigma_min * num_train_timesteps as f64,
                (num_inference_steps + 1) as usize,
            );

            Tensor::from_data_dtype(&timesteps[..timesteps.len() - 1], device, DType::F64)
                / num_train_timesteps
        };

        let sigmas = match shift {
            Shift::Dynamic { mu } => Self::time_shift(mu, 1.0, sigmas),
            Shift::Constant(shift) => shift * sigmas.clone() / (1. + (shift - 1.) * sigmas),
        };

        let timesteps = sigmas.clone() * num_train_timesteps;

        let sigmas = Tensor::cat(
            vec![
                sigmas,
                Tensor::<B, 1, Float>::zeros([1], device).cast(DType::F64),
            ],
            0,
        );

        Self { timesteps, sigmas }
    }

    pub fn timesteps(&self) -> Tensor<B, 1> {
        self.timesteps.clone().cast(DType::F32)
    }

    pub fn step(
        &self,
        model_output: Tensor<B, 4>,
        timestep: u32,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma_idx = timestep as usize;
        let sigma = self.sigmas.clone().slice(s![sigma_idx]);
        let sigma_next = self.sigmas.clone().slice(s![sigma_idx + 1]);

        let dt = sigma_next - sigma;
        let sample_dtype = sample.dtype();
        let prev_sample = sample + dt.unsqueeze().cast(sample_dtype) * model_output;

        prev_sample
    }

    pub fn sampler<'a>(
        &'a self,
        model: &'a ZImageModel<B>,
        input_latents: Tensor<B, 4>,
        prompt: Tensor<B, 3>,
    ) -> FlowMatchEulerDiscreteSampler<'a, B> {
        FlowMatchEulerDiscreteSampler {
            scheduler: self,
            model,
            latents: input_latents,
            prompt,
            step: 0,
        }
    }

    fn time_shift(mu: f64, sigma: f64, t: Tensor<B, 1>) -> Tensor<B, 1> {
        let inner: Tensor<B, 1> = 1. / t - 1.;
        mu.exp() / (mu.exp() + inner.powf_scalar(sigma))
    }
}

pub struct FlowMatchEulerDiscreteSampler<'a, B: Backend> {
    scheduler: &'a FlowMatchEulerDiscreteScheduler<B>,
    model: &'a ZImageModel<B>,
    step: u32,
    latents: Tensor<B, 4>,
    prompt: Tensor<B, 3>,
}

impl<'a, B: Backend> FlowMatchEulerDiscreteSampler<'a, B> {
    /// Whether all sampling steps have been executed.
    pub fn finished(&self) -> bool {
        self.remaining_steps() <= 0
    }

    /// The number of steps remaining to finish sampling.
    pub fn remaining_steps(&self) -> u32 {
        (self.scheduler.timesteps.dims()[0] as u32).saturating_sub(self.step)
    }

    /// Run one sampling step if there are any steps left to run.
    pub fn step(&mut self) {
        if self.finished() {
            return;
        }

        let timestep = self
            .scheduler
            .timesteps
            .clone()
            .slice(s![self.step as usize]);

        let timestep = (timestep / 1000.).repeat(&[self.latents.dims()[0]]);

        let model_output = self
            .model
            .forward(self.latents.clone(), timestep, self.prompt.clone());

        self.latents = self
            .scheduler
            .step(model_output, self.step, self.latents.clone());
        self.step += 1;
    }

    /// Peform all (remaining) inference steps and return the resulting latents, consuming the
    /// sampler.
    pub fn result(mut self) -> Tensor<B, 4> {
        while !self.finished() {
            self.step();
        }

        self.latents
    }
}

impl<'a, B: Backend> Iterator for FlowMatchEulerDiscreteSampler<'a, B> {
    type Item = Tensor<B, 4>;

    /// Execute the next sampling step any return the resulting latents.
    ///
    /// # Returns
    ///
    /// The resulting latent after performing this step.
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished() {
            return None;
        }

        self.step();
        Some(self.latents.clone())
    }
}
