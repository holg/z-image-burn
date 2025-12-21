use burn::{
    Tensor,
    prelude::{Backend, ToElement},
    tensor::{DType, Float, s},
};

use crate::compat::{self, float_vec_linspace};

pub struct FlowMatchEulerDiscreteScheduler<B: Backend> {
    num_train_timesteps: i64,
    shift: f32,
    use_dynamic_shifting: bool,

    step_index: Option<usize>,
    begin_index: Option<usize>,
    num_inference_steps: Option<usize>,

    timesteps: Tensor<B, 1>,
    sigmas: Tensor<B, 1>,
    sigma_min: f32,
    sigma_max: f32,
}

impl<B: Backend> FlowMatchEulerDiscreteScheduler<B> {
    pub fn new(
        num_train_timesteps: i64,
        shift: f64,
        use_dynamic_shifting: bool,
        device: &B::Device,
    ) -> Self {
        let shift = shift as f32;
        let mut timesteps =
            float_vec_linspace(1., num_train_timesteps as f64, num_train_timesteps as usize);
        timesteps.reverse();
        let timesteps: Vec<f32> = timesteps.into_iter().map(|x| x as f32).collect();
        let timesteps = Tensor::<B, 1>::from_data_dtype(&*timesteps, device, DType::F32);
        let sigmas = timesteps.clone() / num_train_timesteps as f32;

        let sigmas = match use_dynamic_shifting {
            true => sigmas,
            false => shift * sigmas.clone() / (1. + (shift - 1.) * sigmas),
        };
        let sigma_max: <B as Backend>::FloatElem = sigmas.clone().slice(s![0]).into_scalar();

        Self {
            num_train_timesteps,
            num_inference_steps: None,
            shift,
            use_dynamic_shifting,
            step_index: None,
            begin_index: None,
            sigma_min: 0.,
            sigma_max: sigma_max.to_f32(),
            timesteps: sigmas.clone() * num_train_timesteps as f32,
            sigmas,
        }
    }

    pub fn timesteps(&self) -> Tensor<B, 1> {
        self.timesteps.clone().cast(DType::F32)
    }

    pub fn set_timesteps(
        &mut self,
        num_inference_steps: Option<usize>,
        device: &B::Device,
        sigmas: Option<Vec<f32>>,
        mu: Option<f32>,
        timesteps: Option<Vec<f32>>,
    ) {
        let num_inference_steps = num_inference_steps.unwrap_or_else(|| {
            sigmas
                .as_ref()
                .map(|s| s.len())
                // TODO: don't unwrap here (the reference implementation doesn't take this into
                // account ????)
                .unwrap_or_else(|| timesteps.as_ref().unwrap().len())
        });
        let passed_timesteps = timesteps.clone();

        self.num_inference_steps = Some(num_inference_steps);

        let sigmas = match sigmas {
            Some(sigmas) => Tensor::<B, 1>::from_data_dtype(&*sigmas, device, DType::F32),
            None => {
                let timesteps = timesteps.unwrap_or_else(|| {
                    compat::float_vec_linspace(
                        self.sigma_to_t(self.sigma_max) as f64,
                        self.sigma_to_t(self.sigma_min) as f64,
                        num_inference_steps + 1,
                    )
                    .into_iter()
                    .map(|x| x as f32)
                    .collect()
                });

                Tensor::from_data_dtype(&timesteps[..timesteps.len() - 1], device, DType::F32)
                    / self.num_train_timesteps as f32
            }
        };

        let sigmas = match self.use_dynamic_shifting {
            // TODO: don't use optional.. sigh
            true => Self::time_shift(mu.expect("mu should be present"), 1.0, sigmas),
            false => self.shift * sigmas.clone() / (1. + (self.shift - 1.) * sigmas),
        };

        let timesteps = match passed_timesteps {
            None => sigmas.clone() * self.num_train_timesteps as f32,
            Some(passed_timesteps) => {
                Tensor::from_data_dtype(passed_timesteps.as_slice(), device, DType::F32)
            }
        };

        let sigmas = Tensor::cat(
            vec![
                sigmas,
                Tensor::<B, 1, Float>::zeros([1], device).cast(DType::F32),
            ],
            0,
        );

        self.timesteps = timesteps;
        self.sigmas = sigmas;
        self.step_index = None;
        self.begin_index = None;
    }

    pub fn step(
        &mut self,
        model_output: Tensor<B, 4>,
        timestep: f32,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        if self.step_index.is_none() {
            self.init_step_index(timestep);
        }
        let step_index = self
            .step_index
            .as_mut()
            .expect("step_index was calculated previously");

        let sigma_idx = step_index.clone();
        let sigma = self.sigmas.clone().slice(s![sigma_idx]);
        let sigma_next = self.sigmas.clone().slice(s![sigma_idx + 1]);

        let dt = sigma_next - sigma;
        let sample_dtype = sample.dtype();
        let prev_sample = sample + dt.unsqueeze().cast(sample_dtype) * model_output;
        *step_index += 1;

        prev_sample
    }

    fn init_step_index(&mut self, timestep: f32) {
        match &self.begin_index {
            Some(begin_index) => self.step_index = Some(*begin_index),
            None => self.step_index = Some(self.index_for_timestep(timestep)),
        }
    }

    fn index_for_timestep(&self, timestep: f32) -> usize {
        // Avoid using argwhere which returns I64 (not supported on Metal)
        // Instead, iterate through the timesteps on CPU
        let timesteps_data = self.timesteps.clone().into_data();
        let timesteps_slice = timesteps_data
            .as_slice::<f32>()
            .expect("timesteps should be f32");

        let indices: Vec<usize> = timesteps_slice
            .iter()
            .enumerate()
            .filter(|&(_, t)| (*t - timestep).abs() < 1e-5)
            .map(|(i, _)| i)
            .collect();

        if indices.len() > 1 {
            indices[1]
        } else {
            indices.first().copied().unwrap_or(0)
        }
    }

    fn sigma_to_t(&self, sigma: f32) -> f32 {
        sigma * self.num_train_timesteps as f32
    }

    fn time_shift(mu: f32, sigma: f32, t: Tensor<B, 1>) -> Tensor<B, 1> {
        let inner: Tensor<B, 1> = 1. / t - 1.;
        mu.exp() / (mu.exp() + inner.powf_scalar(sigma))
    }
}
