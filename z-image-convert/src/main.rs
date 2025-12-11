use std::{path::PathBuf, process::ExitCode};

use burn::{
    Tensor,
    backend::{Cpu, cpu::CpuDevice},
    module::{Module, ModuleMapper, Param, Quantizer},
    tensor::{
        FloatDType,
        quantization::{Calibration, QuantScheme, QuantValue},
    },
};
use burn_store::{BurnpackStore, ModuleSnapshot};
use clap::{Parser, ValueEnum};
use z_image::modules::transformer::ZImageModelConfig;

#[derive(clap::Parser)]
struct Args {
    /// The file path to save the resulting model to.
    ///
    /// A file extension is automatically added if not provided. Output is always a `.bpk` model.
    #[arg(short, long)]
    output: PathBuf,
    /// The model file to convert.
    ///
    /// Supports the following formats:
    /// - `.bpk` files native to the project
    /// - `.safetensors` files with ComfyUI's format
    #[arg(short, long)]
    input: PathBuf,
    /// The output dtype to convert to.
    ///
    /// Defaults to the dtype found in the input if not specified.
    #[arg(short, long, value_enum)]
    dtype: Option<Format>,
    /// Whether to overwrite any existing files when saving.
    #[arg(long, action)]
    overwrite: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Format {
    FP32,
    FP16,
    BF16,
    Q8,
    Q8E5M2,
    Q8E4M3,
    Q8S,
}

type B = Cpu;

fn main() -> ExitCode {
    let args = Args::parse();

    let device = CpuDevice::default();
    println!("Loading model...");
    let model = ZImageModelConfig::default()
        .init::<B>(&device)
        .with_weights(args.input);
    let model = match model {
        Ok(m) => m,
        Err(err) => {
            eprintln!("Failed to load model: {err}");
            return ExitCode::FAILURE;
        }
    };

    let model = match args.dtype {
        None => model,
        Some(f @ (Format::Q8 | Format::Q8E5M2 | Format::Q8E4M3 | Format::Q8S)) => {
            println!("Quantizing model...");
            let scheme = match f {
                Format::Q8 => QuantScheme {
                    value: QuantValue::Q8F,
                    ..Default::default()
                },
                Format::Q8E5M2 => QuantScheme {
                    value: QuantValue::E5M2,
                    ..Default::default()
                },
                Format::Q8E4M3 => QuantScheme {
                    value: QuantValue::E4M3,
                    ..Default::default()
                },
                Format::Q8S => QuantScheme {
                    value: QuantValue::Q8S,
                    ..Default::default()
                },
                _ => unreachable!(),
            };
            let mut quantizer = Quantizer {
                calibration: Calibration::MinMax,
                scheme,
            };

            model.quantize_weights(&mut quantizer)
        }
        Some(f @ (Format::FP32 | Format::BF16 | Format::FP16)) => {
            println!("Casting model...");
            let dtype = match f {
                Format::FP32 => FloatDType::F32,
                Format::FP16 => FloatDType::F16,
                Format::BF16 => FloatDType::BF16,
                _ => unreachable!(),
            };
            model.map(&mut ModuleTypeChanger { dtype })
        }
    };

    println!("Saving model...");
    let mut out_weights = BurnpackStore::from_file(args.output).overwrite(args.overwrite);
    if let Err(err) = model.save_into(&mut out_weights) {
        eprintln!("Failed to save model: {err}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

struct ModuleTypeChanger {
    dtype: FloatDType,
}

impl ModuleMapper<B> for ModuleTypeChanger {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        Param::from_mapped_value(id, tensor.cast(self.dtype), mapper)
    }
}
