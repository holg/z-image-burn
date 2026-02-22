use std::{path::PathBuf, process::ExitCode};

use burn::{
    Tensor,
    backend::{NdArray, ndarray::NdArrayDevice},
    module::{Module, ModuleMapper, Param, Quantizer},
    tensor::{
        FloatDType,
        quantization::{Calibration, QuantScheme, QuantStore, QuantValue},
    },
};
use burn_store::{BurnpackStore, ModuleSnapshot};
use clap::{Parser, ValueEnum};
use qwen3_burn::Qwen3Config;

#[derive(clap::Parser)]
struct Args {
    #[arg(short, long)]
    output: PathBuf,
    #[arg(short, long)]
    input: PathBuf,
    #[arg(short, long, value_enum)]
    dtype: Option<Format>,
    #[arg(long, action)]
    overwrite: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Format {
    FP32,
    FP16,
    BF16,
    Q8,
    Q8S,
    Q4,
    Q4S,
}

type B = NdArray;

fn main() -> ExitCode {
    let args = Args::parse();

    let device = NdArrayDevice::Cpu;
    println!("Loading model...");
    let model = Qwen3Config::z_image_text_encoder()
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
        Some(f @ (Format::Q8 | Format::Q8S | Format::Q4 | Format::Q4S)) => {
            println!("Quantizing model to {:?}...", f);
            let scheme = match f {
                Format::Q8 => QuantScheme {
                    value: QuantValue::Q8F,
                    store: QuantStore::Native,
                    ..Default::default()
                },
                Format::Q8S => QuantScheme {
                    value: QuantValue::Q8S,
                    store: QuantStore::Native,
                    ..Default::default()
                },
                Format::Q4 => QuantScheme {
                    value: QuantValue::Q4F,
                    store: QuantStore::Native,
                    ..Default::default()
                },
                Format::Q4S => QuantScheme {
                    value: QuantValue::Q4S,
                    store: QuantStore::Native,
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

    println!("Done!");
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
