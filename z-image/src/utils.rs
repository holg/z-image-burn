use burn::tensor::{DType, quantization::QuantParam};

/// Get the DType of a float tensor used in actual operations. Workaround for quantized floats being
/// a special case.
pub fn effective_dtype(actual: DType) -> DType {
    match actual {
        DType::F64 | DType::F32 | DType::BF16 | DType::F16 | DType::Flex32 => actual,
        DType::QFloat(quant_scheme) => match quant_scheme.param {
            QuantParam::F32 => DType::F32,
            QuantParam::F16 => DType::F16,
            QuantParam::BF16 => DType::BF16,
            _ => unimplemented!(),
        },
        _ => unimplemented!(),
    }
}
