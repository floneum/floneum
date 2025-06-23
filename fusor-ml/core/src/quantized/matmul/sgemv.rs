use crate::{
    DataType, DataTypeEnum, Device, Tensor, TensorData,
    compute_graph::AnyComputeKey,
    mir::{
        inputs::{MirValue, QMatrixInput, TensorInput},
        kernel::GenericKernel,
        operation::Operation,
    },
    quantized::matmul::QMatMulOperation,
};
use std::fmt::Write;

use super::{QMatrix, dequantize_block};

pub(crate) fn sgemv(
    op: &QMatMulOperation,
    kernel: &mut String,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    output: &TensorInput,
    global_id: &str,
    n_size: &str,
    // m size is always 1 for sgemv
    _m_size: &str,
    k_size: &str,
    elements_per_block: u32,
) {
    writeln!(kernel, "let x = {global_id}.x;").unwrap();

    writeln!(kernel, "var acc = 0.0;").unwrap();

    // Calculate one block sized group
    writeln!(kernel, "if x < {n_size} {{").unwrap();

    writeln!(
        kernel,
        "for (var k = 0u; k < {k_size} / {elements_per_block}; k += 1u) {{"
    )
    .unwrap();

    writeln!(
        kernel,
        "let chunk = {input_b}[k + x * {k_size} / {elements_per_block}];"
    )
    .unwrap();

    dequantize_block(
        kernel,
        op.matrix.datatype,
        "chunk".to_string(),
        DataTypeEnum::F32,
        |i, data, code| {
            write!(code, "acc = fma({input_a}[").unwrap();
            input_a.strided_index(
                code,
                ["0".to_string(), format!("k * {elements_per_block} + {i}")],
            );
            write!(code, "], {data}, acc);").unwrap();
        },
    );

    writeln!(kernel, "}}").unwrap();

    writeln!(kernel, "}}").unwrap();

    // Then write the result
    writeln!(kernel, "if x < {n_size} {{").unwrap();
    write!(kernel, "let output_index = ").unwrap();
    output.strided_index(kernel, ["0".to_string(), "x".to_string()]);
    writeln!(kernel, ";").unwrap();
    writeln!(kernel, "{output}[output_index] = acc;").unwrap();
    writeln!(kernel, "}}").unwrap();
}
