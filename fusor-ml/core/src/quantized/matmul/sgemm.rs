use crate::{
    DataTypeEnum,
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    quantized::matmul::QMatMulOperation,
};
use std::fmt::Write;

use super::dequantize_block;

pub(crate) fn sgemm(
    op: &QMatMulOperation,
    generic_kernel: &mut GenericKernel,
    _: &WorkgroupShape,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    output: &TensorInput,
    n_size: &str,
    m_size: &str,
    k_size: &str,
) {
    let global_id = generic_kernel.global_id();
    let elements_per_block = op.elements_per_block();
    let mut kernel = String::new();

    writeln!(&mut kernel, "let x = {global_id}.x;").unwrap();
    writeln!(&mut kernel, "let y = {global_id}.y;").unwrap();

    writeln!(&mut kernel, "var acc = 0.0;").unwrap();

    // Calculate one block sized group
    writeln!(&mut kernel, "if x < {n_size} && y < {m_size} {{").unwrap();

    writeln!(
        &mut kernel,
        "for (var k = 0u; k < {k_size} / {elements_per_block}; k += 1u) {{"
    )
    .unwrap();

    writeln!(
        &mut kernel,
        "let chunk = {input_b}[k + x * {k_size} / {elements_per_block}];"
    )
    .unwrap();

    dequantize_block(
        &mut kernel,
        op.matrix.datatype,
        "chunk".to_string(),
        DataTypeEnum::F32,
        |i, data, code| {
            write!(code, "acc = fma({input_a}[").unwrap();
            input_a.strided_index(
                code,
                ["y".to_string(), format!("k * {elements_per_block} + {i}")],
            );
            write!(code, "], {data}, acc);").unwrap();
        },
    );

    writeln!(&mut kernel, "}}").unwrap();

    writeln!(&mut kernel, "}}").unwrap();

    // Then write the result
    writeln!(&mut kernel, "if x < {n_size} && y < {m_size} {{").unwrap();
    write!(&mut kernel, "let output_index = ").unwrap();
    output.strided_index(&mut kernel, ["y".to_string(), "x".to_string()]);
    writeln!(&mut kernel, ";").unwrap();
    writeln!(&mut kernel, "{output}[output_index] = acc;").unwrap();
    writeln!(&mut kernel, "}}").unwrap();

    generic_kernel.push_body(&kernel);
}
