use crate::{
    DataTypeEnum, dequantize_vec4_block,
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    quantized::matmul::QMatMulOperation,
};
use std::fmt::Write;

pub(crate) const SGEMM_VECTOR_SIZE: u32 = 4; // This is the size of the chunk we will dot at a time

#[allow(clippy::too_many_arguments)]
pub(crate) fn general_sgemm(
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
    let dtype = op.input_datatype;

    let mut kernel = String::new();

    writeln!(&mut kernel, "let x = {global_id}.x;").unwrap();
    writeln!(&mut kernel, "let y = {global_id}.y;").unwrap();

    writeln!(&mut kernel, "var acc = 0.0;").unwrap();

    writeln!(
        &mut kernel,
        "let k_block_size = ({k_size} + {elements_per_block} - 1u) / {elements_per_block};"
    )
    .unwrap();
    writeln!(&mut kernel, "var a_index_offset = 0u;").unwrap();

    // Calculate one block sized group
    writeln!(&mut kernel, "if x < {n_size} && y < {m_size} {{").unwrap();

    writeln!(
        &mut kernel,
        "for (var k = 0u; k < k_block_size; k += 1u) {{"
    )
    .unwrap();

    // Pack the individual dequantized values into vectors
    writeln!(&mut kernel, "let chunk = {input_b}[k + x * k_block_size];").unwrap();

    dequantize_vec4_block(
        &mut kernel,
        op.matrix.datatype,
        "chunk".to_string(),
        DataTypeEnum::F32,
        |index, data, code| {
            writeln!(code, "{{",).unwrap();
            writeln!(
                code,
                "let a_index_local_offset = a_index_offset + ({index})*{SGEMM_VECTOR_SIZE};"
            )
            .unwrap();
            write!(code, "let a_values = vec{SGEMM_VECTOR_SIZE}<{dtype}>(",).unwrap();
            for local in 0..SGEMM_VECTOR_SIZE {
                if local > 0 {
                    write!(code, ", ").unwrap();
                }
                write!(code, "{input_a}[").unwrap();
                input_a.strided_index(
                    code,
                    ["y".to_string(), format!("a_index_local_offset + {local}")],
                );
                write!(code, "]").unwrap();
            }
            writeln!(code, ");").unwrap();

            writeln!(code, "acc += dot(a_values, {data});").unwrap();
            writeln!(code, "}}").unwrap();
        },
    );

    writeln!(&mut kernel, "a_index_offset += {elements_per_block};").unwrap();

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
