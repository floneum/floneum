use crate::{
    DataTypeEnum,
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    quantized::matmul::QMatMulOperation,
    unrolled_dequantize_block,
};
use std::fmt::Write;

pub(crate) const SGEMM_VECTOR_SIZE: u32 = 4; // This is the size of the chunk we will dot at a time

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
    let chunk_blocks = elements_per_block / SGEMM_VECTOR_SIZE;
    let dtype = op.input_datatype;

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

    unrolled_dequantize_block(
        &mut kernel,
        op.matrix.datatype,
        "chunk".to_string(),
        DataTypeEnum::F32,
        |index, data, code| {
            write!(code, "let dequantized_{index} = {data};").unwrap();
        },
    );

    // Pack the individual dequantized values into vectors
    for i in 0..chunk_blocks {
        write!(
            &mut kernel,
            "let dequantized_vec_{i} = vec{SGEMM_VECTOR_SIZE}<{dtype}>("
        )
        .unwrap();
        for j in 0..SGEMM_VECTOR_SIZE {
            if j > 0 {
                write!(&mut kernel, ", ").unwrap();
            }
            let index = i * SGEMM_VECTOR_SIZE + j;
            write!(&mut kernel, "dequantized_{index}").unwrap();
        }
        writeln!(&mut kernel, ");").unwrap();
        write!(
            &mut kernel,
            "let a_values_{i} = vec{SGEMM_VECTOR_SIZE}<{dtype}>(",
        )
        .unwrap();
        for local in 0..SGEMM_VECTOR_SIZE {
            if local > 0 {
                write!(&mut kernel, ", ").unwrap();
            }
            write!(&mut kernel, "{input_a}[").unwrap();
            input_a.strided_index(
                &mut kernel,
                [
                    "y".to_string(),
                    format!("k * {elements_per_block} + {i}*{SGEMM_VECTOR_SIZE} + {local}"),
                ],
            );
            write!(&mut kernel, "]").unwrap();
        }
        writeln!(&mut kernel, ");").unwrap();

        writeln!(
            &mut kernel,
            "acc += dot(a_values_{i}, dequantized_vec_{i});"
        )
        .unwrap();
    }

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
