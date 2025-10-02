use crate::{
    DataTypeEnum, dequantize_mat4x4_block, dequantize_mat4x4_block_count,
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    quantized::matmul::QMatMulOperation,
};
use std::fmt::Write;

pub(crate) const SGEMM_MATRIX_SIZE: u32 = 16; // This is the size of the chunk we will dot at a time
pub(crate) const SGEMM_SUBGROUP_THERADS_PER_BLOCK: u32 = 2;

#[allow(clippy::too_many_arguments)]
pub(crate) fn chunked_sgemm(
    op: &QMatMulOperation,
    kernel: &mut GenericKernel,
    _: &WorkgroupShape,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    output: &TensorInput,
    n_size: &str,
    m_size: &str,
    k_size: &str,
) {
    let global_id = kernel.global_id();
    let elements_per_block = op.elements_per_block();
    let dtype = op.input_datatype;

    writeln!(
        kernel,
        "let block_x = {global_id}.x / {SGEMM_SUBGROUP_THERADS_PER_BLOCK};"
    )
    .unwrap();
    writeln!(
        kernel,
        "let index_x = {global_id}.x % {SGEMM_SUBGROUP_THERADS_PER_BLOCK};"
    )
    .unwrap();
    writeln!(kernel, "let y = {global_id}.y;").unwrap();

    // Handle batch dimensions
    writeln!(kernel, "var block_batch = {global_id}.z;").unwrap();

    // Decompose the batch index for higher-dimensional tensors
    for dim in (0..input_a.rank()).rev().skip(2) {
        let shape = input_a.shape_binding(dim);
        writeln!(kernel, "let block_batch_{dim} = block_batch % {shape};").unwrap();
        writeln!(kernel, "block_batch = block_batch / {shape};").unwrap();
    }

    writeln!(kernel, "var acc = 0.0;").unwrap();

    writeln!(
        kernel,
        "let k_block_size = ({k_size} + {elements_per_block} - 1u) / {elements_per_block};"
    )
    .unwrap();
    writeln!(kernel, "var a_index_offset = 0u;").unwrap();

    // Calculate one block sized group
    writeln!(kernel, "if block_x < {n_size} && y < {m_size} {{").unwrap();

    writeln!(kernel, "for (var k = 0u; k < k_block_size; k += 1u) {{").unwrap();

    let sub_chunks = dequantize_mat4x4_block_count(op.matrix.datatype);
    writeln!(
        kernel,
        "for (var chunk_sub_index = index_x; chunk_sub_index < {sub_chunks}; chunk_sub_index += {SGEMM_SUBGROUP_THERADS_PER_BLOCK}u) {{"
    )
    .unwrap();
    dequantize_mat4x4_block(
        kernel,
        op.matrix.datatype,
        "chunk_sub_index",
        format!("{input_b}[k + block_x * k_block_size]"),
        DataTypeEnum::F32,
        |data, code| {
            writeln!(
                code,
                "let a_index_local_offset = a_index_offset + (chunk_sub_index)*{SGEMM_MATRIX_SIZE};"
            )
            .unwrap();
            write!(code, "let a_values = mat4x4<{dtype}>(",).unwrap();
            for local in 0..SGEMM_MATRIX_SIZE {
                if local > 0 {
                    write!(code, ", ").unwrap();
                }
                write!(code, "{input_a}[").unwrap();
                let mut indices = vec![];
                // Add batch indices first1
                for dim in (0..input_a.rank()).rev().skip(2) {
                    indices.push(format!("block_batch_{dim}"));
                }
                // Then add M and K indices
                indices.push("y".to_string());
                indices.push(format!("a_index_local_offset + {local}"));
                input_a.strided_index(code, indices);
                write!(code, "]").unwrap();
            }
            writeln!(code, ");").unwrap();
            writeln!(code, "let b_values = {data};").unwrap();

            for i in 0..4 {
                writeln!(code, "acc += dot(a_values[{i}], b_values[{i}]);").unwrap();
            }
        },
    );
    writeln!(kernel, "}}").unwrap();

    writeln!(kernel, "a_index_offset += {elements_per_block};").unwrap();

    writeln!(kernel, "}}").unwrap();

    writeln!(kernel, "}}").unwrap();

    // Then write the result
    writeln!(kernel, "if block_x < {n_size} && y < {m_size} {{").unwrap();
    write!(kernel, "let output_index = ").unwrap();
    let mut output_indices = vec![];
    // Add batch indices first
    for dim in (0..output.rank()).rev().skip(2) {
        output_indices.push(format!("block_batch_{dim}"));
    }
    // Then add M and N indices
    output_indices.push("y".to_string());
    output_indices.push("block_x".to_string());
    output.strided_index(kernel, output_indices);
    writeln!(kernel, ";").unwrap();
    // Reduce over the simd group in SGEMM_SUBGROUP_THERADS_PER_BLOCK blocks
    let mut offset = SGEMM_SUBGROUP_THERADS_PER_BLOCK;
    let subgroup_size = kernel.subgroup_size();
    let subgroup_local_index = kernel.subgroup_local_index();
    while offset > 1 {
        writeln!(kernel, "if {subgroup_size} >= {offset}u {{").unwrap();
        offset /= 2;
        writeln!(
            kernel,
            "let neighbor = subgroupShuffleDown(acc, {offset}u);"
        )
        .unwrap();
        writeln!(kernel, "acc = neighbor + acc;",).unwrap();
        writeln!(kernel, "}}").unwrap();
    }
    writeln!(
        kernel,
        "if {subgroup_local_index} % {SGEMM_SUBGROUP_THERADS_PER_BLOCK} == 0 {{"
    )
    .unwrap();
    writeln!(kernel, "{output}[output_index] = acc;").unwrap();
    writeln!(kernel, "}}").unwrap();
    writeln!(kernel, "}}").unwrap();
}
