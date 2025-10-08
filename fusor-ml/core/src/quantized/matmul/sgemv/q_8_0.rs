use crate::{
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    quantized::matmul::QMatMulOperation,
    util::{maybe_vec_storage_index, maybe_vec_storage_subgroup_add, maybe_vec_storage_type},
};
use std::fmt::Write;

pub(crate) const Q_8_0_SGEMV_CHUNK_SIZE: u32 = 4; // This is the size of the chunk each thread will process at a time
const STEP_SIZE: u32 = 8;
const SUBGROUP_COUNT: u32 = 2;
const SUBGROUP_SIZE: u32 = 32;

// https://github.com/ggml-org/llama.cpp/blob/6efcd65945a98cf6883cdd9de4c8ccd8c79d219a/ggml/src/ggml-metal/ggml-metal.metal#L2452
#[allow(clippy::too_many_arguments)]
pub(crate) fn q_8_0_sgemv(
    op: &QMatMulOperation,
    kernel: &mut GenericKernel,
    _: &WorkgroupShape,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    output: &TensorInput,
    _n_size: &str,
    // m size is always 1 for sgemv
    _m_size: &str,
    k_size: &str,
) {
    let dtype = op.input_datatype;
    let global_id = kernel.global_id();
    let workgroup_index = kernel.workgroup_index();
    let subgroup_index = kernel.subgroup_index();
    let subgroup_local_index = kernel.subgroup_local_index();
    let elements_per_block = op.elements_per_block();

    // Handle batch dimensions
    writeln!(kernel, "let batch_idx = {global_id}.z;").unwrap();

    // Find the reduce size in blocks rounded up
    writeln!(
        kernel,
        "let k_block_size = ({k_size} + {elements_per_block} - 1) / {elements_per_block};"
    )
    .unwrap();

    // In index of the single element in the vector we are multiplying against
    writeln!(kernel, "let workgroup_offset = {workgroup_index}.x;").unwrap();
    writeln!(
        kernel,
        "let row = ({SUBGROUP_COUNT} * workgroup_offset + {subgroup_index}) * {Q_8_0_SGEMV_CHUNK_SIZE};"
    )
    .unwrap();

    writeln!(kernel, "let row_block_offset = row * k_block_size;").unwrap();

    writeln!(kernel, "let thread_id = {subgroup_local_index} / 4;").unwrap();
    writeln!(kernel, "let thread_local_id = {subgroup_local_index} % 4;").unwrap();

    writeln!(kernel, "let lane_index = thread_local_id * {STEP_SIZE};").unwrap();

    writeln!(
        kernel,
        "var y_offset = thread_id * {elements_per_block} + lane_index;"
    )
    .unwrap();

    let sum_storage_type = maybe_vec_storage_type(Q_8_0_SGEMV_CHUNK_SIZE, dtype);
    writeln!(kernel, "var sum = {sum_storage_type}();",).unwrap();

    writeln!(
        kernel,
        "var cached_a_values = array<{dtype}, {STEP_SIZE}>();",
    )
    .unwrap();

    // Loop over all of the blocks this thread is responsible for
    writeln!(
        kernel,
        "for (var i = thread_id; i < k_block_size; i += {SUBGROUP_SIZE}/4u) {{"
    )
    .unwrap();
    {
        // First load the values of a into cached_a_values
        for j in 0..STEP_SIZE {
            writeln!(kernel, "{{").unwrap();

            write!(kernel, "cached_a_values[{j}] = {input_a}[").unwrap();
            input_a.strided_index(
                kernel,
                vec![
                    "batch_idx".to_string(),
                    "0".to_string(),
                    format!("y_offset + {j}"),
                ],
            );
            writeln!(kernel, "];").unwrap();

            writeln!(kernel, "}}").unwrap();
        }
        writeln!(kernel, "var block_offset = row_block_offset + i;").unwrap();
        for offset in 0..Q_8_0_SGEMV_CHUNK_SIZE {
            writeln!(kernel, "{{").unwrap();

            writeln!(kernel, "var local_sum = {dtype}();").unwrap();
            for data_offset in 0..(STEP_SIZE / 4) {
                writeln!(kernel, "{{").unwrap();

                writeln!(kernel, "let block = vec4<{dtype}>(unpack4xI8({input_b}[block_offset].data[thread_local_id * 2u + {data_offset}]));").unwrap();
                writeln!(kernel, "let float_block = vec4<{dtype}>(cached_a_values[{data_offset} * 4u + 0], cached_a_values[{data_offset} * 4u + 1], cached_a_values[{data_offset} * 4u + 2], cached_a_values[{data_offset} * 4u + 3]);").unwrap();
                writeln!(kernel, "local_sum += dot(block, float_block);").unwrap();

                writeln!(kernel, "}}").unwrap();
            }
            let indexed = maybe_vec_storage_index(Q_8_0_SGEMV_CHUNK_SIZE, "sum", offset);
            writeln!(
                kernel,
                "{indexed} += local_sum * {dtype}({input_b}[block_offset].scale);"
            )
            .unwrap();

            writeln!(kernel, "block_offset += k_block_size;").unwrap();
            writeln!(kernel, "}}").unwrap();
        }

        writeln!(kernel, "y_offset += {elements_per_block} * {STEP_SIZE};").unwrap();
    }
    writeln!(kernel, "}}").unwrap();

    // Get the sum among all threads in the subgroup
    writeln!(
        kernel,
        "sum = {};",
        maybe_vec_storage_subgroup_add(Q_8_0_SGEMV_CHUNK_SIZE, "sum")
    )
    .unwrap();

    for offset in 0..Q_8_0_SGEMV_CHUNK_SIZE {
        writeln!(kernel, "{{").unwrap();

        // If this is not the first simd thread in the workgroup, we can return early
        writeln!(kernel, "if {subgroup_local_index} == 0u {{").unwrap();
        {
            // Write the output to the output tensor if this is the first thread in the workgroup
            write!(kernel, "{output}[").unwrap();
            let index = format!("row + {offset}");
            let output_indices = vec!["batch_idx".to_string(), "0".to_string(), index];
            output.strided_index(kernel, output_indices);
            let indexed = maybe_vec_storage_index(Q_8_0_SGEMV_CHUNK_SIZE, "sum", offset);
            writeln!(kernel, "] = {indexed};").unwrap();
        }
        writeln!(kernel, "}}").unwrap();

        writeln!(kernel, "}}").unwrap();
    }
}
