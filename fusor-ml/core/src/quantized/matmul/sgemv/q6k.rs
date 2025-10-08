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

pub(crate) const Q6K_SGEMV_CHUNK_SIZE: u32 = 2; // This is the size of the chunk each thread will process at a time
const PRELOAD: bool = false;

// https://github.com/ggml-org/llama.cpp/blob/6efcd65945a98cf6883cdd9de4c8ccd8c79d219a/ggml/src/ggml-metal/ggml-metal.metal#L5564
#[allow(clippy::too_many_arguments)]
pub(crate) fn q6k_sgemv(
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
        "let row = (2 * workgroup_offset + {subgroup_index}) * {Q6K_SGEMV_CHUNK_SIZE};"
    )
    .unwrap();

    writeln!(kernel, "let block_offset = row * k_block_size;").unwrap();

    writeln!(kernel, "let thread_id = {subgroup_local_index} / 2;").unwrap();
    writeln!(kernel, "let thread_local_id = {subgroup_local_index} % 2;").unwrap();
    writeln!(kernel, "let half_subgroup_id = thread_id / 8;").unwrap();
    writeln!(kernel, "let half_subgroup_local_id = thread_id % 8;").unwrap();
    const CHUNKS_PER_STRIPE: usize = 4;
    writeln!(
        kernel,
        "let stripe_offset = half_subgroup_local_id * {CHUNKS_PER_STRIPE};"
    )
    .unwrap();
    writeln!(kernel, "let scale_index_offset = half_subgroup_id * 2;").unwrap();

    writeln!(kernel, "let scale_pair_offset = stripe_offset / 16;").unwrap();

    // Every half subgroup handles 128 normal values
    writeln!(
        kernel,
        "let y_offset = 128 * half_subgroup_id + stripe_offset;"
    )
    .unwrap();
    // Every half subgroup handles 64 lower values (half bytes)
    writeln!(
        kernel,
        "let q_offset_l = 16 * half_subgroup_id + stripe_offset / 4;"
    )
    .unwrap();
    // Every half subgroup handles 32 upper values (quarter bytes)
    writeln!(
        kernel,
        "let q_offset_h = 8 * half_subgroup_id + stripe_offset / 4;"
    )
    .unwrap();

    let sum_storage_type = maybe_vec_storage_type(Q6K_SGEMV_CHUNK_SIZE, dtype);
    writeln!(kernel, "var sum = {sum_storage_type}();",).unwrap();

    if PRELOAD {
        writeln!(kernel, "var cached_a_values = array<{dtype}, 16>();",).unwrap();
    }

    // Loop over all of the blocks this thread is responsible for
    writeln!(
        kernel,
        "for (var i = thread_local_id; i < k_block_size; i += 2) {{"
    )
    .unwrap();
    {
        // First load the values of a into cached_a_values
        writeln!(
            kernel,
            "let vector_offset = i * {elements_per_block} + y_offset;"
        )
        .unwrap();
        let load_value = |kernel: &mut GenericKernel, j: &str, offset: u32| {
            write!(kernel, "{input_a}[").unwrap();
            input_a.strided_index(
                kernel,
                vec![
                    "batch_idx".to_string(),
                    "0".to_string(),
                    format!("{j} + vector_offset + {}", offset * 32),
                ],
            );
            write!(kernel, "]").unwrap();
        };
        if PRELOAD {
            writeln!(kernel, "for (var j = 0u; j < 4; j += 1u) {{").unwrap();
            for offset in 0..4 {
                write!(kernel, "cached_a_values[j*4u + {offset}] = ",).unwrap();
                load_value(kernel, "j", offset);
                writeln!(kernel, ";").unwrap();
            }
            writeln!(kernel, "}}").unwrap();
        }

        if Q6K_SGEMV_CHUNK_SIZE > 1 {
            writeln!(
                kernel,
                "for (var offset = 0u; offset < {Q6K_SGEMV_CHUNK_SIZE}; offset += 1u) {{"
            )
            .unwrap();
        }
        {
            if Q6K_SGEMV_CHUNK_SIZE > 1 {
                writeln!(
                    kernel,
                    "let local_block_offset = i + block_offset + offset * k_block_size;"
                )
                .unwrap();
            } else {
                writeln!(kernel, "let local_block_offset = i + block_offset;").unwrap();
            }
            writeln!(kernel, "let low_offset_1 = q_offset_l;").unwrap();
            writeln!(
                kernel,
                "let low_bytes_1 = unpack4xU8({input_b}[local_block_offset].data_low_bits[low_offset_1]);"
            )
            .unwrap();
            writeln!(kernel, "let low_offset_2 = q_offset_l + 8;").unwrap();
            writeln!(kernel, "let low_bytes_2 = unpack4xU8({input_b}[local_block_offset].data_low_bits[low_offset_2]);").unwrap();
            writeln!(kernel, "let high_offset = q_offset_h;").unwrap();
            writeln!(
                kernel,
                "let high_bytes = unpack4xU8({input_b}[local_block_offset].data_high_bits[high_offset]);"
            )
            .unwrap();
            writeln!(kernel, "let scale_offset = scale_index_offset;").unwrap();
            writeln!(
                kernel,
                "let scale_chunk_1 = unpack4xI8({input_b}[local_block_offset].scales[scale_offset]);"
            )
            .unwrap();
            writeln!(
                kernel,
                "let scale_chunk_2 = unpack4xI8({input_b}[local_block_offset].scales[scale_offset + 1]);"
            )
            .unwrap();
            writeln!(
                kernel,
                "let scales = vec4({dtype}(scale_chunk_1[scale_pair_offset]), {dtype}(scale_chunk_1[2 + scale_pair_offset]), {dtype}(scale_chunk_2[scale_pair_offset]), {dtype}(scale_chunk_2[2 + scale_pair_offset]));"
            )
            .unwrap();

            writeln!(
                kernel,
                "let scale = {dtype}({input_b}[local_block_offset].scale);"
            )
            .unwrap();

            writeln!(kernel, "var sums = vec4f();").unwrap();
            writeln!(kernel, "for (var j = 0u; j < 4u; j += 1u) {{").unwrap();
            {
                let first_four_bytes = 0b00001111u8;
                let first_two_bytes = 0b00000011u8;
                let second_two_bytes = 0b00001100u8;
                let third_two_bytes = 0b00110000u8;
                let fourth_two_bytes = 0b11000000u8;
                let get_value = |kernel: &mut GenericKernel, j: &str, offset: u32| {
                    if PRELOAD {
                        write!(kernel, "cached_a_values[{j} + {offset}]").unwrap();
                    } else {
                        load_value(kernel, j, offset);
                    }
                };
                write!(kernel, "sums[0] += ").unwrap();
                get_value(kernel, "j", 0);
                writeln!(kernel,"* {dtype}(i32((low_bytes_1[j] & {first_four_bytes}) | ((high_bytes[j] & {first_two_bytes})  << 4)) - 32);").unwrap();
                write!(kernel, "sums[1] += ").unwrap();
                get_value(kernel, "j", 1);
                writeln!(kernel,"* {dtype}(i32((low_bytes_2[j] & {first_four_bytes}) | ((high_bytes[j] & {second_two_bytes}) << 2)) - 32);").unwrap();
                write!(kernel, "sums[2] += ").unwrap();
                get_value(kernel, "j", 2);
                writeln!(kernel,"* {dtype}(i32((low_bytes_1[j]                 >> 4) | ((high_bytes[j] & {third_two_bytes})  << 0)) - 32);").unwrap();
                write!(kernel, "sums[3] += ").unwrap();
                get_value(kernel, "j", 3);
                writeln!(kernel,"* {dtype}(i32((low_bytes_2[j]                 >> 4) | ((high_bytes[j] & {fourth_two_bytes}) >> 2)) - 32);").unwrap();
            }
            writeln!(kernel, "}}").unwrap();
            let indexed = maybe_vec_storage_index(Q6K_SGEMV_CHUNK_SIZE, "sum", "offset");
            writeln!(kernel, "{indexed} += scale * dot(sums, scales);").unwrap();
        }
        if Q6K_SGEMV_CHUNK_SIZE > 1 {
            writeln!(kernel, "}}").unwrap();
        }
    }
    writeln!(kernel, "}}").unwrap();

    // Get the sum among all threads in the subgroup
    writeln!(
        kernel,
        "sum = {};",
        maybe_vec_storage_subgroup_add(Q6K_SGEMV_CHUNK_SIZE, "sum")
    )
    .unwrap();

    // If this is not the first simd thread in the workgroup, we can return early
    writeln!(kernel, "if {subgroup_local_index} != 0u {{ return; }}").unwrap();

    if Q6K_SGEMV_CHUNK_SIZE > 1 {
        writeln!(
            kernel,
            "for (var offset = 0u; offset < {Q6K_SGEMV_CHUNK_SIZE}; offset += 1u) {{"
        )
        .unwrap();
    }
    {
        // Write the output to the output tensor if this is the first thread in the workgroup
        write!(kernel, "{output}[").unwrap();
        let index = if Q6K_SGEMV_CHUNK_SIZE > 1 {
            "row + offset".to_string()
        } else {
            "row".to_string()
        };
        let output_indices = vec!["batch_idx".to_string(), "0".to_string(), index];
        output.strided_index(kernel, output_indices);
        let indexed = maybe_vec_storage_index(Q6K_SGEMV_CHUNK_SIZE, "sum", "offset");
        writeln!(kernel, "] = {indexed};").unwrap();
    }
    if Q6K_SGEMV_CHUNK_SIZE > 1 {
        writeln!(kernel, "}}").unwrap();
    }
}
