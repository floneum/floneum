use crate::{
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    quantized::matmul::QMatMulOperation,
    shift_right_scale,
    util::{maybe_vec_storage_index, maybe_vec_storage_subgroup_add, maybe_vec_storage_type},
};
use std::fmt::Write;

pub(crate) const Q_N_SGEMV_CHUNK_SIZE: u32 = 4; // This is the size of the chunk each thread will process at a time
const SUBGROUP_COUNT: u32 = 2;
const SUBGROUP_SIZE: u32 = 32;

// https://github.com/ggml-org/llama.cpp/blob/6efcd65945a98cf6883cdd9de4c8ccd8c79d219a/ggml/src/ggml-metal/ggml-metal.metal#L2329
#[allow(clippy::too_many_arguments)]
pub(crate) fn q_n_sgemv(
    op: &QMatMulOperation,
    generic_kernel: &mut GenericKernel,
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
    let workgroup_index = generic_kernel.workgroup_index();
    let subgroup_index = generic_kernel.subgroup_index();
    let subgroup_local_index = generic_kernel.subgroup_local_index();
    let elements_per_block = op.elements_per_block();

    let mut kernel = String::new();

    // Handle batch dimensions
    writeln!(&mut kernel, "let batch_idx = {workgroup_index}.z;").unwrap();

    // Find the reduce size in blocks rounded up
    writeln!(
        &mut kernel,
        "let k_block_size = ({k_size} + {elements_per_block} - 1) / {elements_per_block};"
    )
    .unwrap();

    // In index of the single element in the vector we are multiplying against
    writeln!(&mut kernel, "let workgroup_offset = {workgroup_index}.x;").unwrap();
    writeln!(&mut kernel, "let batch_offset = batch_idx * {k_size};").unwrap();
    writeln!(
        &mut kernel,
        "let row = ({SUBGROUP_COUNT} * workgroup_offset + {subgroup_index}) * {Q_N_SGEMV_CHUNK_SIZE};"
    )
    .unwrap();

    writeln!(&mut kernel, "let row_block_offset = row * k_block_size;").unwrap();

    writeln!(&mut kernel, "let thread_id = {subgroup_local_index} / 2;").unwrap();
    writeln!(
        &mut kernel,
        "let thread_local_id = {subgroup_local_index} % 2;"
    )
    .unwrap();

    writeln!(&mut kernel, "let lane_index = thread_local_id * 8;").unwrap();

    writeln!(
        &mut kernel,
        "var y_offset = thread_id * {elements_per_block} + lane_index;"
    )
    .unwrap();

    let sum_storage_type = maybe_vec_storage_type(Q_N_SGEMV_CHUNK_SIZE, dtype);
    writeln!(&mut kernel, "var sum = {sum_storage_type}();",).unwrap();

    writeln!(&mut kernel, "var cached_a_values = array<{dtype}, 16>();",).unwrap();

    // Loop over all of the blocks this thread is responsible for
    writeln!(
        &mut kernel,
        "for (var i = thread_id; i < k_block_size; i += {SUBGROUP_SIZE}/2u) {{"
    )
    .unwrap();
    {
        // First load the values of a into cached_a_values
        writeln!(&mut kernel, "var vector_sum = vec2f();").unwrap();
        writeln!(&mut kernel, "for (var j = 0u; j < 8; j += 2u) {{").unwrap();
        {
            writeln!(
                &mut kernel,
                "vector_sum[0] += {input_a}[batch_offset + j + y_offset + 0] + {input_a}[batch_offset + j + y_offset + 1];"
            )
            .unwrap();
            writeln!(
                &mut kernel,
                "cached_a_values[j + 0] = {input_a}[batch_offset + j + y_offset + 0];"
            )
            .unwrap();
            writeln!(
                &mut kernel,
                "cached_a_values[j + 1] = {input_a}[batch_offset + j + y_offset + 1] * {};",
                shift_right_scale(8)
            )
            .unwrap();

            writeln!(
                &mut kernel,
                "vector_sum[1] += {input_a}[batch_offset + j + y_offset + 16] + {input_a}[batch_offset + j + y_offset + 17];"
            )
            .unwrap();
            writeln!(
                &mut kernel,
                "cached_a_values[j + 8] = {input_a}[batch_offset + j + y_offset + 16] * {};",
                shift_right_scale(4)
            )
            .unwrap();
            writeln!(
                &mut kernel,
                "cached_a_values[j + 9] = {input_a}[batch_offset + j + y_offset + 17] * {};",
                shift_right_scale(12)
            )
            .unwrap();
        }
        writeln!(&mut kernel, "}}").unwrap();

        writeln!(
            &mut kernel,
            "let vector_total = vector_sum.x + vector_sum.y;"
        )
        .unwrap();

        writeln!(&mut kernel, "var block_offset = row_block_offset + i;").unwrap();
        if Q_N_SGEMV_CHUNK_SIZE > 1 {
            writeln!(
                &mut kernel,
                "for (var offset = 0u; offset < {Q_N_SGEMV_CHUNK_SIZE}; offset += 1u) {{"
            )
            .unwrap();
        }
        {
            block_dot(&mut kernel, op, input_b);
            let indexed = maybe_vec_storage_index(Q_N_SGEMV_CHUNK_SIZE, "sum", "offset");
            writeln!(&mut kernel, "{indexed} += product;").unwrap();
        }
        if Q_N_SGEMV_CHUNK_SIZE > 1 {
            writeln!(&mut kernel, "block_offset += k_block_size;").unwrap();
            writeln!(&mut kernel, "}}").unwrap();
        }

        writeln!(&mut kernel, "y_offset += {elements_per_block} * 16;").unwrap();
    }
    writeln!(&mut kernel, "}}").unwrap();

    // Get the sum among all threads in the subgroup
    writeln!(
        &mut kernel,
        "sum = {};",
        maybe_vec_storage_subgroup_add(Q_N_SGEMV_CHUNK_SIZE, "sum")
    )
    .unwrap();

    // If this is not the first simd thread in the workgroup, we can return early

    if Q_N_SGEMV_CHUNK_SIZE > 1 {
        writeln!(
            &mut kernel,
            "for (var offset = 0u; offset < {Q_N_SGEMV_CHUNK_SIZE}; offset += 1u) {{"
        )
        .unwrap();
    }
    {
        writeln!(&mut kernel, "if {subgroup_local_index} == 0u {{").unwrap();
        {
            // Write the output to the output tensor if this is the first thread in the workgroup
            write!(&mut kernel, "{output}[").unwrap();
            let index = if Q_N_SGEMV_CHUNK_SIZE > 1 {
                "row + offset".to_string()
            } else {
                "row".to_string()
            };
            let output_indices = vec!["batch_idx".to_string(), "0".to_string(), index];
            output.strided_index(&mut kernel, output_indices);
            let indexed = maybe_vec_storage_index(Q_N_SGEMV_CHUNK_SIZE, "sum", "offset");
            writeln!(&mut kernel, "] = {indexed};").unwrap();
        }
        writeln!(&mut kernel, "}}").unwrap();
    }
    if Q_N_SGEMV_CHUNK_SIZE > 1 {
        writeln!(&mut kernel, "}}").unwrap();
    }

    generic_kernel.push_body(&kernel);
}

fn block_dot(kernel: &mut String, op: &QMatMulOperation, input_b: &QMatrixInput) {
    match op.matrix.datatype {
        fusor_gguf::GgmlType::F32 => todo!(),
        fusor_gguf::GgmlType::F16 => todo!(),
        fusor_gguf::GgmlType::Q4_0 => block_dot_q4_0(kernel, op, input_b),
        fusor_gguf::GgmlType::Q4_1 => todo!(),
        fusor_gguf::GgmlType::Q5_0 => block_dot_q5_0(kernel, op, input_b),
        fusor_gguf::GgmlType::Q5_1 => todo!(),
        fusor_gguf::GgmlType::Q8_0 => todo!(),
        fusor_gguf::GgmlType::Q8_1 => todo!(),
        fusor_gguf::GgmlType::Q2K => todo!(),
        fusor_gguf::GgmlType::Q3K => todo!(),
        fusor_gguf::GgmlType::Q4K => todo!(),
        fusor_gguf::GgmlType::Q5K => todo!(),
        fusor_gguf::GgmlType::Q6K => todo!(),
        fusor_gguf::GgmlType::Q8K => todo!(),
    }
}

fn block_dot_q4_0(kernel: &mut String, op: &QMatMulOperation, input_b: &QMatrixInput) {
    let dtype = op.input_datatype;
    writeln!(kernel, "var chunk_sum = vec4<{dtype}>();").unwrap();

    writeln!(kernel, "for (var j = 0u; j < 8u; j += 4u) {{").unwrap();
    {
        writeln!(
            kernel,
            "var data_u32 = {input_b}[block_offset].data[lane_index / 4 + j / 4];"
        )
        .unwrap();
        writeln!(
            kernel,
            "chunk_sum[0] += cached_a_values[j + 0] * {dtype}(data_u32 & 0x000F);"
        )
        .unwrap();
        writeln!(
            kernel,
            "chunk_sum[1] += cached_a_values[j + 1] * {dtype}(data_u32 & 0x0F00);"
        )
        .unwrap();
        writeln!(
            kernel,
            "chunk_sum[2] += cached_a_values[j + 8] * {dtype}(data_u32 & 0x00F0);"
        )
        .unwrap();
        writeln!(
            kernel,
            "chunk_sum[3] += cached_a_values[j + 9] * {dtype}(data_u32 & 0xF000);"
        )
        .unwrap();
        writeln!(kernel, "data_u32 >>= 16u;").unwrap();
        writeln!(
            kernel,
            "chunk_sum[0] += cached_a_values[j + 2] * {dtype}(data_u32 & 0x000F);"
        )
        .unwrap();
        writeln!(
            kernel,
            "chunk_sum[1] += cached_a_values[j + 3] * {dtype}(data_u32 & 0x0F00);"
        )
        .unwrap();
        writeln!(
            kernel,
            "chunk_sum[2] += cached_a_values[j + 10] * {dtype}(data_u32 & 0x00F0);"
        )
        .unwrap();
        writeln!(
            kernel,
            "chunk_sum[3] += cached_a_values[j + 11] * {dtype}(data_u32 & 0xF000);"
        )
        .unwrap();
    }
    writeln!(kernel, "}}").unwrap();

    writeln!(kernel, "let product = {dtype}({input_b}[block_offset].scale) * (chunk_sum.x + chunk_sum.y + chunk_sum.z + chunk_sum.w + vector_total * -8.0);").unwrap();
}

fn block_dot_q5_0(kernel: &mut String, op: &QMatMulOperation, input_b: &QMatrixInput) {
    let dtype = op.input_datatype;
    let elements_per_block = op.elements_per_block();
    writeln!(kernel, "var chunk_sum = vec4<{dtype}>();").unwrap();

    writeln!(
        kernel,
        "let high_u32 = {input_b}[block_offset].data_high_bits[0];"
    )
    .unwrap();

    writeln!(kernel, "for (var j = 0u; j < 8u; j += 4u) {{").unwrap();
    {
        writeln!(
            kernel,
            "var low_u32 = {input_b}[block_offset].data_low_bits[lane_index / 4 + j / 4];"
        )
        .unwrap();
        writeln!(
            kernel,
            "chunk_sum[0] += cached_a_values[j + 0] * {dtype}((low_u32 & 0x000F) | ((high_u32 >> (j + 0 + lane_index) << 4) & 0x00010));"
        )
        .unwrap();
        writeln!(
            kernel,
            "chunk_sum[1] += cached_a_values[j + 1] * {dtype}((low_u32 & 0x0F00) | ((high_u32 >> (j + 1 + lane_index) << 12) & 0x01000));"
        )
        .unwrap();
        writeln!(
            kernel,
            "chunk_sum[2] += cached_a_values[j + 8] * {dtype}((low_u32 & 0x00F0) | ((high_u32 >> (j + 0 + lane_index + {elements_per_block}/2) << 8) & 0x00100));"
        )
        .unwrap();
        writeln!(
            kernel,
            "chunk_sum[3] += cached_a_values[j + 9] * {dtype}((low_u32 & 0xF000) | ((high_u32 >> (j + 1 + lane_index + {elements_per_block}/2) << 16) & 0x10000));"
        )
        .unwrap();
        writeln!(kernel, "low_u32 >>= 16u;").unwrap();
        writeln!(
            kernel,
            "chunk_sum[0] += cached_a_values[j + 2] * {dtype}((low_u32 & 0x000F) | ((high_u32 >> (j + 2 + lane_index) << 4) & 0x00010));"
        )
        .unwrap();
        writeln!(
            kernel,
            "chunk_sum[1] += cached_a_values[j + 3] * {dtype}((low_u32 & 0x0F00) | ((high_u32 >> (j + 3 + lane_index) << 12) & 0x01000));"
        )
        .unwrap();
        writeln!(
            kernel,
            "chunk_sum[2] += cached_a_values[j + 10] * {dtype}((low_u32 & 0x00F0) | ((high_u32 >> (j + 2 + lane_index + {elements_per_block}/2) << 8) & 0x00100));"
        )
        .unwrap();
        writeln!(
            kernel,
            "chunk_sum[3] += cached_a_values[j + 11] * {dtype}((low_u32 & 0xF000) | ((high_u32 >> (j + 3 + lane_index + {elements_per_block}/2) << 16) & 0x10000));"
        )
        .unwrap();
    }
    writeln!(kernel, "}}").unwrap();

    writeln!(kernel, "let product = {dtype}({input_b}[block_offset].scale) * (chunk_sum.x + chunk_sum.y + chunk_sum.z + chunk_sum.w + vector_total * -16.0);").unwrap();
}
