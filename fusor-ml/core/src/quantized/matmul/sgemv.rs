use crate::{
    mir::{
        globals::{KernelGlobalSpace, KernelGlobalType, VectorType},
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    }, quantized::matmul::QMatMulOperation, unrolled_dequantize_block, DataTypeEnum
};
use std::fmt::Write;

use super::dequantize_block;

pub(crate) const SGEMV_CHUNK_SIZE: u32 = 4; // This is the size of the chunk we will dequantize at a time

pub(crate) fn sgemv(
    op: &QMatMulOperation,
    generic_kernel: &mut GenericKernel,
    workgroup_size: &WorkgroupShape,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    output: &TensorInput,
    _n_size: &str,
    // m size is always 1 for sgemv
    _m_size: &str,
    k_size: &str,
) {
    let blocksize = workgroup_size.x();
    let dtype = op.input_datatype;
    let local_data = generic_kernel.add_global_array(
        KernelGlobalSpace::Workgroup,
        KernelGlobalType::Vector(VectorType::new(SGEMV_CHUNK_SIZE.to_string(), dtype)),
        blocksize.to_string(),
    );
    let workgroup_index = generic_kernel.workgroup_index();
    let workgroup_local_index = generic_kernel.workgroup_local_index();
    let subgroup_id = generic_kernel.subgroup_index();
    let subgroup_local_id = generic_kernel.subgroup_local_index();
    let subgroups_per_workgroup = generic_kernel.subgroups_per_workgroup();
    let elements_per_block = op.elements_per_block();
    let mut kernel = String::new();

    // In index of the single element in the vector we are multiplying against
    writeln!(
        &mut kernel,
        "let workgroup_offset = {workgroup_index}.x * {SGEMV_CHUNK_SIZE};"
    )
    .unwrap();

    writeln!(&mut kernel, "var acc = vec{SGEMV_CHUNK_SIZE}<{dtype}>();").unwrap();

    // First merge values on each thread individually. We divide the column allocated to the thread group into equal sized buckets
    // Round up
    writeln!(
        &mut kernel,
        "let bucket_size = ({k_size} + {blocksize}u - 1) / {blocksize}u;"
    )
    .unwrap();

    // Round the bucket size to the nearest multiple of elements per block
    writeln!(
        &mut kernel,
        "let bucket_block_size = (bucket_size + {elements_per_block} - 1) / {elements_per_block};"
    )
    .unwrap();

    // Find the reduce size in blocks rounded up
    writeln!(
        &mut kernel,
        "let k_block_size = ({k_size} + {elements_per_block} - 1) / {elements_per_block};"
    )
    .unwrap();

    // Find this threads position in the workgroup
    writeln!(
        &mut kernel,
        "let base_axis_index = {workgroup_local_index} * bucket_block_size;"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "let end_axis_index = min({workgroup_local_index} * bucket_block_size + bucket_block_size, k_block_size);"
    )
    .unwrap();
    writeln!(&mut kernel, "var index = base_axis_index;").unwrap();

    let chunk_blocks = elements_per_block / SGEMV_CHUNK_SIZE;
    debug_assert!(elements_per_block % SGEMV_CHUNK_SIZE == 0);
    writeln!(
        &mut kernel,
        "var a_cache = array<vec{SGEMV_CHUNK_SIZE}<{dtype}>, {chunk_blocks}>();"
    )
    .unwrap();

    // Loop over all of the blocks this thread is responsible for
    writeln!(&mut kernel, "while (index < end_axis_index) {{").unwrap();
    {
        // Load all elements of a into a cache first
        writeln!(
            &mut kernel,
            "for (var i = 0u; i < {chunk_blocks}; i += 1u) {{"
        )
        .unwrap();
        {
            write!(&mut kernel, "a_cache[i] = vec{SGEMV_CHUNK_SIZE}(").unwrap();
            for i in 0..SGEMV_CHUNK_SIZE {
                if i > 0 {
                    write!(&mut kernel, ", ").unwrap();
                }
                write!(&mut kernel, "{input_a}[").unwrap();
                input_a.strided_index(
                    &mut kernel,
                    [
                        "0".to_string(),
                        format!("index * {elements_per_block} + i * {SGEMV_CHUNK_SIZE} + {i}"),
                    ],
                );
                write!(&mut kernel, "]").unwrap();
            }
            writeln!(&mut kernel, ");").unwrap();
        }
        writeln!(&mut kernel, "}}").unwrap();

        writeln!(
            &mut kernel,
            "for (var offset = 0u; offset < {SGEMV_CHUNK_SIZE}; offset += 1u) {{"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "let chunk = {input_b}[(workgroup_offset + offset) * k_block_size + index];"
        )
        .unwrap();

        unrolled_dequantize_block(
            &mut kernel,
            op.matrix.datatype,
            "chunk".to_string(),
            DataTypeEnum::F32,
            |index, data, code| {
                writeln!(code, "let dequantized_{index} = {data};").unwrap();
            },
        );

        // Pack the individual dequantized values into vectors
        for i in 0..chunk_blocks {
            write!(&mut kernel, "let dequantized_vec_{i} = vec{SGEMV_CHUNK_SIZE}<{dtype}>(").unwrap();
            for j in 0..SGEMV_CHUNK_SIZE {
                if j > 0 {
                    write!(&mut kernel, ", ").unwrap();
                }
                let index = i * SGEMV_CHUNK_SIZE + j;
                write!(
                    &mut kernel,
                    "dequantized_{index}"
                )
                .unwrap();
            }
            writeln!(&mut kernel, ");").unwrap();
        }

        for i in 0..chunk_blocks {
            write!(
                &mut kernel,
                "acc[offset] += dot(a_cache[{i}], dequantized_vec_{i});"
            )
            .unwrap();
        }
        writeln!(&mut kernel, "}}").unwrap();

        writeln!(&mut kernel, "index += 1;").unwrap();
    }

    writeln!(&mut kernel, "}}").unwrap();

    // Get the sum among all threads in the subgroup
    writeln!(&mut kernel, "acc = subgroupAdd(acc);").unwrap();

    // Write the output to the workgroup memory if this is the first thread in the subgroup
    writeln!(&mut kernel, "if {subgroup_local_id} == 0u {{").unwrap();
    {
        writeln!(&mut kernel, "{local_data}[{subgroup_id}] = acc;").unwrap();
    }
    writeln!(&mut kernel, "}}").unwrap();

    // Wait until all threads have written to the workgroup shared memory
    writeln!(&mut kernel, "workgroupBarrier();").unwrap();

    // Then if this is the first subgroup, do one final shuffle down reduction
    writeln!(&mut kernel, "if {subgroup_id} == 0u {{").unwrap();
    {
        // Copy over the final value from each subgroup from the workgroup shared memory to the acc variable
        writeln!(
            &mut kernel,
            "if {subgroup_local_id} < {subgroups_per_workgroup} {{"
        )
        .unwrap();
        {
            writeln!(&mut kernel, "acc = {local_data}[{subgroup_local_id}];").unwrap();
        }
        writeln!(&mut kernel, "}}").unwrap();
        writeln!(&mut kernel, "else {{").unwrap();
        {
            writeln!(&mut kernel, "acc = vec{SGEMV_CHUNK_SIZE}<{dtype}>();").unwrap();
        }
        writeln!(&mut kernel, "}}").unwrap();

        // Finally get the final sum across all threads in the workgroup
        writeln!(&mut kernel, "acc = subgroupAdd(acc);").unwrap();

        // Write the output to the output tensor if this is the first thread in the workgroup
        for offset in 0..SGEMV_CHUNK_SIZE {
            write!(&mut kernel, "{output}[").unwrap();
            output.strided_index(
                &mut kernel,
                ["0".to_string(), format!("workgroup_offset + {offset}")],
            );
            writeln!(&mut kernel, "] = acc[{offset}];").unwrap();
        }
    }
    writeln!(&mut kernel, "}}").unwrap();

    generic_kernel.push_body(&kernel);
}
