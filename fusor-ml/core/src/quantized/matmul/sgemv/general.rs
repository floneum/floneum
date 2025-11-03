use crate::{
    DataTypeEnum, dequantize_vec4_block,
    mir::{
        globals::KernelGlobalSpace,
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    quantized::matmul::{
        QMatMulOperation,
        sgemv::{SGEMV_CHUNK_SIZE, SGEMV_VECTOR_SIZE},
    },
    util::{
        maybe_vec_storage_index, maybe_vec_storage_subgroup_add, maybe_vec_storage_type,
        maybe_vec_storage_type_enum,
    },
};
use std::fmt::Write;

#[allow(clippy::too_many_arguments)]
pub(crate) fn general_sgemv(
    op: &QMatMulOperation,
    kernel: &mut GenericKernel,
    workgroup_size: &WorkgroupShape,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    output: &TensorInput,
    _n_size: &str,
    _m_size: &str,
    k_size: &str,
    graph: &crate::compute_graph::ComputeGraphInner,
) {
    let blocksize = workgroup_size.x();
    let dtype = op.input_datatype;
    let global_id = kernel.global_id();
    let workgroup_index = kernel.workgroup_index();
    let workgroup_local_index = kernel.workgroup_local_index();
    let elements_per_block = op.elements_per_block();
    // We don't need to synchronize between the whole workgroup if there is only one subgroup
    let subgroup_size = graph.device.limits().max_subgroup_size;
    let workgroup_sync_data = (blocksize > subgroup_size).then(|| {
        let local_data = kernel.add_global_array(
            KernelGlobalSpace::Workgroup,
            maybe_vec_storage_type_enum(SGEMV_CHUNK_SIZE, dtype),
            blocksize.to_string(),
        );
        let subgroup_id = kernel.subgroup_index();
        let subgroup_local_id = kernel.subgroup_local_index();
        let subgroups_per_workgroup = kernel.subgroups_per_workgroup();
        (
            local_data,
            subgroup_id,
            subgroup_local_id,
            subgroups_per_workgroup,
        )
    });

    // Handle batch dimensions
    writeln!(kernel, "let batch_idx = {global_id}.z;").unwrap();
    // Handle M dimension - each workgroup handles one M value
    writeln!(kernel, "let m_idx = {global_id}.y;").unwrap();

    // In index of the single element in the vector we are multiplying against
    writeln!(
        kernel,
        "let workgroup_offset = {workgroup_index}.x * {SGEMV_CHUNK_SIZE};"
    )
    .unwrap();

    let storage_type = maybe_vec_storage_type(SGEMV_CHUNK_SIZE, dtype);

    writeln!(kernel, "var acc = {storage_type}();").unwrap();

    // Find the reduce size in blocks rounded up
    writeln!(
        kernel,
        "let k_block_size = ({k_size} + {elements_per_block} - 1) / {elements_per_block};"
    )
    .unwrap();

    // Find this threads position in the workgroup
    writeln!(kernel, "let base_axis_index = {workgroup_local_index};").unwrap();
    writeln!(kernel, "let end_axis_index = k_block_size;").unwrap();
    writeln!(kernel, "var index = base_axis_index;").unwrap();

    let chunk_blocks = elements_per_block / SGEMV_VECTOR_SIZE;
    debug_assert!(elements_per_block.is_multiple_of(SGEMV_VECTOR_SIZE));
    writeln!(
        kernel,
        "var a_cache = array<vec{SGEMV_VECTOR_SIZE}<{dtype}>, {chunk_blocks}>();"
    )
    .unwrap();

    // Loop over all of the blocks this thread is responsible for
    writeln!(kernel, "while (index < end_axis_index) {{").unwrap();
    {
        // Load all elements of a into a cache first
        writeln!(kernel, "for (var i = 0u; i < {chunk_blocks}; i += 1u) {{").unwrap();
        {
            // Get the values first
            for i in 0..SGEMV_VECTOR_SIZE {
                writeln!(kernel, "let input_a_{i}_index = index * {elements_per_block} + i * {SGEMV_VECTOR_SIZE} + {i};").unwrap();
                write!(kernel, "let input_a_{i} = {input_a}[").unwrap();
                input_a.strided_index(
                    kernel,
                    vec![
                        "batch_idx".to_string(),
                        "m_idx".to_string(),
                        format!("input_a_{i}_index"),
                    ],
                );
                writeln!(kernel, "];").unwrap();
            }
            // The pack them into a vector and write to the cache
            write!(kernel, "a_cache[i] = vec{SGEMV_VECTOR_SIZE}(").unwrap();
            for i in 0..SGEMV_VECTOR_SIZE {
                if i > 0 {
                    write!(kernel, ", ").unwrap();
                }
                write!(kernel, "input_a_{i}").unwrap();
            }
            writeln!(kernel, ");").unwrap();
        }
        writeln!(kernel, "}}").unwrap();

        if SGEMV_CHUNK_SIZE > 1 {
            writeln!(
                kernel,
                "for (var offset = 0u; offset < {SGEMV_CHUNK_SIZE}; offset += 1u) {{"
            )
            .unwrap();
        }
        let index = if SGEMV_CHUNK_SIZE > 1 {
            "(workgroup_offset + offset)"
        } else {
            "workgroup_offset"
        };
        writeln!(
            kernel,
            "let chunk = {input_b}[{index} * k_block_size + index];"
        )
        .unwrap();

        let acc_indexed = maybe_vec_storage_index(SGEMV_CHUNK_SIZE, "acc", "offset");
        dequantize_vec4_block(
            kernel,
            op.matrix.datatype,
            "chunk".to_string(),
            DataTypeEnum::F32,
            |index, data, code| {
                writeln!(code, "{acc_indexed} += dot(a_cache[{index}], {data});").unwrap();
            },
        );

        if SGEMV_CHUNK_SIZE > 1 {
            writeln!(kernel, "}}").unwrap();
        }

        writeln!(kernel, "index += {blocksize}u;").unwrap();
    }

    writeln!(kernel, "}}").unwrap();

    // Get the sum among all threads in the subgroup
    writeln!(
        kernel,
        "acc = {};",
        maybe_vec_storage_subgroup_add(SGEMV_CHUNK_SIZE, "acc",)
    )
    .unwrap();

    if let Some((local_data, subgroup_id, subgroup_local_id, subgroups_per_workgroup)) =
        workgroup_sync_data
    {
        // Write the output to the workgroup memory if this is the first thread in the subgroup
        writeln!(kernel, "if {subgroup_local_id} == 0u {{").unwrap();
        {
            writeln!(kernel, "{local_data}[{subgroup_id}] = acc;").unwrap();
        }
        writeln!(kernel, "}}").unwrap();

        // Wait until all threads have written to the workgroup shared memory
        writeln!(kernel, "workgroupBarrier();").unwrap();

        // Then if this is the first subgroup, do one final shuffle down reduction
        writeln!(kernel, "if {subgroup_id} != 0u {{").unwrap();
        {
            // Copy over the final value from each subgroup from the workgroup shared memory to the acc variable
            writeln!(
                kernel,
                "if {subgroup_local_id} < {subgroups_per_workgroup} {{"
            )
            .unwrap();
            {
                writeln!(kernel, "acc = {local_data}[{subgroup_local_id}];").unwrap();
            }
            writeln!(kernel, "}}").unwrap();
            writeln!(kernel, "else {{").unwrap();
            {
                writeln!(kernel, "acc = {storage_type}();").unwrap();
            }
            writeln!(kernel, "}}").unwrap();
        }
        writeln!(kernel, "}}").unwrap();

        // Finally get the final sum across all threads in the workgroup
        writeln!(
            kernel,
            "acc = {};",
            maybe_vec_storage_subgroup_add(SGEMV_CHUNK_SIZE, "acc",)
        )
        .unwrap();
    }

    // If this is not the first simd thread in the workgroup, we can return early
    writeln!(kernel, "if {workgroup_local_index} != 0u {{ return; }}").unwrap();

    // Write the output to the output tensor if this is the first thread in the workgroup
    if SGEMV_CHUNK_SIZE > 1 {
        writeln!(
            kernel,
            "for (var offset = 0u; offset < {SGEMV_CHUNK_SIZE}; offset += 1u) {{"
        )
        .unwrap();
    }
    {
        if SGEMV_CHUNK_SIZE > 1 {
            writeln!(kernel, "let output_index = workgroup_offset + offset;").unwrap();
        } else {
            writeln!(kernel, "let output_index = workgroup_offset;").unwrap();
        }
        write!(kernel, "{output}[").unwrap();
        output.strided_index(
            kernel,
            vec![
                "batch_idx".to_string(),
                "m_idx".to_string(),
                "output_index".to_string(),
            ],
        );
        writeln!(
            kernel,
            "] = {};",
            maybe_vec_storage_index(SGEMV_CHUNK_SIZE, "acc", "offset")
        )
        .unwrap();
    }
    if SGEMV_CHUNK_SIZE > 1 {
        writeln!(kernel, "}}").unwrap();
    }
}
