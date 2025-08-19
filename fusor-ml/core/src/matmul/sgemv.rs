use std::fmt::Write;

use crate::{
    mir::{
        globals::KernelGlobalSpace, inputs::TensorInput, kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    }, util::{
        maybe_vec_dot, maybe_vec_storage_index, maybe_vec_storage_subgroup_add, maybe_vec_storage_type, maybe_vec_storage_type_enum
    }, MatMulOperation
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemv(
    op: &MatMulOperation,
    generic_kernel: &mut GenericKernel,
    workgroup_size: &WorkgroupShape,
    input_a: &TensorInput,
    input_b: &TensorInput,
    output: &TensorInput,
    _n_size: &str,
    // m size is always 1 for sgemv
    _m_size: &str,
    k_size: &str,
    params: &SgemvParams,
) {
    let blocksize = workgroup_size.x();
    let dtype = op.datatype;
    let workgroup_index = generic_kernel.workgroup_index();
    let workgroup_local_index = generic_kernel.workgroup_local_index();

    let chunk_size = params.chunk_size;
    let vector_size = params.vector_size;

    // We don't need to synchronize between the whole workgroup if there is only one subgroup
    let workgroup_sync_data = (blocksize > 32).then(|| {
        let local_data = generic_kernel.add_global_array(
            KernelGlobalSpace::Workgroup,
            maybe_vec_storage_type_enum(chunk_size, dtype),
            blocksize.to_string(),
        );
        let subgroup_id = generic_kernel.subgroup_index();
        let subgroup_local_id = generic_kernel.subgroup_local_index();
        let subgroups_per_workgroup = generic_kernel.subgroups_per_workgroup();
        (
            local_data,
            subgroup_id,
            subgroup_local_id,
            subgroups_per_workgroup,
        )
    });

    let mut kernel = String::new();

    // Handle batch dimensions
    writeln!(&mut kernel, "var block_batch = {workgroup_index}.z;").unwrap();

    // Decompose the batch index for higher-dimensional tensors
    for dim in (0..op.rank()).rev().skip(2) {
        let shape = input_a.shape_binding(dim);
        writeln!(
            &mut kernel,
            "let block_batch_{dim} = block_batch % {shape};"
        )
        .unwrap();
        writeln!(&mut kernel, "block_batch /= {shape};").unwrap();
    }

    // Find the batch offset for a, b and output
    for (name, tensor) in [("a", input_a), ("b", input_b), ("c", output)] {
        writeln!(&mut kernel, "let {name}_start_index = ").unwrap();
        let offset = tensor.offset_binding();
        write!(&mut kernel, "{offset}").unwrap();
        for dim in (0..op.rank()).rev().skip(2) {
            let stride = tensor.stride_binding(dim);
            write!(&mut kernel, " + block_batch_{dim}*{stride}").unwrap();
        }
        writeln!(&mut kernel, ";").unwrap();
    }

    // In index of the single element in the vector we are multiplying against
    writeln!(
        &mut kernel,
        "let workgroup_offset = {workgroup_index}.x * {chunk_size};"
    )
    .unwrap();

    let storage_type = maybe_vec_storage_type(chunk_size, dtype);

    writeln!(&mut kernel, "var acc = {storage_type}();").unwrap();

    // Find this threads position in the workgroup
    writeln!(
        &mut kernel,
        "let base_axis_index = {workgroup_local_index};"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "let end_axis_index = ({k_size} + {vector_size} - 1) / {vector_size};"
    )
    .unwrap();
    writeln!(&mut kernel, "var index = base_axis_index;").unwrap();

    let vec_storage = maybe_vec_storage_type(vector_size, dtype);
    writeln!(
        &mut kernel,
        "var a_cache = array<{vec_storage}, 1>();"
    )
    .unwrap();

    // Loop over all of the vector chunks this thread is responsible for
    writeln!(&mut kernel, "while (index < end_axis_index) {{").unwrap();
    {
        if chunk_size > 1 {
            writeln!(
                &mut kernel,
                "for (var offset = 0u; offset < {chunk_size}; offset += 1u) {{"
            )
            .unwrap();
        }
        let row_index = if chunk_size > 1 {
            "(workgroup_offset + offset)"
        } else {
            "workgroup_offset"
        };

        // Load vector elements into cache (from input_b)
        writeln!(
            &mut kernel,
            "var b_cache = array<{vec_storage}, 1>();"
        )
        .unwrap();
        writeln!(&mut kernel, "{{").unwrap();
        {
            for i in 0..vector_size {
                writeln!(
                    &mut kernel,
                    "let input_b_{i}_index = index * {vector_size} + {i};"
                )
                .unwrap();
                let b_row_stride = input_b.stride_binding(op.rank() - 2);
                writeln!(&mut kernel, "let input_b_{i} = select({dtype}(0.0), {input_b}[b_start_index + input_b_{i}_index * {b_row_stride} + 0], input_b_{i}_index < {k_size});").unwrap();
            }
            write!(&mut kernel, "b_cache[0] = {vec_storage}(").unwrap();
            for i in 0..vector_size {
                if i > 0 {
                    write!(&mut kernel, ", ").unwrap();
                }
                write!(&mut kernel, "input_b_{i}").unwrap();
            }
            writeln!(&mut kernel, ");").unwrap();
        }
        writeln!(&mut kernel, "}}").unwrap();

        // Load matrix row elements (from input_a)
        writeln!(&mut kernel, "{{").unwrap();
        {
            // Get the values first
            for i in 0..vector_size {
                writeln!(
                    &mut kernel,
                    "let input_a_{i}_index = index * {vector_size} + {i};"
                )
                .unwrap();
                let a_row_stride = input_a.stride_binding(op.rank() - 2);
                writeln!(&mut kernel, "let input_a_{i} = select({dtype}(0.0), {input_a}[a_start_index + {row_index} * {a_row_stride} + input_a_{i}_index], input_a_{i}_index < {k_size});").unwrap();
            }
            // Then pack them into a vector and write to the cache
            write!(&mut kernel, "a_cache[0] = {vec_storage}(").unwrap();
            for i in 0..vector_size {
                if i > 0 {
                    write!(&mut kernel, ", ").unwrap();
                }
                write!(&mut kernel, "input_a_{i}").unwrap();
            }
            writeln!(&mut kernel, ");").unwrap();
        }
        writeln!(&mut kernel, "}}").unwrap();

        let acc_indexed = maybe_vec_storage_index(chunk_size, "acc", "offset");
        let dot = maybe_vec_dot(vector_size, "a_cache[0]", "b_cache[0]");
        writeln!(&mut kernel, "{acc_indexed} += {dot};").unwrap();

        if chunk_size > 1 {
            writeln!(&mut kernel, "}}").unwrap();
        }

        writeln!(&mut kernel, "index += {blocksize}u;").unwrap();
    }

    writeln!(&mut kernel, "}}").unwrap();

    // Get the sum among all threads in the subgroup
    writeln!(
        &mut kernel,
        "acc = {};",
        maybe_vec_storage_subgroup_add(chunk_size, "acc",)
    )
    .unwrap();

    if let Some((local_data, subgroup_id, subgroup_local_id, subgroups_per_workgroup)) =
        workgroup_sync_data
    {
        // Write the output to the workgroup memory if this is the first thread in the subgroup
        writeln!(&mut kernel, "if {subgroup_local_id} == 0u {{").unwrap();
        {
            writeln!(&mut kernel, "{local_data}[{subgroup_id}] = acc;").unwrap();
        }
        writeln!(&mut kernel, "}}").unwrap();

        // Wait until all threads have written to the workgroup shared memory
        writeln!(&mut kernel, "workgroupBarrier();").unwrap();

        // Then if this is the first subgroup, do one final shuffle down reduction
        writeln!(&mut kernel, "if {subgroup_id} != 0u {{ return; }}").unwrap();
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
            writeln!(&mut kernel, "acc = {storage_type}();").unwrap();
        }
        writeln!(&mut kernel, "}}").unwrap();

        // Finally get the final sum across all threads in the workgroup
        writeln!(
            &mut kernel,
            "acc = {};",
            maybe_vec_storage_subgroup_add(chunk_size, "acc",)
        )
        .unwrap();
    }

    // If this is not the first simd thread in the workgroup, we can return early
    writeln!(
        &mut kernel,
        "if {workgroup_local_index} != 0u {{ return; }}"
    )
    .unwrap();

    // Write the output to the output tensor if this is the first thread in the workgroup
    if chunk_size > 1 {
        writeln!(
            &mut kernel,
            "for (var offset = 0u; offset < {chunk_size}; offset += 1u) {{"
        )
        .unwrap();
    }
    {
        if chunk_size > 1 {
            writeln!(&mut kernel, "let output_index = workgroup_offset + offset;").unwrap();
        } else {
            writeln!(&mut kernel, "let output_index = workgroup_offset;").unwrap();
        }
        let c_row_stride = output.stride_binding(op.rank() - 2);
        write!(
            &mut kernel,
            "{output}[c_start_index + output_index * {c_row_stride} + 0"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "] = {};",
            maybe_vec_storage_index(chunk_size, "acc", "offset")
        )
        .unwrap();
    }
    if chunk_size > 1 {
        writeln!(&mut kernel, "}}").unwrap();
    }

    generic_kernel.push_body(&kernel);
}

pub(crate) fn dispatch_size(n: u32, _m: u32, batch_size: u32, params: &SgemvParams) -> [u32; 3] {
    [n.div_ceil(params.chunk_size), 1, batch_size]
}

pub(crate) fn workgroup_shape_constraints(
    _: &MatMulOperation,
    device: &crate::Device,
    _: &SgemvParams,
) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
    let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();
    let limits = device.wgpu_device().limits();
    constraints.add_constraint(
        0,
        crate::mir::workgroup_shape::Constraint::less_than(limits.max_compute_workgroup_size_x + 1),
    );
    constraints.add_constraint(
        0,
        crate::mir::workgroup_shape::Constraint::equals(limits.min_subgroup_size.max(16)),
    );
    constraints.add_constraint(1, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints.add_constraint(2, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints
}

#[derive(Debug, Clone)]
pub struct SgemvParams {
    chunk_size: u32,
    vector_size: u32,
}

impl SgemvParams {
    pub fn new(chunk_size: u32, vector_size: u32) -> Self {
        Self {
            chunk_size,
            vector_size,
        }
    }

    pub fn chunk_size(&self) -> u32 {
        self.chunk_size
    }
    pub fn vector_size(&self) -> u32 {
        self.vector_size
    }
}

impl Default for SgemvParams {
    fn default() -> Self {
        Self {
            chunk_size: 2,
            vector_size: 4,
        }
    }
}
