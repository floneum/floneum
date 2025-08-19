use std::{fmt::Write, sync::OnceLock};

use crate::MatMulOperation;
use crate::mir::globals::KernelGlobalSpace;
use crate::util::{maybe_vec_storage_index, maybe_vec_storage_type};
use crate::{
    Device,
    mir::{function::Function, kernel::GenericKernel},
    tensor::DataTypeEnum,
};

pub(super) fn workgroup_shape_constraints(
    _: &MatMulOperation,
    _: &Device,
    parameters: &SgemmParams,
) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
    let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();
    constraints.add_constraint(
        0,
        crate::mir::workgroup_shape::Constraint::Equals(
            (parameters.block_m_size * parameters.block_n_size)
                / (parameters.thread_m_size * parameters.thread_n_size),
        ),
    );
    constraints.add_constraint(1, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints.add_constraint(2, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints
}

pub(super) fn dispatch_size(
    last_dim_size: usize,
    second_to_last_dim_size: usize,
    batch_size: usize,
    workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
    parameters: &SgemmParams,
) -> [u32; 3] {
    [
        (last_dim_size as u32).div_ceil(parameters.block_n_size),
        (second_to_last_dim_size as u32).div_ceil(parameters.block_m_size),
        (batch_size as u32).div_ceil(workgroup_shape.z()),
    ]
}

// 1000x1000 dense matmul time on M2 mac pro 1.4743 ms
pub(super) fn build_kernel(
    matmul: &MatMulOperation,
    _: &crate::compute_graph::ComputeGraphInner,
    workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
    inputs: &[crate::mir::inputs::MirValue],
    generic_kernel: &mut GenericKernel,
    parameters: &SgemmParams,
) {
    // Based on CUDA 2D block tiling SGEMM
    let [input_a, input_b, _] = inputs else {
        panic!("MatMulOperation requires 3 inputs");
    };
    let input_a = input_a.as_tensor().unwrap();
    let input_a_datatype = input_a.datatype();
    let input_b = input_b.as_tensor().unwrap();
    let input_b_datatype = input_b.datatype();

    // Calculate parameters we use throughout the kernel
    let block_m_size = parameters.block_m_size;
    let block_n_size = parameters.block_n_size;
    let block_k_size = parameters.block_k_size;
    let thread_m_size = parameters.thread_m_size;
    let thread_n_size = parameters.thread_n_size;
    let double_buffer = parameters.double_buffer;
    let threads_per_workgroup: u32 =
        (block_m_size * block_n_size) / (thread_m_size * thread_n_size);
    let threads_per_k_a: u32 = threads_per_workgroup / block_k_size;
    let unroll_count_a: u32 = block_m_size / threads_per_k_a;
    let threads_per_n_b: u32 = threads_per_workgroup / block_n_size;
    let unroll_count_b: u32 = block_k_size / threads_per_n_b;

    let mut kernel = String::new();

    let pre_element_wise_functions: OnceLock<[Vec<Function>; 2]> = OnceLock::new();
    let post_element_wise_functions = OnceLock::new();

    let input_a = generic_kernel.add_tensor_input(matmul.rank(), false, input_a_datatype);
    let input_b = generic_kernel.add_tensor_input(matmul.rank(), false, input_b_datatype);
    let output = generic_kernel.add_tensor_input(
        matmul.rank(),
        true,
        matmul.post_element_wise.out_datatype(),
    );

    const PADDING: u32 = 1; // Add padding for bank conflict avoidance
    let cache_a_size = if double_buffer { 2 } else { 1 } * block_m_size * (block_k_size + PADDING);
    let cache_a = generic_kernel.add_global_array(
        KernelGlobalSpace::Workgroup,
        matmul.datatype,
        cache_a_size.to_string(),
    );
    let cache_b_size = if double_buffer { 2 } else { 1 } * block_n_size * (block_k_size + PADDING);
    let cache_b = generic_kernel.add_global_array(
        KernelGlobalSpace::Workgroup,
        matmul.datatype,
        cache_b_size.to_string(),
    );

    assert_eq!(
        workgroup_shape.x(),
        threads_per_workgroup,
        "parameters {parameters:?} invalid"
    );

    let datatype = matmul.datatype;
    let workgroup_index = generic_kernel.workgroup_index();
    let workgroup_local_index = generic_kernel.workgroup_local_index();

    // Get dimension bindings
    let k_size = input_a.shape_binding(matmul.rank() - 1);
    let m_size = input_a.shape_binding(matmul.rank() - 2);
    let n_size = input_b.shape_binding(matmul.rank() - 1);

    // Map CUDA block indices to WGSL workgroup indices
    writeln!(&mut kernel, "let cRow = {workgroup_index}.y;").unwrap();
    writeln!(&mut kernel, "let cCol = {workgroup_index}.x;").unwrap();
    writeln!(&mut kernel, "var block_batch = {workgroup_index}.z;").unwrap();

    for dim in (0..matmul.rank()).rev().skip(2) {
        let shape = input_a.shape_binding(dim);
        writeln!(
            &mut kernel,
            "let block_batch_{dim} = block_batch % {shape};"
        )
        .unwrap();
        writeln!(&mut kernel, "block_batch /= {shape};").unwrap();
    }

    writeln!(
        &mut kernel,
        "let totalResultsBlocktile = {block_m_size}u * {block_n_size}u;"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "let numThreadsBlocktile = totalResultsBlocktile / ({thread_m_size}u * {thread_n_size}u);"
    )
    .unwrap();

    // Thread indices within the workgroup
    writeln!(
        &mut kernel,
        "let threadCol = {workgroup_local_index} % ({block_n_size}u / {thread_n_size}u);"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "let threadRow = {workgroup_local_index} / ({block_n_size}u / {thread_n_size}u);"
    )
    .unwrap();

    // Find the batch offset for a, b and output
    for (name, tensor) in [("a", &input_a), ("b", &input_b), ("c", &output)] {
        writeln!(&mut kernel, "let {name}_start_index = ").unwrap();
        let offset = tensor.offset_binding();
        write!(&mut kernel, "{offset}").unwrap();
        for dim in (0..matmul.rank()).rev().skip(2) {
            let stride = tensor.stride_binding(dim);
            write!(&mut kernel, " + block_batch_{dim}*{stride}").unwrap();
        }
        writeln!(&mut kernel, ";").unwrap();
    }

    // Thread indices for loading into shared memory
    writeln!(
        &mut kernel,
        "let innerRowA = {workgroup_local_index} / {block_k_size}u;"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "let innerColA = {workgroup_local_index} % {block_k_size}u;"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "let innerRowB = {workgroup_local_index} / {block_n_size}u;"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "let innerColB = {workgroup_local_index} % {block_n_size}u;"
    )
    .unwrap();

    // Allocate thread-local cache for results
    writeln!(
        &mut kernel,
        "var threadResults: array<{datatype}, {}>;",
        thread_m_size * thread_n_size
    )
    .unwrap();

    let zero_literal = match datatype {
        DataTypeEnum::F16 => "f16(0.0)",
        _ => "0.0",
    };

    // Register caches
    let thread_m_dtype = maybe_vec_storage_type(thread_m_size, datatype);
    writeln!(&mut kernel, "var regM: {thread_m_dtype};").unwrap();
    let thread_n_dtype = maybe_vec_storage_type(thread_n_size, datatype);
    writeln!(&mut kernel, "var regN: {thread_n_dtype};").unwrap();

    writeln!(
        &mut kernel,
        "let tiles = ({k_size} + {block_k_size}u - 1u) / {block_k_size}u;"
    )
    .unwrap();
    if double_buffer {
        writeln!(&mut kernel, "var buf: u32 = 0u;").unwrap();
        // Helper bases for double-buffered shared arrays
        // Use swizzled tile sizes to avoid bank conflicts
        writeln!(
            &mut kernel,
            "let aTileSize = {block_m_size}u * {}u;",
            block_k_size + PADDING
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "let bTileSize = {block_n_size}u * {}u;",
            block_k_size + PADDING
        )
        .unwrap();
    }

    // Preload tile 0 into buffer 0
    writeln!(&mut kernel, "if (tiles > 0u) {{").unwrap();
    writeln!(&mut kernel, "    let bkIdx = 0u;").unwrap();
    if double_buffer {
        writeln!(&mut kernel, "    let aBase = buf * aTileSize;").unwrap();
        writeln!(&mut kernel, "    let bBase = buf * bTileSize;").unwrap();
    }
    // Fast-path predicates for tile 0
    writeln!(
        &mut kernel,
        "    let fullK = (bkIdx + {block_k_size}u) <= {k_size};"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "    let fullA = (((cRow + 1u) * {block_m_size}u) <= {m_size}) && fullK;"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "    let fullB = (((cCol + 1u) * {block_n_size}u) <= {n_size}) && fullK;"
    )
    .unwrap();
    // A tile load (unrolled for optimal performance)
    let pef = pre_element_wise_functions.get_or_init(|| {
        std::array::from_fn(|i| matmul.pre_element_wise[i].add_functions(generic_kernel))
    });
    let a_offset = if double_buffer { "aBase + " } else { "" };
    let b_offset = if double_buffer { "bBase + " } else { "" };
    let write_row_column = |kernel: &mut String, i, tensor: &str| {
        let tensor = tensor.to_uppercase();
        writeln!(kernel, "            let row_raw = innerRow{tensor} + {i}u * (numThreadsBlocktile / {block_k_size}u);").unwrap();
        writeln!(kernel, "            let col_raw = innerCol{tensor};").unwrap();
    };
    let write_back_a = |kernel: &mut String| {
        writeln!(kernel, "            {cache_a}[{a_offset}row_raw * ({block_k_size}u + {PADDING}) + col_raw] = a_val;").unwrap();
    };
    let write_read_a = |kernel: &mut String| {
        writeln!(kernel, "            var a_val = ").unwrap();
        let first_value = pef[0].iter().fold(
            format!("{input_a}[a_start_index + a_row * {k_size} + a_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(kernel, "{first_value};").unwrap();
    };
    writeln!(&mut kernel, "    if (fullA) {{").unwrap();
    for i in 0..unroll_count_a {
        writeln!(&mut kernel, "        {{").unwrap();
        write_row_column(&mut kernel, i, "A");
        writeln!(
            &mut kernel,
            "            let a_row = cRow * {block_m_size}u + row_raw;"
        )
        .unwrap();
        writeln!(&mut kernel, "            let a_col = bkIdx + col_raw;").unwrap();
        write_read_a(&mut kernel);
        write_back_a(&mut kernel);
        writeln!(&mut kernel, "        }}").unwrap();
    }
    writeln!(&mut kernel, "    }} else {{").unwrap();
    for i in 0..unroll_count_a {
        writeln!(&mut kernel, "        {{").unwrap();
        write_row_column(&mut kernel, i, "A");
        writeln!(
            &mut kernel,
            "            let a_row_global = cRow * {block_m_size}u + row_raw;"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "            let a_col_global = bkIdx + col_raw;"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "            let a_row = min(a_row_global, {m_size} - 1u);"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "            let a_col = min(a_col_global, {k_size} - 1u);"
        )
        .unwrap();
        write_read_a(&mut kernel);
        writeln!(&mut kernel, "            a_val = select({zero_literal}, a_val, (a_row_global < {m_size} && a_col_global < {k_size}));").unwrap();
        write_back_a(&mut kernel);
        writeln!(&mut kernel, "        }}").unwrap();
    }
    writeln!(&mut kernel, "    }}").unwrap();
    // B tile load (unrolled for optimal performance)
    let write_back_b = |kernel: &mut String| {
        writeln!(kernel, "            {cache_b}[{b_offset}col_raw * ({block_k_size}u + {PADDING}) + row_raw] = b_val;").unwrap();
    };
    let write_read_b = |kernel: &mut String| {
        writeln!(kernel, "            var b_val = ").unwrap();
        let second_value_fast = pef[1].iter().fold(
            format!("{input_b}[b_start_index + b_row * {n_size} + b_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(kernel, "{second_value_fast};").unwrap();
    };
    writeln!(&mut kernel, "    if (fullB) {{").unwrap();
    for i in 0..unroll_count_b {
        writeln!(&mut kernel, "        {{").unwrap();
        writeln!(
            &mut kernel,
            "            let row_raw = innerRowB + {i}u * (numThreadsBlocktile / {block_n_size}u);"
        )
        .unwrap();
        writeln!(&mut kernel, "            let col_raw = innerColB;").unwrap();
        writeln!(&mut kernel, "            let b_row = bkIdx + row_raw;").unwrap();
        writeln!(
            &mut kernel,
            "            let b_col = cCol * {block_n_size}u + col_raw;"
        )
        .unwrap();
        write_read_b(&mut kernel);
        write_back_b(&mut kernel);
        writeln!(&mut kernel, "        }}").unwrap();
    }
    writeln!(&mut kernel, "    }} else {{").unwrap();
    for i in 0..unroll_count_b {
        writeln!(&mut kernel, "        {{").unwrap();
        writeln!(
            &mut kernel,
            "            let row_raw = innerRowB + {i}u * (numThreadsBlocktile / {block_n_size}u);"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "            let col_raw = innerColB; let b_row_global = bkIdx + row_raw;"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "            let b_col_global = cCol * {block_n_size}u + col_raw;"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "            let b_row = min(b_row_global, {k_size} - 1u);"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "            let b_col = min(b_col_global, {n_size} - 1u);"
        )
        .unwrap();
        write_read_b(&mut kernel);
        writeln!(&mut kernel, "            b_val = select({zero_literal}, b_val, (b_row_global < {k_size} && b_col_global < {n_size}));").unwrap();
        write_back_b(&mut kernel);
        writeln!(&mut kernel, "        }}").unwrap();
    }
    writeln!(&mut kernel, "    }}").unwrap();
    writeln!(&mut kernel, "}}").unwrap();

    // Synchronize after initial preload
    writeln!(&mut kernel, "workgroupBarrier();").unwrap();

    // Tiled compute with prefetch of next tile in
    writeln!(&mut kernel, "for (var t = 0u; t < tiles; t++) {{").unwrap();
    if double_buffer {
        // Bases for current buffers
        writeln!(&mut kernel, "    let aBase = buf * aTileSize;").unwrap();
        writeln!(&mut kernel, "    let bBase = buf * bTileSize;").unwrap();
    }

    // Calculate per-thread results for current tile with overlapped prefetch
    writeln!(
        &mut kernel,
        "    for (var dotIdx = 0u; dotIdx < {block_k_size}u; dotIdx++) {{"
    )
    .unwrap();
    writeln!(&mut kernel, "        let reg_m_offset = {a_offset}threadRow * {thread_m_size}u * ({block_k_size}u + {PADDING}) + dotIdx;").unwrap();

    // Vectorized loads with padding for bank conflict avoidance
    let stride_a = block_k_size + PADDING;
    write!(&mut kernel, "            regM = {thread_m_dtype}(").unwrap();
    for i in 0..thread_m_size {
        if i > 0 {
            write!(&mut kernel, ", ").unwrap();
        }
        write!(&mut kernel, "{cache_a}[reg_m_offset + {}]", i * stride_a).unwrap();
    }
    writeln!(&mut kernel, ");").unwrap();

    writeln!(
        &mut kernel,
        "        let reg_n_offset = {b_offset}threadCol * {thread_n_size}u * ({block_k_size}u + {PADDING}) + dotIdx;"
    )
    .unwrap();

    // Vectorized load for N register with padding
    let stride_b = block_k_size + PADDING;
    write!(&mut kernel, "            regN = {thread_n_dtype}(").unwrap();
    for i in 0..thread_n_size {
        if i > 0 {
            write!(&mut kernel, ", ").unwrap();
        }
        write!(&mut kernel, "{cache_b}[reg_n_offset + {}]", i * stride_b).unwrap();
    }
    writeln!(&mut kernel, ");").unwrap();

    for res_idx_m in 0..thread_m_size {
        let indexed_reg_m = maybe_vec_storage_index(thread_m_size, "regM", res_idx_m);
        writeln!(
            &mut kernel,
            "        let result_{res_idx_m} = {indexed_reg_m} * regN;"
        )
        .unwrap();
        for res_idx_n in 0..thread_n_size {
            let indexed_result = maybe_vec_storage_index(thread_m_size, format_args!("result_{res_idx_m}"), res_idx_n);
            writeln!(
                &mut kernel,
                "        threadResults[{} * {thread_n_size}u + {}] += {indexed_result};",
                res_idx_m, res_idx_n
            )
            .unwrap();
        }
    }
    writeln!(&mut kernel, "    }}").unwrap();

    if !double_buffer {
        // Synchronize all threads before prefetching next tile into the same shared buffers
        writeln!(&mut kernel, "    workgroupBarrier();").unwrap();
    }

    // Prefetch tile into the alternate buffer
    writeln!(&mut kernel, "    if ((t + 1u) < tiles) {{").unwrap();
    writeln!(
        &mut kernel,
        "        let bkIdx = (t + 1u) * {block_k_size}u;"
    )
    .unwrap();
    if double_buffer {
        writeln!(&mut kernel, "        let nextBuf: u32 = 1u - buf;").unwrap();
        writeln!(&mut kernel, "        let aBaseN = nextBuf * aTileSize;").unwrap();
        writeln!(&mut kernel, "        let bBaseN = nextBuf * bTileSize;").unwrap();
    }
    let add_a_base_n = if double_buffer { "aBaseN + " } else { "" };
    let add_b_base_n = if double_buffer { "bBaseN + " } else { "" };
    writeln!(
        &mut kernel,
        "        let fullK = (bkIdx + {block_k_size}u) <= {k_size};"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "        let fullA = (((cRow + 1u) * {block_m_size}u) <= {m_size}) && fullK;"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "        let fullB = (((cCol + 1u) * {block_n_size}u) <= {n_size}) && fullK;"
    )
    .unwrap();

    // A tile prefetch (unrolled for optimal performance)
    writeln!(&mut kernel, "        if (fullA) {{").unwrap();
    let pef = pre_element_wise_functions.get_or_init(|| {
        std::array::from_fn(|i| matmul.pre_element_wise[i].add_functions(generic_kernel))
    });
    for i in 0..unroll_count_a {
        writeln!(&mut kernel, "            {{").unwrap();
        write_row_column(&mut kernel, i, "A");
        writeln!(
            &mut kernel,
            "                let a_row = cRow * {block_m_size}u + row_raw;"
        )
        .unwrap();
        writeln!(&mut kernel, "                let a_col = bkIdx + col_raw;").unwrap();
        writeln!(&mut kernel, "                var a_val = ").unwrap();
        let first_value_fast = pef[0].iter().fold(
            format!("{input_a}[a_start_index + a_row * {k_size} + a_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{first_value_fast};").unwrap();
        writeln!(&mut kernel, "                {cache_a}[{add_a_base_n}row_raw * ({block_k_size}u + {PADDING}) + col_raw] = a_val;").unwrap();
        writeln!(&mut kernel, "            }}").unwrap();
    }
    writeln!(&mut kernel, "        }} else {{").unwrap();
    for i in 0..unroll_count_a {
        writeln!(&mut kernel, "            {{").unwrap();
        write_row_column(&mut kernel, i, "A");
        writeln!(
            &mut kernel,
            "                let a_row_global = cRow * {block_m_size}u + row_raw;"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "                let a_col_global = bkIdx + col_raw;"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "                let a_row = min(a_row_global, {m_size} - 1u);"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "                let a_col = min(a_col_global, {k_size} - 1u);"
        )
        .unwrap();
        writeln!(&mut kernel, "                var a_val = ").unwrap();
        let first_value = pef[0].iter().fold(
            format!("{input_a}[a_start_index + a_row * {k_size} + a_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{first_value};").unwrap();
        writeln!(&mut kernel, "                a_val = select({zero_literal}, a_val, (a_row_global < {m_size} && a_col_global < {k_size}));").unwrap();
        writeln!(&mut kernel, "                {cache_a}[{add_a_base_n}row_raw * ({block_k_size}u + {PADDING}) + col_raw] = a_val;").unwrap();
        writeln!(&mut kernel, "            }}").unwrap();
    }
    writeln!(&mut kernel, "        }}").unwrap();

    // B tile prefetch (unrolled for optimal performance) to transposed layout
    writeln!(&mut kernel, "        if (fullB) {{").unwrap();
    for i in 0..unroll_count_b {
        writeln!(&mut kernel, "            {{").unwrap();
        writeln!(&mut kernel, "                let row_raw = innerRowB + {i}u * (numThreadsBlocktile / {block_n_size}u);").unwrap();
        writeln!(&mut kernel, "                let col_raw = innerColB;").unwrap();
        writeln!(&mut kernel, "                let b_row = bkIdx + row_raw;").unwrap();
        writeln!(
            &mut kernel,
            "                let b_col = cCol * {block_n_size}u + col_raw;"
        )
        .unwrap();
        writeln!(&mut kernel, "                var b_val = ").unwrap();
        let second_value_fast = pef[1].iter().fold(
            format!("{input_b}[b_start_index + b_row * {n_size} + b_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{second_value_fast};").unwrap();
        writeln!(&mut kernel, "                {cache_b}[{add_b_base_n}col_raw * ({block_k_size}u + {PADDING}) + row_raw] = b_val;").unwrap();
        writeln!(&mut kernel, "            }}").unwrap();
    }
    writeln!(&mut kernel, "        }} else {{").unwrap();
    for i in 0..unroll_count_b {
        writeln!(&mut kernel, "            {{").unwrap();
        writeln!(&mut kernel, "                let row_raw = innerRowB + {i}u * (numThreadsBlocktile / {block_n_size}u);").unwrap();
        writeln!(&mut kernel, "                let col_raw = innerColB;").unwrap();
        writeln!(
            &mut kernel,
            "                let b_row_global = bkIdx + row_raw;"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "                let b_col_global = cCol * {block_n_size}u + col_raw;"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "                let b_row = min(b_row_global, {k_size} - 1u);"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "                let b_col = min(b_col_global, {n_size} - 1u);"
        )
        .unwrap();
        writeln!(&mut kernel, "                var b_val = ").unwrap();
        let second_value = pef[1].iter().fold(
            format!("{input_b}[b_start_index + b_row * {n_size} + b_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{second_value};").unwrap();
        writeln!(&mut kernel, "                b_val = select({zero_literal}, b_val, (b_row_global < {k_size} && b_col_global < {n_size}));").unwrap();
        writeln!(&mut kernel, "                {cache_b}[{add_b_base_n}col_raw * ({block_k_size}u + {PADDING}) + row_raw] = b_val;").unwrap();
        writeln!(&mut kernel, "            }}").unwrap();
    }
    writeln!(&mut kernel, "        }}").unwrap();
    writeln!(&mut kernel, "    }}").unwrap();

    // Synchronize before using the newly prefetched buffer in the next iteration
    writeln!(&mut kernel, "    workgroupBarrier();").unwrap();
    if double_buffer {
        // Toggle the buffer index for the next iteration
        writeln!(&mut kernel, "    buf = 1u - buf;").unwrap();
    }
    writeln!(&mut kernel, "}}").unwrap();

    // Write out the results (same as previous implementation)
    writeln!(
        &mut kernel,
        "let outRowOffset = threadRow * {thread_m_size}u + cRow * {block_m_size}u;"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "let outColOffset = threadCol * {thread_n_size}u + cCol * {block_n_size}u;"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "if (outRowOffset < {m_size} && outColOffset < {n_size}) {{"
    )
    .unwrap();
    for res_idx_m in 0..thread_m_size {
        writeln!(
            &mut kernel,
            "let outRow{res_idx_m} = min(outRowOffset + {res_idx_m}, {m_size} - 1);"
        )
        .unwrap();
    }
    for res_idx_n in 0..thread_n_size {
        writeln!(
            &mut kernel,
            "let outCol{res_idx_n} = min(outColOffset + {res_idx_n}, {n_size} - 1);"
        )
        .unwrap();
    }
    for res_idx_m in 0..thread_m_size {
        for res_idx_n in 0..thread_n_size {
            let post_element_wise_functions = post_element_wise_functions
                .get_or_init(|| matmul.post_element_wise.add_functions(generic_kernel));
            write!(
                &mut kernel,
                "{output}[c_start_index + outRow{res_idx_m} * {n_size} + outCol{res_idx_n}] = "
            )
            .unwrap();
            let result = post_element_wise_functions.iter().fold(
                    format!("threadResults[(outRow{res_idx_m} - outRowOffset) * {thread_n_size}u + (outCol{res_idx_n} - outColOffset)]"),
                    |acc, f| f.call(vec![acc]),
                );
            writeln!(&mut kernel, "{result};").unwrap();
        }
    }
    writeln!(&mut kernel, "}}").unwrap();

    generic_kernel.push_body(&kernel);
}

#[derive(Debug, Clone)]
pub struct SgemmParams {
    double_buffer: bool,
    block_m_size: u32,
    block_n_size: u32,
    block_k_size: u32,
    thread_m_size: u32,
    thread_n_size: u32,
}

impl SgemmParams {
    pub fn new(
        double_buffer: bool,
        block_m_size: u32,
        block_n_size: u32,
        block_k_size: u32,
        thread_m_size: u32,
        thread_n_size: u32,
    ) -> Self {
        Self {
            double_buffer,
            block_m_size,
            block_n_size,
            block_k_size,
            thread_m_size,
            thread_n_size,
        }
    }

    pub fn double_buffer(&self) -> bool {
        self.double_buffer
    }
    pub fn block_m_size(&self) -> u32 {
        self.block_m_size
    }
    pub fn block_n_size(&self) -> u32 {
        self.block_n_size
    }
    pub fn block_k_size(&self) -> u32 {
        self.block_k_size
    }
    pub fn thread_m_size(&self) -> u32 {
        self.thread_m_size
    }
    pub fn thread_n_size(&self) -> u32 {
        self.thread_n_size
    }
}

impl Default for SgemmParams {
    fn default() -> Self {
        let thread_m_size: u32 = 4;
        let thread_n_size: u32 = 4;
        let block_m_size: u32 = thread_m_size * 16;
        let block_n_size: u32 = thread_n_size * 8;
        let block_k_size: u32 = 8;
        let double_buffer: bool = false;

        Self {
            double_buffer,
            block_m_size,
            block_n_size,
            block_k_size,
            thread_m_size,
            thread_n_size,
        }
    }
}
