use std::{fmt::Write, sync::OnceLock};

use crate::MatMulOperation;
use crate::mir::globals::KernelGlobalSpace;
use crate::{
    Device,
    mir::{function::Function, kernel::GenericKernel},
    tensor::{DataTypeEnum},
};

pub(super) fn workgroup_shape_constraints(
    _: &MatMulOperation,
    _: &Device,
) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
    let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();
    constraints.add_constraint(
        0,
        crate::mir::workgroup_shape::Constraint::Equals(
            (WORK_GROUP_BLOCK_M_SIZE / THREAD_BLOCK_M_SIZE)
                * (WORK_GROUP_BLOCK_N_SIZE / THREAD_BLOCK_N_SIZE),
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
) -> [u32; 3] {
    [
        (last_dim_size as u32).div_ceil(WORK_GROUP_BLOCK_N_SIZE),
        (second_to_last_dim_size as u32).div_ceil(WORK_GROUP_BLOCK_M_SIZE),
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
) {
    // Based on CUDA 2D block tiling SGEMM
    let [input_a, input_b, _] = inputs else {
        panic!("MatMulOperation requires 3 inputs");
    };
    let input_a = input_a.as_tensor().unwrap();
    let input_a_datatype = input_a.datatype();
    let input_b = input_b.as_tensor().unwrap();
    let input_b_datatype = input_b.datatype();

    // Calculate unroll counts for optimal performance
    const THREADS_PER_WORKGROUP: u32 = (WORK_GROUP_BLOCK_M_SIZE * WORK_GROUP_BLOCK_N_SIZE)
        / (THREAD_BLOCK_M_SIZE * THREAD_BLOCK_N_SIZE);
    const THREADS_PER_K_A: u32 = THREADS_PER_WORKGROUP / WORK_GROUP_BLOCK_K_SIZE;
    const UNROLL_COUNT_A: u32 = WORK_GROUP_BLOCK_M_SIZE / THREADS_PER_K_A;
    const THREADS_PER_N_B: u32 = THREADS_PER_WORKGROUP / WORK_GROUP_BLOCK_N_SIZE;
    const UNROLL_COUNT_B: u32 = WORK_GROUP_BLOCK_K_SIZE / THREADS_PER_N_B;

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
    let cache_a_size = if DOUBLE_BUFFER {
        2 * WORK_GROUP_BLOCK_M_SIZE * (WORK_GROUP_BLOCK_K_SIZE + PADDING)
    } else {
        WORK_GROUP_BLOCK_M_SIZE * (WORK_GROUP_BLOCK_K_SIZE + PADDING)
    };
    let cache_a = generic_kernel.add_global_array(
        KernelGlobalSpace::Workgroup,
        matmul.datatype,
        cache_a_size.to_string(),
    );
    let cache_b_size = if DOUBLE_BUFFER {
        2 * WORK_GROUP_BLOCK_N_SIZE * (WORK_GROUP_BLOCK_K_SIZE + PADDING)
    } else {
        WORK_GROUP_BLOCK_N_SIZE * (WORK_GROUP_BLOCK_K_SIZE + PADDING)
    };
    let cache_b = generic_kernel.add_global_array(
        KernelGlobalSpace::Workgroup,
        matmul.datatype,
        cache_b_size.to_string(),
    );

    assert_eq!(workgroup_shape.x(), THREADS_PER_WORKGROUP);

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
        "let totalResultsBlocktile = {WORK_GROUP_BLOCK_M_SIZE}u * {WORK_GROUP_BLOCK_N_SIZE}u;"
    )
    .unwrap();
    writeln!(&mut kernel, "let numThreadsBlocktile = totalResultsBlocktile / ({THREAD_BLOCK_M_SIZE}u * {THREAD_BLOCK_N_SIZE}u);").unwrap();

    // Thread indices within the workgroup
    writeln!(&mut kernel, "let threadCol = {workgroup_local_index} % ({WORK_GROUP_BLOCK_N_SIZE}u / {THREAD_BLOCK_N_SIZE}u);").unwrap();
    writeln!(&mut kernel, "let threadRow = {workgroup_local_index} / ({WORK_GROUP_BLOCK_N_SIZE}u / {THREAD_BLOCK_N_SIZE}u);").unwrap();

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
        "let innerRowA = {workgroup_local_index} / {WORK_GROUP_BLOCK_K_SIZE}u;"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "let innerColA = {workgroup_local_index} % {WORK_GROUP_BLOCK_K_SIZE}u;"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "let innerRowB = {workgroup_local_index} / {WORK_GROUP_BLOCK_N_SIZE}u;"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "let innerColB = {workgroup_local_index} % {WORK_GROUP_BLOCK_N_SIZE}u;"
    )
    .unwrap();

    // Allocate thread-local cache for results
    writeln!(
        &mut kernel,
        "var threadResults: array<{datatype}, {}>;",
        THREAD_BLOCK_M_SIZE * THREAD_BLOCK_N_SIZE
    )
    .unwrap();

    let zero_literal = match datatype {
        DataTypeEnum::F16 => "f16(0.0)",
        _ => "0.0",
    };

    // Register caches
    writeln!(
        &mut kernel,
        "var regM: vec{THREAD_BLOCK_M_SIZE}<{datatype}>;"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "var regN: vec{THREAD_BLOCK_N_SIZE}<{datatype}>;"
    )
    .unwrap();

    writeln!(
        &mut kernel,
        "let tiles = ({k_size} + {WORK_GROUP_BLOCK_K_SIZE}u - 1u) / {WORK_GROUP_BLOCK_K_SIZE}u;"
    )
    .unwrap();
    if DOUBLE_BUFFER {
        writeln!(&mut kernel, "var buf: u32 = 0u;").unwrap();
        // Helper bases for double-buffered shared arrays
        // Use swizzled tile sizes to avoid bank conflicts
        writeln!(
            &mut kernel,
            "let aTileSize = {WORK_GROUP_BLOCK_M_SIZE}u * {}u;",
            WORK_GROUP_BLOCK_K_SIZE + PADDING
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "let bTileSize = {WORK_GROUP_BLOCK_N_SIZE}u * {}u;",
            WORK_GROUP_BLOCK_K_SIZE + PADDING
        )
        .unwrap();
    }

    // Preload tile 0 into buffer 0
    writeln!(&mut kernel, "if (tiles > 0u) {{").unwrap();
    writeln!(&mut kernel, "    let bkIdx = 0u;").unwrap();
    if DOUBLE_BUFFER {
        writeln!(&mut kernel, "    let aBase = buf * aTileSize;").unwrap();
        writeln!(&mut kernel, "    let bBase = buf * bTileSize;").unwrap();
    }
    // Fast-path predicates for tile 0
    writeln!(
        &mut kernel,
        "    let fullK = (bkIdx + {WORK_GROUP_BLOCK_K_SIZE}u) <= {k_size};"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "    let fullA = (((cRow + 1u) * {WORK_GROUP_BLOCK_M_SIZE}u) <= {m_size}) && fullK;"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "    let fullB = (((cCol + 1u) * {WORK_GROUP_BLOCK_N_SIZE}u) <= {n_size}) && fullK;"
    )
    .unwrap();
    // A tile load (unrolled for optimal performance)
    let pef = pre_element_wise_functions.get_or_init(|| {
        std::array::from_fn(|i| matmul.pre_element_wise[i].add_functions(generic_kernel))
    });
    let a_offset = if DOUBLE_BUFFER { "aBase + " } else { "" };
    let b_offset = if DOUBLE_BUFFER { "bBase + " } else { "" };
    let write_row_column = |kernel: &mut String, i, tensor: &str| {
        let tensor = tensor.to_uppercase();
        writeln!(kernel, "            let row_raw = innerRow{tensor} + {i}u * (numThreadsBlocktile / {WORK_GROUP_BLOCK_K_SIZE}u);").unwrap();
        writeln!(kernel, "            let col_raw = innerCol{tensor};").unwrap();
    };
    let write_back_a = |kernel: &mut String| {
        writeln!(kernel, "            {cache_a}[{a_offset}row_raw * ({WORK_GROUP_BLOCK_K_SIZE}u + {PADDING}) + col_raw] = a_val;").unwrap();
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
    for i in 0..UNROLL_COUNT_A {
        writeln!(&mut kernel, "        {{").unwrap();
        write_row_column(&mut kernel, i, "A");
        writeln!(
            &mut kernel,
            "            let a_row = cRow * {WORK_GROUP_BLOCK_M_SIZE}u + row_raw;"
        )
        .unwrap();
        writeln!(&mut kernel, "            let a_col = bkIdx + col_raw;").unwrap();
        write_read_a(&mut kernel);
        write_back_a(&mut kernel);
        writeln!(&mut kernel, "        }}").unwrap();
    }
    writeln!(&mut kernel, "    }} else {{").unwrap();
    for i in 0..UNROLL_COUNT_A {
        writeln!(&mut kernel, "        {{").unwrap();
        write_row_column(&mut kernel, i, "A");
        writeln!(
            &mut kernel,
            "            let a_row_global = cRow * {WORK_GROUP_BLOCK_M_SIZE}u + row_raw;"
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
        writeln!(kernel, "            {cache_b}[{b_offset}col_raw * ({WORK_GROUP_BLOCK_K_SIZE}u + {PADDING}) + row_raw] = b_val;").unwrap();
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
    for i in 0..UNROLL_COUNT_B {
        writeln!(&mut kernel, "        {{").unwrap();
        writeln!(&mut kernel, "            let row_raw = innerRowB + {i}u * (numThreadsBlocktile / {WORK_GROUP_BLOCK_N_SIZE}u);").unwrap();
        writeln!(&mut kernel, "            let col_raw = innerColB;").unwrap();
        writeln!(&mut kernel, "            let b_row = bkIdx + row_raw;").unwrap();
        writeln!(
            &mut kernel,
            "            let b_col = cCol * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw;"
        )
        .unwrap();
        write_read_b(&mut kernel);
        write_back_b(&mut kernel);
        writeln!(&mut kernel, "        }}").unwrap();
    }
    writeln!(&mut kernel, "    }} else {{").unwrap();
    for i in 0..UNROLL_COUNT_B {
        writeln!(&mut kernel, "        {{").unwrap();
        writeln!(&mut kernel, "            let row_raw = innerRowB + {i}u * (numThreadsBlocktile / {WORK_GROUP_BLOCK_N_SIZE}u);").unwrap();
        writeln!(
            &mut kernel,
            "            let col_raw = innerColB; let b_row_global = bkIdx + row_raw;"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "            let b_col_global = cCol * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw;"
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
    if DOUBLE_BUFFER {
        // Bases for current buffers
        writeln!(&mut kernel, "    let aBase = buf * aTileSize;").unwrap();
        writeln!(&mut kernel, "    let bBase = buf * bTileSize;").unwrap();
    }

    // Calculate per-thread results for current tile with overlapped prefetch
    writeln!(
        &mut kernel,
        "    for (var dotIdx = 0u; dotIdx < {WORK_GROUP_BLOCK_K_SIZE}u; dotIdx++) {{"
    )
    .unwrap();
    writeln!(&mut kernel, "        let reg_m_offset = {a_offset}threadRow * {THREAD_BLOCK_M_SIZE}u * ({WORK_GROUP_BLOCK_K_SIZE}u + {PADDING}) + dotIdx;").unwrap();

    // Vectorized loads with padding for bank conflict avoidance
    let stride_a = WORK_GROUP_BLOCK_K_SIZE + PADDING;
    write!(&mut kernel, "            regM = vec{THREAD_BLOCK_M_SIZE}(").unwrap();
    for i in 0..THREAD_BLOCK_M_SIZE {
        if i > 0 {
            write!(&mut kernel, ", ").unwrap();
        }
        write!(&mut kernel, "{cache_a}[reg_m_offset + {}]", i * stride_a).unwrap();
    }
    writeln!(&mut kernel, ");").unwrap();

    writeln!(
            &mut kernel,
            "        let reg_n_offset = {b_offset}threadCol * {THREAD_BLOCK_N_SIZE}u * ({WORK_GROUP_BLOCK_K_SIZE}u + {PADDING}) + dotIdx;"
        )
        .unwrap();

    // Vectorized load for N register with padding
    let stride_b = WORK_GROUP_BLOCK_K_SIZE + PADDING;
    write!(&mut kernel, "            regN = vec{THREAD_BLOCK_N_SIZE}(").unwrap();
    for i in 0..THREAD_BLOCK_N_SIZE {
        if i > 0 {
            write!(&mut kernel, ", ").unwrap();
        }
        write!(&mut kernel, "{cache_b}[reg_n_offset + {}]", i * stride_b).unwrap();
    }
    writeln!(&mut kernel, ");").unwrap();

    for res_idx_m in 0..THREAD_BLOCK_M_SIZE {
        writeln!(
            &mut kernel,
            "        let result_{res_idx_m} = regM[{}] * regN;",
            res_idx_m
        )
        .unwrap();
        for res_idx_n in 0..THREAD_BLOCK_N_SIZE {
            writeln!(
                    &mut kernel,
                    "        threadResults[{} * {THREAD_BLOCK_N_SIZE}u + {}] += result_{res_idx_m}[{}];",
                    res_idx_m, res_idx_n, res_idx_n
                )
                .unwrap();
        }
    }
    writeln!(&mut kernel, "    }}").unwrap();

    if !DOUBLE_BUFFER {
        // Synchronize all threads before prefetching next tile into the same shared buffers
        writeln!(&mut kernel, "    workgroupBarrier();").unwrap();
    }

    // Prefetch tile into the alternate buffer
    writeln!(&mut kernel, "    if ((t + 1u) < tiles) {{").unwrap();
    writeln!(
        &mut kernel,
        "        let bkIdx = (t + 1u) * {WORK_GROUP_BLOCK_K_SIZE}u;"
    )
    .unwrap();
    if DOUBLE_BUFFER {
        writeln!(&mut kernel, "        let nextBuf: u32 = 1u - buf;").unwrap();
        writeln!(&mut kernel, "        let aBaseN = nextBuf * aTileSize;").unwrap();
        writeln!(&mut kernel, "        let bBaseN = nextBuf * bTileSize;").unwrap();
    }
    let add_a_base_n = if DOUBLE_BUFFER { "aBaseN + " } else { "" };
    let add_b_base_n = if DOUBLE_BUFFER { "bBaseN + " } else { "" };
    writeln!(
        &mut kernel,
        "        let fullK = (bkIdx + {WORK_GROUP_BLOCK_K_SIZE}u) <= {k_size};"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "        let fullA = (((cRow + 1u) * {WORK_GROUP_BLOCK_M_SIZE}u) <= {m_size}) && fullK;"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "        let fullB = (((cCol + 1u) * {WORK_GROUP_BLOCK_N_SIZE}u) <= {n_size}) && fullK;"
    )
    .unwrap();

    // A tile prefetch (unrolled for optimal performance)
    writeln!(&mut kernel, "        if (fullA) {{").unwrap();
    let pef = pre_element_wise_functions.get_or_init(|| {
        std::array::from_fn(|i| matmul.pre_element_wise[i].add_functions(generic_kernel))
    });
    for i in 0..UNROLL_COUNT_A {
        writeln!(&mut kernel, "            {{").unwrap();
        write_row_column(&mut kernel, i, "A");
        writeln!(
            &mut kernel,
            "                let a_row = cRow * {WORK_GROUP_BLOCK_M_SIZE}u + row_raw;"
        )
        .unwrap();
        writeln!(&mut kernel, "                let a_col = bkIdx + col_raw;").unwrap();
        writeln!(&mut kernel, "                var a_val = ").unwrap();
        let first_value_fast = pef[0].iter().fold(
            format!("{input_a}[a_start_index + a_row * {k_size} + a_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{first_value_fast};").unwrap();
        writeln!(&mut kernel, "                {cache_a}[{add_a_base_n}row_raw * ({WORK_GROUP_BLOCK_K_SIZE}u + {PADDING}) + col_raw] = a_val;").unwrap();
        writeln!(&mut kernel, "            }}").unwrap();
    }
    writeln!(&mut kernel, "        }} else {{").unwrap();
    for i in 0..UNROLL_COUNT_A {
        writeln!(&mut kernel, "            {{").unwrap();
        write_row_column(&mut kernel, i, "A");
        writeln!(
            &mut kernel,
            "                let a_row_global = cRow * {WORK_GROUP_BLOCK_M_SIZE}u + row_raw;"
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
        writeln!(&mut kernel, "                {cache_a}[{add_a_base_n}row_raw * ({WORK_GROUP_BLOCK_K_SIZE}u + {PADDING}) + col_raw] = a_val;").unwrap();
        writeln!(&mut kernel, "            }}").unwrap();
    }
    writeln!(&mut kernel, "        }}").unwrap();

    // B tile prefetch (unrolled for optimal performance) to transposed layout
    writeln!(&mut kernel, "        if (fullB) {{").unwrap();
    for i in 0..UNROLL_COUNT_B {
        writeln!(&mut kernel, "            {{").unwrap();
        writeln!(&mut kernel, "                let row_raw = innerRowB + {i}u * (numThreadsBlocktile / {WORK_GROUP_BLOCK_N_SIZE}u);").unwrap();
        writeln!(&mut kernel, "                let col_raw = innerColB;").unwrap();
        writeln!(&mut kernel, "                let b_row = bkIdx + row_raw;").unwrap();
        writeln!(
            &mut kernel,
            "                let b_col = cCol * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw;"
        )
        .unwrap();
        writeln!(&mut kernel, "                var b_val = ").unwrap();
        let second_value_fast = pef[1].iter().fold(
            format!("{input_b}[b_start_index + b_row * {n_size} + b_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{second_value_fast};").unwrap();
        writeln!(&mut kernel, "                {cache_b}[{add_b_base_n}col_raw * ({WORK_GROUP_BLOCK_K_SIZE}u + {PADDING}) + row_raw] = b_val;").unwrap();
        writeln!(&mut kernel, "            }}").unwrap();
    }
    writeln!(&mut kernel, "        }} else {{").unwrap();
    for i in 0..UNROLL_COUNT_B {
        writeln!(&mut kernel, "            {{").unwrap();
        writeln!(&mut kernel, "                let row_raw = innerRowB + {i}u * (numThreadsBlocktile / {WORK_GROUP_BLOCK_N_SIZE}u);").unwrap();
        writeln!(&mut kernel, "                let col_raw = innerColB;").unwrap();
        writeln!(
            &mut kernel,
            "                let b_row_global = bkIdx + row_raw;"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "                let b_col_global = cCol * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw;"
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
        writeln!(&mut kernel, "                {cache_b}[{add_b_base_n}col_raw * ({WORK_GROUP_BLOCK_K_SIZE}u + {PADDING}) + row_raw] = b_val;").unwrap();
        writeln!(&mut kernel, "            }}").unwrap();
    }
    writeln!(&mut kernel, "        }}").unwrap();
    writeln!(&mut kernel, "    }}").unwrap();

    // Synchronize before using the newly prefetched buffer in the next iteration
    writeln!(&mut kernel, "    workgroupBarrier();").unwrap();
    if DOUBLE_BUFFER {
        // Toggle the buffer index for the next iteration
        writeln!(&mut kernel, "    buf = 1u - buf;").unwrap();
    }
    writeln!(&mut kernel, "}}").unwrap();

    // Write out the results (same as previous implementation)
    writeln!(
        &mut kernel,
        "let outRowOffset = threadRow * {THREAD_BLOCK_M_SIZE}u + cRow * {WORK_GROUP_BLOCK_M_SIZE}u;"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "let outColOffset = threadCol * {THREAD_BLOCK_N_SIZE}u + cCol * {WORK_GROUP_BLOCK_N_SIZE}u;"
    )
    .unwrap();
    writeln!(
        &mut kernel,
        "if (outRowOffset < {m_size} && outColOffset < {n_size}) {{"
    )
    .unwrap();
    for res_idx_m in 0..THREAD_BLOCK_M_SIZE {
        writeln!(
            &mut kernel,
            "let outRow{res_idx_m} = min(outRowOffset + {res_idx_m}, {m_size} - 1);"
        )
        .unwrap();
    }
    for res_idx_n in 0..THREAD_BLOCK_N_SIZE {
        writeln!(
            &mut kernel,
            "let outCol{res_idx_n} = min(outColOffset + {res_idx_n}, {n_size} - 1);"
        )
        .unwrap();
    }
    for res_idx_m in 0..THREAD_BLOCK_M_SIZE {
        for res_idx_n in 0..THREAD_BLOCK_N_SIZE {
            let post_element_wise_functions = post_element_wise_functions
                .get_or_init(|| matmul.post_element_wise.add_functions(generic_kernel));
            write!(
                &mut kernel,
                "{output}[c_start_index + outRow{res_idx_m} * {n_size} + outCol{res_idx_n}] = "
            )
            .unwrap();
            let result = post_element_wise_functions.iter().fold(
                    format!("threadResults[(outRow{res_idx_m} - outRowOffset) * {THREAD_BLOCK_N_SIZE}u + (outCol{res_idx_n} - outColOffset)]"),
                    |acc, f| f.call(vec![acc]),
                );
            writeln!(&mut kernel, "{result};").unwrap();
        }
    }
    writeln!(&mut kernel, "}}").unwrap();

    generic_kernel.push_body(&kernel);
}

const WORK_GROUP_BLOCK_M_SIZE: u32 = THREAD_BLOCK_M_SIZE * 16;
const WORK_GROUP_BLOCK_N_SIZE: u32 = THREAD_BLOCK_N_SIZE * 8;
const WORK_GROUP_BLOCK_K_SIZE: u32 = 8;

const THREAD_BLOCK_M_SIZE: u32 = 4;
const THREAD_BLOCK_N_SIZE: u32 = 4;

const DOUBLE_BUFFER: bool = false;
