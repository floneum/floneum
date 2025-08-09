use std::{fmt::Write, sync::OnceLock};

use crate::mir::globals::KernelGlobalSpace;
use crate::mir::operation::Operation;
use crate::{
    Device, ElementWiseFunctions, Tensor,
    compute_graph::AnyComputeKey,
    mir::{function::Function, kernel::GenericKernel},
    tensor::{DataType, DataTypeEnum, TensorData},
};

#[derive(Debug, Clone)]
pub(crate) struct MatMulOperation {
    pub(crate) datatype: DataTypeEnum,
    pub(crate) first: AnyComputeKey,
    pub(crate) second: AnyComputeKey,
    pub(crate) first_shape: Box<[usize]>,
    pub(crate) second_shape: Box<[usize]>,
    pub(crate) out_shape: Box<[usize]>,
    pub(crate) pre_element_wise: [ElementWiseFunctions; 2],
    pub(crate) post_element_wise: ElementWiseFunctions,
}

impl MatMulOperation {
    pub fn new(
        datatype: DataTypeEnum,
        first: AnyComputeKey,
        second: AnyComputeKey,
        first_shape: &[usize],
        second_shape: &[usize],
    ) -> Self {
        let last_dim = first_shape.len() - 1;
        let second_to_last_dim = first_shape.len() - 2;
        let mut out_shape = first_shape.to_vec();
        out_shape[second_to_last_dim] = first_shape[second_to_last_dim];
        out_shape[last_dim] = second_shape[last_dim];
        assert_eq!(first_shape[last_dim], second_shape[second_to_last_dim]);
        assert!(
            first_shape
                .iter()
                .rev()
                .skip(2)
                .zip(second_shape.iter().rev().skip(2))
                .all(|(a, b)| a == b)
        );

        Self {
            first,
            second,
            first_shape: first_shape.into(),
            second_shape: second_shape.into(),
            out_shape: out_shape.into(),
            datatype,
            pre_element_wise: [
                ElementWiseFunctions::empty(datatype),
                ElementWiseFunctions::empty(datatype),
            ],
            post_element_wise: ElementWiseFunctions::empty(datatype),
        }
    }

    pub fn rank(&self) -> u32 {
        self.out_shape.len() as u32
    }

    pub(crate) fn set_pre_element_wise(&mut self, pre_element_wise: [ElementWiseFunctions; 2]) {
        self.pre_element_wise = pre_element_wise;
    }

    pub(crate) fn set_post_element_wise(&mut self, post_element_wise: ElementWiseFunctions) {
        self.post_element_wise = post_element_wise;
    }
}

impl Operation for MatMulOperation {
    fn workgroup_shape_constraints(
        &self,
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

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> [u32; 3] {
        let [input_a, input_b, _output] = inputs else {
            panic!("MatMulOperation requires 3 inputs");
        };
        let input_a = input_a.as_tensor().unwrap();
        let input_b = input_b.as_tensor().unwrap();
        let a_shape = input_a.layout().shape();
        let b_shape = input_b.layout().shape();
        let last_dim = self.rank() as usize - 1;
        let second_to_last_dim = self.rank() as usize - 2;
        let batch_size = a_shape.iter().rev().skip(2).product::<usize>();

        [
            (b_shape[last_dim] as u32).div_ceil(WORK_GROUP_BLOCK_N_SIZE),
            (a_shape[second_to_last_dim] as u32).div_ceil(WORK_GROUP_BLOCK_M_SIZE),
            (batch_size as u32).div_ceil(workgroup_shape.z()),
        ]
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(AnyComputeKey)) {
        f(self.first);
        f(self.second);
    }

    fn inputs(
        &self,
        nodes: &crate::compute_graph::ComputeGraphInner,
    ) -> Vec<crate::mir::inputs::MirValue> {
        let a = nodes.get_result(self.first).unwrap();
        let b = nodes.get_result(self.second).unwrap();
        let last_dim = self.rank() as usize - 1;
        let second_to_last_dim = self.rank() as usize - 2;
        let device = a.device();
        let a_shape = a.layout().shape();
        let b_shape = b.layout().shape();
        let mut out_shape = a_shape.to_vec();
        out_shape[second_to_last_dim] = a_shape[second_to_last_dim];
        out_shape[last_dim] = b_shape[last_dim];
        let output_tensor = TensorData::new_for_shape(device, &out_shape, a.datatype());
        vec![a.into(), b.into(), output_tensor.into()]
    }

    // 1000x1000 dense matmul time on M2 mac pro 1.4743 ms
    fn build_kernel(
        &self,
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

        let mut kernel = String::new();

        let pre_element_wise_functions: OnceLock<[Vec<Function>; 2]> = OnceLock::new();
        let post_element_wise_functions = OnceLock::new();

        let input_a = generic_kernel.add_tensor_input(self.rank(), false, input_a_datatype);
        let input_b = generic_kernel.add_tensor_input(self.rank(), false, input_b_datatype);
        let output = generic_kernel.add_tensor_input(
            self.rank(),
            true,
            self.post_element_wise.out_datatype(),
        );

        // Shared memory arrays (double-buffered)
        let cache_a = generic_kernel.add_global_array(
            KernelGlobalSpace::Workgroup,
            self.datatype,
            (WORK_GROUP_BLOCK_M_SIZE * WORK_GROUP_BLOCK_K_SIZE * 2).to_string(),
        );
        let cache_b = generic_kernel.add_global_array(
            KernelGlobalSpace::Workgroup,
            self.datatype,
            (WORK_GROUP_BLOCK_N_SIZE * WORK_GROUP_BLOCK_K_SIZE * 2).to_string(),
        );

        const TOTAL_RESULTS_BLOCK_TILE: u32 = WORK_GROUP_BLOCK_M_SIZE * WORK_GROUP_BLOCK_N_SIZE;
        const THREADS_PER_WORKGROUP: u32 =
            TOTAL_RESULTS_BLOCK_TILE / (THREAD_BLOCK_M_SIZE * THREAD_BLOCK_N_SIZE);
        assert_eq!(workgroup_shape.x(), THREADS_PER_WORKGROUP);

        let datatype = self.datatype;
        let workgroup_index = generic_kernel.workgroup_index();
        let workgroup_local_index = generic_kernel.workgroup_local_index();

        // Get dimension bindings
        let k_size = input_a.shape_binding(self.rank() - 1);
        let m_size = input_a.shape_binding(self.rank() - 2);
        let n_size = input_b.shape_binding(self.rank() - 1);

        // Map CUDA block indices to WGSL workgroup indices
        writeln!(&mut kernel, "let cRow = {workgroup_index}.y;").unwrap();
        writeln!(&mut kernel, "let cCol = {workgroup_index}.x;").unwrap();
        writeln!(&mut kernel, "var block_batch = {workgroup_index}.z;").unwrap();

        for dim in (0..self.rank()).rev().skip(2) {
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
            for dim in (0..self.rank()).rev().skip(2) {
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

        // Initialize threadResults to zero to avoid using undefined values
        let zero_literal = match datatype {
            DataTypeEnum::F16 => "f16(0.0)",
            _ => "0.0",
        };
        writeln!(
            &mut kernel,
            "for (var init_idx = 0u; init_idx < {}u; init_idx++) {{ threadResults[init_idx] = {zero}; }}",
            THREAD_BLOCK_M_SIZE * THREAD_BLOCK_N_SIZE,
            zero = zero_literal
        ).unwrap();

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

        // Double-buffered main loop over K dimension in tiles
        // Compute the number of K tiles
        writeln!(&mut kernel, "let tiles = ({k_size} + {WORK_GROUP_BLOCK_K_SIZE}u - 1u) / {WORK_GROUP_BLOCK_K_SIZE}u;").unwrap();
        writeln!(&mut kernel, "var buf: u32 = 0u;").unwrap();
        // Helper bases for double-buffered shared arrays
        writeln!(&mut kernel, "let aTileSize = {WORK_GROUP_BLOCK_M_SIZE}u * {WORK_GROUP_BLOCK_K_SIZE}u;").unwrap();
        writeln!(&mut kernel, "let bTileSize = {WORK_GROUP_BLOCK_N_SIZE}u * {WORK_GROUP_BLOCK_K_SIZE}u;").unwrap();

        // Preload tile 0 into buffer 0
        writeln!(&mut kernel, "if (tiles > 0u) {{").unwrap();
        writeln!(&mut kernel, "    let bkIdx = 0u;").unwrap();
        writeln!(&mut kernel, "    let aBase = buf * aTileSize;").unwrap();
        writeln!(&mut kernel, "    let bBase = buf * bTileSize;").unwrap();
        // Fast-path predicates for tile 0
        writeln!(&mut kernel, "    let fullK = (bkIdx + {WORK_GROUP_BLOCK_K_SIZE}u) <= {k_size};").unwrap();
        writeln!(&mut kernel, "    let fullA = (((cRow + 1u) * {WORK_GROUP_BLOCK_M_SIZE}u) <= {m_size}) && fullK;").unwrap();
        writeln!(&mut kernel, "    let fullB = (((cCol + 1u) * {WORK_GROUP_BLOCK_N_SIZE}u) <= {n_size}) && fullK;").unwrap();
        // A tile load (2 iterations unrolled)
        writeln!(&mut kernel, "    if (fullA) {{").unwrap();
        writeln!(&mut kernel, "        {{ let row_raw = innerRowA + 0u; let col_raw = innerColA; let a_row = cRow * {WORK_GROUP_BLOCK_M_SIZE}u + row_raw; let a_col = bkIdx + col_raw; var a_val = ").unwrap();
        let pef = pre_element_wise_functions.get_or_init(|| { std::array::from_fn(|i| self.pre_element_wise[i].add_functions(generic_kernel)) });
        let first_value0_fast = pef[0].iter().fold(
            format!("{input_a}[a_start_index + a_row * {k_size} + a_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{first_value0_fast}; {cache_a}[aBase + row_raw * {WORK_GROUP_BLOCK_K_SIZE}u + col_raw] = a_val; }}").unwrap();
        writeln!(&mut kernel, "        {{ let row_raw = innerRowA + (numThreadsBlocktile / {WORK_GROUP_BLOCK_K_SIZE}u); let col_raw = innerColA; let a_row = cRow * {WORK_GROUP_BLOCK_M_SIZE}u + row_raw; let a_col = bkIdx + col_raw; var a_val = ").unwrap();
        let first_value1_fast = pef[0].iter().fold(
            format!("{input_a}[a_start_index + a_row * {k_size} + a_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{first_value1_fast}; {cache_a}[aBase + row_raw * {WORK_GROUP_BLOCK_K_SIZE}u + col_raw] = a_val; }}").unwrap();
        writeln!(&mut kernel, "    }} else {{").unwrap();
        writeln!(&mut kernel, "        {{ let row_raw = innerRowA + 0u; let col_raw = innerColA; let a_row_global = cRow * {WORK_GROUP_BLOCK_M_SIZE}u + row_raw; let a_col_global = bkIdx + col_raw; let a_row = min(a_row_global, {m_size} - 1u); let a_col = min(a_col_global, {k_size} - 1u); var a_val = ").unwrap();
        let first_value0 = pef[0].iter().fold(
            format!("{input_a}[a_start_index + a_row * {k_size} + a_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{first_value0}; a_val = select({zero}, a_val, (a_row_global < {m_size} && a_col_global < {k_size})); {cache_a}[aBase + row_raw * {WORK_GROUP_BLOCK_K_SIZE}u + col_raw] = a_val; }}", zero = zero_literal).unwrap();
        writeln!(&mut kernel, "        {{ let row_raw = innerRowA + (numThreadsBlocktile / {WORK_GROUP_BLOCK_K_SIZE}u); let col_raw = innerColA; let a_row_global = cRow * {WORK_GROUP_BLOCK_M_SIZE}u + row_raw; let a_col_global = bkIdx + col_raw; let a_row = min(a_row_global, {m_size} - 1u); let a_col = min(a_col_global, {k_size} - 1u); var a_val = ").unwrap();
        let first_value1 = pef[0].iter().fold(
            format!("{input_a}[a_start_index + a_row * {k_size} + a_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{first_value1}; a_val = select({zero}, a_val, (a_row_global < {m_size} && a_col_global < {k_size})); {cache_a}[aBase + row_raw * {WORK_GROUP_BLOCK_K_SIZE}u + col_raw] = a_val; }}", zero = zero_literal).unwrap();
        writeln!(&mut kernel, "    }}").unwrap();
        // B tile load (2 iterations unrolled)
        writeln!(&mut kernel, "    if (fullB) {{").unwrap();
        writeln!(&mut kernel, "        {{ let row_raw = innerRowB + 0u; let col_raw = innerColB; let b_row = bkIdx + row_raw; let b_col = cCol * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw; var b_val = ").unwrap();
        let second_value0_fast = pef[1].iter().fold(
            format!("{input_b}[b_start_index + b_row * {n_size} + b_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{second_value0_fast}; {cache_b}[bBase + row_raw * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw] = b_val; }}").unwrap();
        writeln!(&mut kernel, "        {{ let row_raw = innerRowB + (numThreadsBlocktile / {WORK_GROUP_BLOCK_N_SIZE}u); let col_raw = innerColB; let b_row = bkIdx + row_raw; let b_col = cCol * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw; var b_val = ").unwrap();
        let second_value1_fast = pef[1].iter().fold(
            format!("{input_b}[b_start_index + b_row * {n_size} + b_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{second_value1_fast}; {cache_b}[bBase + row_raw * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw] = b_val; }}").unwrap();
        writeln!(&mut kernel, "    }} else {{").unwrap();
        writeln!(&mut kernel, "        {{ let row_raw = innerRowB + 0u; let col_raw = innerColB; let b_row_global = bkIdx + row_raw; let b_col_global = cCol * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw; let b_row = min(b_row_global, {k_size} - 1u); let b_col = min(b_col_global, {n_size} - 1u); var b_val = ").unwrap();
        let second_value0 = pef[1].iter().fold(
            format!("{input_b}[b_start_index + b_row * {n_size} + b_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{second_value0}; b_val = select({zero}, b_val, (b_row_global < {k_size} && b_col_global < {n_size})); {cache_b}[bBase + row_raw * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw] = b_val; }}", zero = zero_literal).unwrap();
        writeln!(&mut kernel, "        {{ let row_raw = innerRowB + (numThreadsBlocktile / {WORK_GROUP_BLOCK_N_SIZE}u); let col_raw = innerColB; let b_row_global = bkIdx + row_raw; let b_col_global = cCol * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw; let b_row = min(b_row_global, {k_size} - 1u); let b_col = min(b_col_global, {n_size} - 1u); var b_val = ").unwrap();
        let second_value1 = pef[1].iter().fold(
            format!("{input_b}[b_start_index + b_row * {n_size} + b_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{second_value1}; b_val = select({zero}, b_val, (b_row_global < {k_size} && b_col_global < {n_size})); {cache_b}[bBase + row_raw * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw] = b_val; }}", zero = zero_literal).unwrap();
        writeln!(&mut kernel, "    }}").unwrap();
        writeln!(&mut kernel, "}}").unwrap();

        // Synchronize after initial preload
        writeln!(&mut kernel, "workgroupBarrier();").unwrap();

        // Tiled compute with prefetch of next tile in
        writeln!(&mut kernel, "for (var t = 0u; t < tiles; t++) {{").unwrap();
        // Bases for current buffers
        writeln!(&mut kernel, "    let aBase = buf * aTileSize;").unwrap();
        writeln!(&mut kernel, "    let bBase = buf * bTileSize;").unwrap();

        // Calculate per-thread results for current tile
        writeln!(
            &mut kernel,
            "    for (var dotIdx = 0u; dotIdx < {WORK_GROUP_BLOCK_K_SIZE}u; dotIdx++) {{"
        )
        .unwrap();
        // Load values into registers from current buffer
        writeln!(&mut kernel, "        let reg_m_offset = aBase + threadRow * {THREAD_BLOCK_M_SIZE}u * {WORK_GROUP_BLOCK_K_SIZE}u + dotIdx;").unwrap();
        write!(&mut kernel, "            regM = vec{THREAD_BLOCK_M_SIZE}(").unwrap();
        for i in 0..THREAD_BLOCK_M_SIZE {
            if i > 0 { write!(&mut kernel, ", ").unwrap(); }
            write!(
                &mut kernel,
                "{cache_a}[reg_m_offset + {}]",
                i * WORK_GROUP_BLOCK_K_SIZE
            )
            .unwrap();
        }
        writeln!(&mut kernel, ");").unwrap();
        writeln!(
            &mut kernel,
            "        let reg_n_offset = bBase + dotIdx * {WORK_GROUP_BLOCK_N_SIZE}u + threadCol * {THREAD_BLOCK_N_SIZE}u;"
        )
        .unwrap();
        write!(&mut kernel, "            regN = vec{THREAD_BLOCK_N_SIZE}(").unwrap();
        for i in 0..THREAD_BLOCK_N_SIZE {
            if i > 0 { write!(&mut kernel, ", ").unwrap(); }
            write!(&mut kernel, "{cache_b}[reg_n_offset + {}]", i).unwrap();
        }
        writeln!(&mut kernel, ");").unwrap();

        // Perform outer product accumulation
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

        // Prefetch next tile into the alternate buffer
        writeln!(&mut kernel, "    if ((t + 1u) < tiles) {{").unwrap();
        writeln!(&mut kernel, "        let bkIdx = (t + 1u) * {WORK_GROUP_BLOCK_K_SIZE}u;").unwrap();
        writeln!(&mut kernel, "        let nextBuf: u32 = 1u - buf;").unwrap();
        writeln!(&mut kernel, "        let aBaseN = nextBuf * aTileSize;").unwrap();
        writeln!(&mut kernel, "        let bBaseN = nextBuf * bTileSize;").unwrap();
        writeln!(&mut kernel, "        let fullK = (bkIdx + {WORK_GROUP_BLOCK_K_SIZE}u) <= {k_size};").unwrap();
        writeln!(&mut kernel, "        let fullA = (((cRow + 1u) * {WORK_GROUP_BLOCK_M_SIZE}u) <= {m_size}) && fullK;").unwrap();
        writeln!(&mut kernel, "        let fullB = (((cCol + 1u) * {WORK_GROUP_BLOCK_N_SIZE}u) <= {n_size}) && fullK;").unwrap();

        // A tile prefetch (2 iterations unrolled)
        writeln!(&mut kernel, "        if (fullA) {{").unwrap();
        writeln!(&mut kernel, "            {{ let row_raw = innerRowA + 0u; let col_raw = innerColA; let a_row = cRow * {WORK_GROUP_BLOCK_M_SIZE}u + row_raw; let a_col = bkIdx + col_raw; var a_val = ").unwrap();
        let pef = pre_element_wise_functions.get_or_init(|| { std::array::from_fn(|i| self.pre_element_wise[i].add_functions(generic_kernel)) });
        let first_value0_fast = pef[0].iter().fold(
            format!("{input_a}[a_start_index + a_row * {k_size} + a_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{first_value0_fast}; {cache_a}[aBaseN + row_raw * {WORK_GROUP_BLOCK_K_SIZE}u + col_raw] = a_val; }}").unwrap();
        writeln!(&mut kernel, "            {{ let row_raw = innerRowA + (numThreadsBlocktile / {WORK_GROUP_BLOCK_K_SIZE}u); let col_raw = innerColA; let a_row = cRow * {WORK_GROUP_BLOCK_M_SIZE}u + row_raw; let a_col = bkIdx + col_raw; var a_val = ").unwrap();
        let first_value1_fast = pef[0].iter().fold(
            format!("{input_a}[a_start_index + a_row * {k_size} + a_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{first_value1_fast}; {cache_a}[aBaseN + row_raw * {WORK_GROUP_BLOCK_K_SIZE}u + col_raw] = a_val; }}").unwrap();
        writeln!(&mut kernel, "        }} else {{").unwrap();
        writeln!(&mut kernel, "            {{ let row_raw = innerRowA + 0u; let col_raw = innerColA; let a_row_global = cRow * {WORK_GROUP_BLOCK_M_SIZE}u + row_raw; let a_col_global = bkIdx + col_raw; let a_row = min(a_row_global, {m_size} - 1u); let a_col = min(a_col_global, {k_size} - 1u); var a_val = ").unwrap();
        let first_value0 = pef[0].iter().fold(
            format!("{input_a}[a_start_index + a_row * {k_size} + a_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{first_value0}; a_val = select({zero}, a_val, (a_row_global < {m_size} && a_col_global < {k_size})); {cache_a}[aBaseN + row_raw * {WORK_GROUP_BLOCK_K_SIZE}u + col_raw] = a_val; }}", zero = zero_literal).unwrap();
        writeln!(&mut kernel, "            {{ let row_raw = innerRowA + (numThreadsBlocktile / {WORK_GROUP_BLOCK_K_SIZE}u); let col_raw = innerColA; let a_row_global = cRow * {WORK_GROUP_BLOCK_M_SIZE}u + row_raw; let a_col_global = bkIdx + col_raw; let a_row = min(a_row_global, {m_size} - 1u); let a_col = min(a_col_global, {k_size} - 1u); var a_val = ").unwrap();
        let first_value1 = pef[0].iter().fold(
            format!("{input_a}[a_start_index + a_row * {k_size} + a_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{first_value1}; a_val = select({zero}, a_val, (a_row_global < {m_size} && a_col_global < {k_size})); {cache_a}[aBaseN + row_raw * {WORK_GROUP_BLOCK_K_SIZE}u + col_raw] = a_val; }}", zero = zero_literal).unwrap();
        writeln!(&mut kernel, "        }}").unwrap();

        // B tile prefetch (2 iterations unrolled)
        writeln!(&mut kernel, "        if (fullB) {{").unwrap();
        writeln!(&mut kernel, "            {{ let row_raw = innerRowB + 0u; let col_raw = innerColB; let b_row = bkIdx + row_raw; let b_col = cCol * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw; var b_val = ").unwrap();
        let second_value0_fast = pef[1].iter().fold(
            format!("{input_b}[b_start_index + b_row * {n_size} + b_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{second_value0_fast}; {cache_b}[bBaseN + row_raw * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw] = b_val; }}").unwrap();
        writeln!(&mut kernel, "            {{ let row_raw = innerRowB + (numThreadsBlocktile / {WORK_GROUP_BLOCK_N_SIZE}u); let col_raw = innerColB; let b_row = bkIdx + row_raw; let b_col = cCol * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw; var b_val = ").unwrap();
        let second_value1_fast = pef[1].iter().fold(
            format!("{input_b}[b_start_index + b_row * {n_size} + b_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{second_value1_fast}; {cache_b}[bBaseN + row_raw * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw] = b_val; }}").unwrap();
        writeln!(&mut kernel, "        }} else {{").unwrap();
        writeln!(&mut kernel, "            {{ let row_raw = innerRowB + 0u; let col_raw = innerColB; let b_row_global = bkIdx + row_raw; let b_col_global = cCol * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw; let b_row = min(b_row_global, {k_size} - 1u); let b_col = min(b_col_global, {n_size} - 1u); var b_val = ").unwrap();
        let second_value0 = pef[1].iter().fold(
            format!("{input_b}[b_start_index + b_row * {n_size} + b_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{second_value0}; b_val = select({zero}, b_val, (b_row_global < {k_size} && b_col_global < {n_size})); {cache_b}[bBaseN + row_raw * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw] = b_val; }}", zero = zero_literal).unwrap();
        writeln!(&mut kernel, "            {{ let row_raw = innerRowB + (numThreadsBlocktile / {WORK_GROUP_BLOCK_N_SIZE}u); let col_raw = innerColB; let b_row_global = bkIdx + row_raw; let b_col_global = cCol * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw; let b_row = min(b_row_global, {k_size} - 1u); let b_col = min(b_col_global, {n_size} - 1u); var b_val = ").unwrap();
        let second_value1 = pef[1].iter().fold(
            format!("{input_b}[b_start_index + b_row * {n_size} + b_col]"),
            |acc, f| f.call(vec![acc]),
        );
        writeln!(&mut kernel, "{second_value1}; b_val = select({zero}, b_val, (b_row_global < {k_size} && b_col_global < {n_size})); {cache_b}[bBaseN + row_raw * {WORK_GROUP_BLOCK_N_SIZE}u + col_raw] = b_val; }}", zero = zero_literal).unwrap();
        writeln!(&mut kernel, "        }}").unwrap();
        writeln!(&mut kernel, "    }}").unwrap();

        // Synchronize before using the newly prefetched buffer in the next iteration
        writeln!(&mut kernel, "    workgroupBarrier();").unwrap();
        writeln!(&mut kernel, "    buf = 1u - buf;").unwrap();
        writeln!(&mut kernel, "}}").unwrap();

        // Write out the results (same as previous implementation)
        writeln!(&mut kernel, "let outRowOffset = threadRow * {THREAD_BLOCK_M_SIZE}u + cRow * {WORK_GROUP_BLOCK_M_SIZE}u;").unwrap();
        writeln!(&mut kernel, "let outColOffset = threadCol * {THREAD_BLOCK_N_SIZE}u + cCol * {WORK_GROUP_BLOCK_N_SIZE}u;").unwrap();
        writeln!(&mut kernel, "if (outRowOffset < {m_size} && outColOffset < {n_size}) {{").unwrap();
        for res_idx_m in 0..THREAD_BLOCK_M_SIZE {
            writeln!(&mut kernel, "let outRow{res_idx_m} = min(outRowOffset + {res_idx_m}, {m_size} - 1);").unwrap();
        }
        for res_idx_n in 0..THREAD_BLOCK_N_SIZE {
            writeln!(&mut kernel, "let outCol{res_idx_n} = min(outColOffset + {res_idx_n}, {n_size} - 1);").unwrap();
        }
        for res_idx_m in 0..THREAD_BLOCK_M_SIZE {
            for res_idx_n in 0..THREAD_BLOCK_N_SIZE {
                let post_element_wise_functions = post_element_wise_functions
                    .get_or_init(|| self.post_element_wise.add_functions(generic_kernel));
                write!(&mut kernel, "{output}[c_start_index + outRow{res_idx_m} * {n_size} + outCol{res_idx_n}] = ").unwrap();
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

    fn output(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> crate::mir::inputs::MirValue {
        let output_tensor = inputs[2].as_tensor().unwrap().clone();
        output_tensor.into()
    }

    fn name(&self) -> String {
        format!(
            "matmul_{}_{}_by_{}",
            self.datatype,
            self.first_shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("x"),
            self.second_shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
    }
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn mat_mul(&self, other: &Self) -> Self {
        self.add_mat_mul(other)
    }
}

const WORK_GROUP_BLOCK_M_SIZE: u32 = THREAD_BLOCK_M_SIZE * 8;
const WORK_GROUP_BLOCK_N_SIZE: u32 = THREAD_BLOCK_N_SIZE * 8;
const WORK_GROUP_BLOCK_K_SIZE: u32 = 4;

const THREAD_BLOCK_M_SIZE: u32 = 4;
const THREAD_BLOCK_N_SIZE: u32 = 4;

#[cfg(test)]
#[tokio::test]
async fn test_matmul() {
    let device = Device::new().await.unwrap();

    let data_a = [[1.], [3.]];
    let data_b = [[1., 2.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);
    let tensor = tensor_a.mat_mul(&tensor_b);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice[[1, 1]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_matmul_fused() {
    let device = Device::new().await.unwrap();

    let data_a = [[1.], [3.]];
    let data_b = [[1., 2.]];
    let tensor_a = Tensor::new(&device, &data_a) * 2.;
    let tensor_b = Tensor::new(&device, &data_b);
    let tensor = tensor_a.mat_mul(&tensor_b) / 4.;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], 1. / 2.);
    assert_eq!(as_slice[[0, 1]], 2. / 2.);
    assert_eq!(as_slice[[1, 0]], 3. / 2.);
    assert_eq!(as_slice[[1, 1]], 6. / 2.);
}

#[cfg(test)]
#[tokio::test]
async fn test_transposed_matmul() {
    let device = Device::new().await.unwrap();

    let data_a = [[1.], [3.]];
    let data_b = [[1., 2.]];
    let tensor_a = Tensor::new(&device, &data_a).t();
    let tensor_b = Tensor::new(&device, &data_b).t();
    let tensor = tensor_a.mat_mul(&tensor_b);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], 7.);
}

#[cfg(test)]
#[tokio::test]
async fn test_batched_matmul() {
    let device = Device::new().await.unwrap();

    let data_a = [[[1.], [3.]], [[2.], [6.]]];
    let data_b = [[[1., 2.]], [[2., 4.]]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);
    let tensor = tensor_a.mat_mul(&tensor_b);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0, 0]], 1.);
    assert_eq!(as_slice[[0, 0, 1]], 2.);
    assert_eq!(as_slice[[0, 1, 0]], 3.);
    assert_eq!(as_slice[[0, 1, 1]], 6.);

    assert_eq!(as_slice[[1, 0, 0]], 4.);
    assert_eq!(as_slice[[1, 0, 1]], 8.);
    assert_eq!(as_slice[[1, 1, 0]], 12.);
    assert_eq!(as_slice[[1, 1, 1]], 24.);
}

#[cfg(test)]
#[tokio::test]
async fn test_matmul_f16() {
    let device = Device::new().await.unwrap();

    let data_a = [[half::f16::from_f32(1.)], [half::f16::from_f32(3.)]];
    let data_b = [[half::f16::from_f32(1.), half::f16::from_f32(2.)]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = tensor_a.mat_mul(&tensor_b);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], half::f16::from_f32(1.));
    assert_eq!(as_slice[[0, 1]], half::f16::from_f32(2.));
    assert_eq!(as_slice[[1, 0]], half::f16::from_f32(3.));
    assert_eq!(as_slice[[1, 1]], half::f16::from_f32(6.));
}

#[cfg(test)]
#[tokio::test]
async fn fuzz_matmul() {
    use rand::Rng;

    let device = Device::new().await.unwrap();

    let min_size = 1;
    let max_size = 512;
    let iterations = if cfg!(debug_assertions) { 10 } else { 100 };

    for _ in 0..iterations {
        let size1 = rand::rng().random_range(min_size..max_size);
        let size2 = rand::rng().random_range(min_size..max_size);
        let size3 = rand::rng().random_range(min_size..max_size);

        let data_a: Vec<Vec<f32>> = (0..size1)
            .map(|_| (0..size2).map(|_| rand::random()).collect())
            .collect();
        let data_b: Vec<Vec<f32>> = (0..size2)
            .map(|_| (0..size3).map(|_| rand::random()).collect())
            .collect();

        let tensor_a = Tensor::new(&device, &data_a);
        let tensor_b = Tensor::new(&device, &data_b);

        let mut ndarray_a = ndarray::Array2::zeros((size1, size2));
        for i in 0..size1 {
            for j in 0..size2 {
                ndarray_a[[i, j]] = data_a[i][j];
            }
        }
        let mut ndarray_b = ndarray::Array2::zeros((size2, size3));
        for i in 0..size2 {
            for j in 0..size3 {
                ndarray_b[[i, j]] = data_b[i][j];
            }
        }

        let dot = ndarray_a.dot(&ndarray_b);

        let tensor = tensor_a.mat_mul(&tensor_b);
        let as_slice = tensor.as_slice().await.unwrap();
        for i in 0..size1 {
            for j in 0..size3 {
                if (as_slice[[i, j]] - dot[[i, j]]).abs() > 0.001 {
                    println!(
                        "Mismatch at ({}, {}): {} != {}",
                        i,
                        j,
                        as_slice[[i, j]],
                        dot[[i, j]]
                    );
                    panic!("fuzz failed with size ({size1}x{size2})*({size2}x{size3})");
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn fuzz_batched_matmul() {
    use rand::Rng;
    let device = Device::new().await.unwrap();

    let min_batch_size = 2;
    let max_batch_size = 20;
    let min_size = 1;
    let max_size = 512;
    let iterations = if cfg!(debug_assertions) { 10 } else { 100 };

    for _ in 0..iterations {
        let batch_size = rand::rng().random_range(min_batch_size..max_batch_size);
        let size1 = rand::rng().random_range(min_size..max_size);
        let size2 = rand::rng().random_range(min_size..max_size);
        let size3 = rand::rng().random_range(min_size..max_size);

        let data_a: Vec<Vec<Vec<f32>>> = (0..batch_size)
            .map(|_| {
                (0..size1)
                    .map(|_| (0..size2).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let data_b: Vec<Vec<Vec<f32>>> = (0..batch_size)
            .map(|_| {
                (0..size2)
                    .map(|_| (0..size3).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();

        let tensor_a = Tensor::new(&device, &data_a);
        let tensor_b = Tensor::new(&device, &data_b);

        let ndarray_a = (0..batch_size)
            .map(|i| {
                let mut array = ndarray::Array2::zeros((size1, size2));
                for j in 0..size1 {
                    for k in 0..size2 {
                        array[[j, k]] = data_a[i][j][k];
                    }
                }
                array
            })
            .collect::<Vec<_>>();

        let ndarray_b = (0..batch_size)
            .map(|i| {
                let mut array = ndarray::Array2::zeros((size2, size3));
                for j in 0..size2 {
                    for k in 0..size3 {
                        array[[j, k]] = data_b[i][j][k];
                    }
                }
                array
            })
            .collect::<Vec<_>>();
        let dot = ndarray_a
            .iter()
            .zip(ndarray_b.iter())
            .map(|(a, b)| a.dot(b))
            .collect::<Vec<_>>();

        let tensor = tensor_a.mat_mul(&tensor_b);
        let as_slice = tensor.as_slice().await.unwrap();
        for batch in 0..batch_size {
            for i in 0..size1 {
                for j in 0..size3 {
                    if (as_slice[[batch, i, j]] - dot[batch][[i, j]]).abs() > 0.001 {
                        println!(
                            "Mismatch at ({}, {}): {} != {}",
                            i,
                            j,
                            as_slice[[batch, i, j]],
                            dot[batch][[i, j]]
                        );
                        panic!("fuzz failed with size ({size1}x{size2})*({size2}x{size3})");
                    }
                }
            }
        }
    }
}
