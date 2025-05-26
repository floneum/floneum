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
                WORK_GROUP_BLOCK_M_SIZE / THREAD_BLOCK_M_SIZE,
            ),
        );
        constraints.add_constraint(
            1,
            crate::mir::workgroup_shape::Constraint::Equals(WORK_GROUP_BLOCK_N_SIZE),
        );
        constraints.add_constraint(2, crate::mir::workgroup_shape::Constraint::Equals(1));
        constraints
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::KernelInputValue],
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
    ) -> Vec<crate::mir::inputs::KernelInputValue> {
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
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::KernelInputValue],
        generic_kernel: &mut GenericKernel,
    ) -> crate::mir::inputs::KernelInputValue {
        // based on https://siboehm.com/articles/22/CUDA-MMM
        let [input_a, input_b, output] = inputs else {
            panic!("MatMulOperation requires 3 inputs");
        };
        let input_a = input_a.as_tensor().unwrap();
        let input_a_datatype = input_a.datatype();
        let input_b = input_b.as_tensor().unwrap();
        let input_b_datatype = input_b.datatype();
        let output_tensor = output.as_tensor().unwrap().clone();

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

        let cache_a = generic_kernel.add_global_array(
            KernelGlobalSpace::Workgroup,
            self.datatype,
            (WORK_GROUP_BLOCK_M_SIZE * WORK_GROUP_BLOCK_K_SIZE).to_string(),
        );
        let cache_b = generic_kernel.add_global_array(
            KernelGlobalSpace::Workgroup,
            self.datatype,
            (WORK_GROUP_BLOCK_N_SIZE * WORK_GROUP_BLOCK_K_SIZE).to_string(),
        );

        let datatype = self.datatype;
        let workgroup_index = generic_kernel.workgroup_index();
        let workgroup_local_index = generic_kernel.workgroup_local_index();

        let k_size = input_a.shape_binding(self.rank() - 1);
        let a_k_stride = input_a.stride_binding(self.rank() - 1);
        let b_k_stride = input_b.stride_binding(self.rank() - 2);

        let m_size = input_a.shape_binding(self.rank() - 2);
        let a_m_stride = input_a.stride_binding(self.rank() - 2);
        let c_m_stride = output.stride_binding(self.rank() - 2);

        let n_size = input_b.shape_binding(self.rank() - 1);
        let b_n_stride = input_b.stride_binding(self.rank() - 1);
        let c_n_stride = output.stride_binding(self.rank() - 1);

        writeln!(&mut kernel, "let block_col = {workgroup_index}.x;").unwrap();
        writeln!(&mut kernel, "let block_row = {workgroup_index}.y;").unwrap();
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

        // Find the batch offset for a, b and output
        for (name, tensor) in [("a", &input_a), ("b", &input_b), ("output", &output)] {
            writeln!(&mut kernel, "let {name}_start_index = ").unwrap();
            let offset = tensor.offset_binding();
            write!(&mut kernel, "{offset}").unwrap();
            for dim in (0..self.rank()).rev().skip(2) {
                let stride = tensor.stride_binding(dim);
                write!(&mut kernel, " + block_batch_{dim}*{stride}").unwrap();
            }
            writeln!(&mut kernel, ";").unwrap();
        }

        writeln!(
            &mut kernel,
            "let thread_col = {workgroup_local_index} % {WORK_GROUP_BLOCK_N_SIZE};"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "let thread_row = {workgroup_local_index} / {WORK_GROUP_BLOCK_N_SIZE};"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "let a_thread_col = {workgroup_local_index} % {WORK_GROUP_BLOCK_K_SIZE};"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "let a_thread_row = {workgroup_local_index} / {WORK_GROUP_BLOCK_K_SIZE};"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "let b_thread_col = {workgroup_local_index} % {WORK_GROUP_BLOCK_N_SIZE};"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "let b_thread_row = {workgroup_local_index} / {WORK_GROUP_BLOCK_N_SIZE};"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "var results: array<{datatype}, {THREAD_BLOCK_M_SIZE}>;"
        )
        .unwrap();
        // Each thread in the workgroup is offset by an amount in the a matrix in the x direction. It will shift
        // by blocks inside the loop in the K direction.
        writeln!(&mut kernel, "var a_col = {a_k_stride} * a_thread_col;").unwrap();
        // Each thread in the workgroup is offset by an amount in the a matrix in the y direction.
        writeln!(
            &mut kernel,
            "let a_row = {a_m_stride} * (a_thread_row + block_row * {WORK_GROUP_BLOCK_M_SIZE});"
        )
        .unwrap();
        // The max x index on the a matrix is k size
        writeln!(&mut kernel, "let a_col_max = {k_size} * {a_k_stride};").unwrap();
        // The max y index on the a matrix is m*k size
        writeln!(&mut kernel, "let a_row_max = {m_size} * {a_m_stride};").unwrap();
        // The b matrix x index it determined by the thread and block index. It doesn't change with k
        writeln!(
            &mut kernel,
            "let b_col = {b_n_stride} * (b_thread_col + block_col * {WORK_GROUP_BLOCK_N_SIZE});"
        )
        .unwrap();
        // The b matrix y index has an offset based on the thread index. It will shift by blocks of k in the loop
        writeln!(&mut kernel, "var b_row = {b_k_stride} * b_thread_row;").unwrap();
        // The max x index on the b matrix is n size
        writeln!(&mut kernel, "let b_col_max = {n_size} * {b_n_stride};").unwrap();
        // The max y index on the b matrix is k*n size
        writeln!(&mut kernel, "let b_row_max = {k_size} * {b_k_stride};").unwrap();

        // Loop over the K dimension in blocks of WORK_GROUP_BLOCK_K_SIZE
        writeln!(&mut kernel, "for (var block_index = 0u; block_index < {k_size}; block_index += {WORK_GROUP_BLOCK_K_SIZE}) {{").unwrap();

        let pre_element_wise_functions = pre_element_wise_functions.get_or_init(|| {
            std::array::from_fn(|i| self.pre_element_wise[i].add_functions(generic_kernel))
        });

        // Make sure everything is in bounds of the a matrix
        writeln!(&mut kernel, "if a_col < a_col_max && a_row < a_row_max {{").unwrap();
        writeln!(&mut kernel, "let a_index = a_row + a_col;").unwrap();
        // If everything is in range, load the value into the workgroup cache
        write!(
            &mut kernel,
            "{cache_a}[a_thread_row * {WORK_GROUP_BLOCK_K_SIZE} + a_thread_col] = "
        )
        .unwrap();
        let first_value = pre_element_wise_functions[0]
            .iter()
            .fold(format!("{input_a}[a_start_index + a_index]"), |acc, f| {
                f.call(vec![acc])
            });
        writeln!(&mut kernel, "{first_value};").unwrap();
        writeln!(&mut kernel, "}}").unwrap();
        writeln!(&mut kernel, "else {{").unwrap();
        writeln!(
            &mut kernel,
            "{cache_a}[a_thread_row * {WORK_GROUP_BLOCK_K_SIZE} + a_thread_col] = 0.0;"
        )
        .unwrap();
        writeln!(&mut kernel, "}}").unwrap();
        // Make sure everything is in bounds of the b matrix
        writeln!(&mut kernel, "if b_col < b_col_max && b_row < b_row_max {{").unwrap();
        writeln!(&mut kernel, "let b_index = b_row + b_col;").unwrap();
        // If everything is in range, load the value into the workgroup cache
        write!(
            &mut kernel,
            "{cache_b}[b_thread_row * {WORK_GROUP_BLOCK_N_SIZE} + b_thread_col] = "
        )
        .unwrap();
        let first_value = pre_element_wise_functions[1]
            .iter()
            .fold(format!("{input_b}[b_start_index + b_index]"), |acc, f| {
                f.call(vec![acc])
            });
        writeln!(&mut kernel, "{first_value};").unwrap();
        writeln!(&mut kernel, "}}").unwrap();
        writeln!(&mut kernel, "else {{").unwrap();
        writeln!(
            &mut kernel,
            "{cache_b}[b_thread_row * {WORK_GROUP_BLOCK_N_SIZE} + b_thread_col] = 0.0;"
        )
        .unwrap();
        writeln!(&mut kernel, "}}").unwrap();

        writeln!(&mut kernel, "workgroupBarrier();").unwrap();

        // Move a forward by WORK_GROUP_BLOCK_K_SIZE
        writeln!(
            &mut kernel,
            "a_col += {WORK_GROUP_BLOCK_K_SIZE} * {a_k_stride};"
        )
        .unwrap();
        // Move b forward by WORK_GROUP_BLOCK_K_SIZE in the y direction
        writeln!(
            &mut kernel,
            "b_row += {WORK_GROUP_BLOCK_K_SIZE} * {b_k_stride};"
        )
        .unwrap();

        // Go through every row/column pair in the K dim
        writeln!(
            &mut kernel,
            "for (var dot_index = 0u; dot_index < {WORK_GROUP_BLOCK_K_SIZE}; dot_index += 1u) {{"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "let tmp = {cache_b}[dot_index * {WORK_GROUP_BLOCK_N_SIZE} + thread_col];"
        )
        .unwrap();
        writeln!(&mut kernel, "for (var result_index = 0u; result_index < {THREAD_BLOCK_M_SIZE}; result_index += 1u) {{").unwrap();
        writeln!(&mut kernel, "results[result_index] = fma({cache_a}[(thread_row * {THREAD_BLOCK_M_SIZE} + result_index) * {WORK_GROUP_BLOCK_K_SIZE} + dot_index], tmp, results[result_index]);").unwrap();
        writeln!(&mut kernel, "}}").unwrap();
        writeln!(&mut kernel, "}}").unwrap();

        writeln!(&mut kernel, "workgroupBarrier();").unwrap();

        writeln!(&mut kernel, "}}").unwrap();

        writeln!(
            &mut kernel,
            "let start_output_col = thread_col + block_col * {WORK_GROUP_BLOCK_N_SIZE};"
        )
        .unwrap();
        writeln!(&mut kernel, "let start_output_row = thread_row * {THREAD_BLOCK_M_SIZE} + block_row * {WORK_GROUP_BLOCK_M_SIZE};").unwrap();
        writeln!(&mut kernel, "for (var result_index = 0u; result_index < {THREAD_BLOCK_M_SIZE}; result_index += 1u) {{").unwrap();
        writeln!(&mut kernel, "let output_col = start_output_col;").unwrap();
        writeln!(
            &mut kernel,
            "let output_row = start_output_row + result_index;"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "if output_col < {n_size} && output_row < {m_size} {{"
        )
        .unwrap();
        writeln!(
            &mut kernel,
            "let output_index = output_row * {c_m_stride} + output_col * {c_n_stride};"
        )
        .unwrap();
        write!(
            &mut kernel,
            "{output}[output_start_index + output_index] = "
        )
        .unwrap();
        let post_element_wise_functions = post_element_wise_functions
            .get_or_init(|| self.post_element_wise.add_functions(generic_kernel));
        let result = post_element_wise_functions
            .iter()
            .fold("results[result_index]".to_string(), |acc, f| {
                f.call(vec![acc])
            });
        writeln!(&mut kernel, "{result};").unwrap();
        writeln!(&mut kernel, "}}").unwrap();
        writeln!(&mut kernel, "}}").unwrap();

        generic_kernel.set_body(kernel);

        output_tensor.into()
    }
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    // (..., M, K) @ (..., K, N) -> (..., M, N)
    pub fn mat_mul(&self, other: &Self) -> Self {
        self.add_mat_mul(other)
    }
}

const WORK_GROUP_BLOCK_M_SIZE: u32 = THREAD_BLOCK_M_SIZE * 4;
const WORK_GROUP_BLOCK_N_SIZE: u32 = 16;
const WORK_GROUP_BLOCK_K_SIZE: u32 = 4;

const THREAD_BLOCK_M_SIZE: u32 = 4;

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
    println!("{:?}", as_slice);

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
    println!("{:?}", as_slice);

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
    println!("{:?}", as_slice);

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
    println!("{:?}", as_slice);

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
    println!("{:?}", as_slice);

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
                    panic!(
                        "fuzz failed with size ({}x{})*({}x{})",
                        size1, size2, size2, size3
                    );
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
                        panic!(
                            "fuzz failed with size ({}x{})*({}x{})",
                            size1, size2, size2, size3
                        );
                    }
                }
            }
        }
    }
}
