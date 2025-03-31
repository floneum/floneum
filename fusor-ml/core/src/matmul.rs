use std::{fmt::Write, sync::OnceLock};

use wgpu::CommandEncoder;

use crate::{
    Device, Tensor,
    compute_graph::AnyComputeKey,
    kernel::{GenericKernel, KernelGlobalSpace},
    query::PerformanceQueries,
    tensor::{DataType, DataTypeEnum, TensorData, padded_tensor_size},
};

#[derive(Clone)]
pub(crate) struct MatMulOperation {
    pub(crate) first: AnyComputeKey,
    pub(crate) second: AnyComputeKey,
}

impl MatMulOperation {
    pub fn new(first: AnyComputeKey, second: AnyComputeKey) -> Self {
        Self { first, second }
    }
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn mat_mul(&self, other: &Self) -> Self {
        self.add_mat_mul(other)
    }
}

const WORK_GROUP_BLOCK_M_SIZE: u32 = THREAD_BLOCK_M_SIZE * 2;
const WORK_GROUP_BLOCK_N_SIZE: u32 = 8;
const WORK_GROUP_BLOCK_K_SIZE: u32 = 64;

const THREAD_BLOCK_M_SIZE: u32 = 8;

const WORK_GROUP_SIZE_ELEMENT: u32 =
    (WORK_GROUP_BLOCK_N_SIZE * WORK_GROUP_BLOCK_M_SIZE) / THREAD_BLOCK_M_SIZE;
const WORK_GROUP_SIZE: [u32; 3] = [WORK_GROUP_SIZE_ELEMENT, WORK_GROUP_SIZE_ELEMENT, 1];

pub(crate) struct UntypedMatMul {
    #[allow(unused)]
    sparse_kernel: OnceLock<GenericKernel>,
    first_dim_dense_kernel: OnceLock<GenericKernel>,
    datatype: DataTypeEnum,
    rank: u32,
}

impl UntypedMatMul {
    pub(crate) const fn new(datatype: DataTypeEnum, rank: u32) -> Self {
        Self {
            sparse_kernel: OnceLock::new(),
            first_dim_dense_kernel: OnceLock::new(),
            datatype,
            rank,
        }
    }

    // 1000x1000 dense matmul time on M2 mac pro 1.4743 ms
    fn compile(&self) -> &GenericKernel {
        self.first_dim_dense_kernel.get_or_init(|| {
            // based on https://siboehm.com/articles/22/CUDA-MMM
            let mut generic_kernel = GenericKernel::new();
            generic_kernel.set_workgroup_size(WORK_GROUP_SIZE);

            let mut kernel = String::new();

            let input_a = generic_kernel.add_tensor_input(self.rank, false, self.datatype);
            let input_b = generic_kernel.add_tensor_input(self.rank, false, self.datatype);
            let output = generic_kernel.add_tensor_input(self.rank, true, self.datatype);

            let cache_a = generic_kernel.add_global_array(
                KernelGlobalSpace::Workgroup,
                self.datatype,
                (WORK_GROUP_BLOCK_K_SIZE * WORK_GROUP_BLOCK_K_SIZE).to_string(),
            );
            let cache_b = generic_kernel.add_global_array(
                KernelGlobalSpace::Workgroup,
                self.datatype,
                (WORK_GROUP_BLOCK_N_SIZE * WORK_GROUP_BLOCK_N_SIZE).to_string(),
            );

            let datatype = self.datatype;
            let workgroup_index = generic_kernel.workgroup_index();
            let workgroup_local_index = generic_kernel.workgroup_local_index();

            let m_size = input_a.shape_binding(0);
            let n_size = input_b.shape_binding(1);
            let k_size = input_a.shape_binding(1);

            writeln!(&mut kernel, "let block_row = {workgroup_index}.y;").unwrap();
            writeln!(&mut kernel, "let block_col = {workgroup_index}.x;").unwrap();
            writeln!(&mut kernel, "let thread_col = {workgroup_local_index} % {WORK_GROUP_BLOCK_N_SIZE};").unwrap();
            writeln!(&mut kernel, "let thread_row = {workgroup_local_index} / {WORK_GROUP_BLOCK_N_SIZE};").unwrap();
            writeln!(&mut kernel, "let a_thread_col = {workgroup_local_index} % {WORK_GROUP_BLOCK_K_SIZE};").unwrap();
            writeln!(&mut kernel, "let a_thread_row = {workgroup_local_index} / {WORK_GROUP_BLOCK_K_SIZE};").unwrap();
            writeln!(&mut kernel, "let b_thread_col = {workgroup_local_index} % {WORK_GROUP_BLOCK_N_SIZE};").unwrap();
            writeln!(&mut kernel, "let b_thread_row = {workgroup_local_index} / {WORK_GROUP_BLOCK_N_SIZE};").unwrap();
            writeln!(&mut kernel, "var results: array<{datatype}, {THREAD_BLOCK_M_SIZE}>;").unwrap();
            writeln!(&mut kernel, "let a_row = {k_size} * (a_thread_row + block_row * {WORK_GROUP_BLOCK_M_SIZE});").unwrap();
            writeln!(&mut kernel, "var a_col = a_thread_col;").unwrap();
            writeln!(&mut kernel, "let a_row_max = {m_size} * {k_size};").unwrap();
            writeln!(&mut kernel, "let a_col_max = {k_size};").unwrap();
            writeln!(&mut kernel, "var b_row = {n_size} * b_thread_row;").unwrap(); 
            writeln!(&mut kernel, "let b_col = b_thread_col + block_col * {WORK_GROUP_BLOCK_N_SIZE};").unwrap();
            writeln!(&mut kernel, "let b_row_max = {k_size} * {n_size};").unwrap();
            writeln!(&mut kernel, "let b_col_max = {n_size};").unwrap();


            writeln!(&mut kernel, "for (var block_index = 0u; block_index < {k_size}; block_index += {WORK_GROUP_BLOCK_K_SIZE}) {{").unwrap();

            writeln!(&mut kernel, "if a_col < a_col_max && a_row < a_row_max {{").unwrap();
            writeln!(&mut kernel, "let a_index = a_row + a_col;").unwrap();
            writeln!(&mut kernel, "{cache_a}[a_thread_row * {WORK_GROUP_BLOCK_K_SIZE} + a_thread_col] = {input_a}[a_index];").unwrap();
            writeln!(&mut kernel, "}}").unwrap();
            writeln!(&mut kernel, "else {{").unwrap();
            writeln!(&mut kernel, "{cache_a}[a_thread_row * {WORK_GROUP_BLOCK_K_SIZE} + a_thread_col] = 0.0;").unwrap();
            writeln!(&mut kernel, "}}").unwrap();
            writeln!(&mut kernel, "if b_row < b_row_max && b_col < {n_size} {{").unwrap();
            writeln!(&mut kernel, "let b_index = b_row + b_col;").unwrap();
            writeln!(&mut kernel, "{cache_b}[b_thread_row * {WORK_GROUP_BLOCK_N_SIZE} + b_thread_col] = {input_b}[b_index];").unwrap(); 
            writeln!(&mut kernel, "}}").unwrap();
            writeln!(&mut kernel, "else {{").unwrap();
            writeln!(&mut kernel, "{cache_b}[b_thread_row * {WORK_GROUP_BLOCK_N_SIZE} + b_thread_col] = 0.0;").unwrap();
            writeln!(&mut kernel, "}}").unwrap();

            writeln!(&mut kernel, "workgroupBarrier();").unwrap();

            writeln!(&mut kernel, "a_col += {WORK_GROUP_BLOCK_K_SIZE};").unwrap();
            writeln!(&mut kernel, "b_row += {WORK_GROUP_BLOCK_K_SIZE} * {n_size};").unwrap();

            writeln!(&mut kernel, "for (var dot_index = 0u; dot_index < {WORK_GROUP_BLOCK_K_SIZE}; dot_index += 1u) {{").unwrap();
            writeln!(&mut kernel, "let tmp = {cache_b}[dot_index * {WORK_GROUP_BLOCK_N_SIZE} + thread_col];").unwrap();
            writeln!(&mut kernel, "for (var result_index = 0u; result_index < {THREAD_BLOCK_M_SIZE}; result_index += 1u) {{").unwrap();
            writeln!(&mut kernel, "results[result_index] += {cache_a}[(thread_row * {THREAD_BLOCK_M_SIZE} + result_index) * {WORK_GROUP_BLOCK_K_SIZE} + dot_index] * tmp;").unwrap();
            writeln!(&mut kernel, "}}").unwrap();
            writeln!(&mut kernel, "}}").unwrap();

            writeln!(&mut kernel, "workgroupBarrier();").unwrap();

            writeln!(&mut kernel, "}}").unwrap();


            writeln!(&mut kernel, "let start_output_row = thread_row * {THREAD_BLOCK_M_SIZE} + block_row * {WORK_GROUP_BLOCK_M_SIZE};").unwrap();
            writeln!(&mut kernel, "let start_output_col = thread_col + block_col * {WORK_GROUP_BLOCK_N_SIZE};").unwrap();
            writeln!(&mut kernel, "for (var result_index = 0u; result_index < {THREAD_BLOCK_M_SIZE}; result_index += 1u) {{").unwrap();
            writeln!(&mut kernel, "let output_row = start_output_row + result_index;").unwrap();
            writeln!(&mut kernel, "let output_col = start_output_col;").unwrap();
            writeln!(&mut kernel, "if output_col < {n_size} && output_row < {m_size} {{").unwrap();
            writeln!(&mut kernel, "let output_index = output_row * {n_size} + output_col;").unwrap();
            writeln!(&mut kernel, "{output}[output_index] = results[result_index];").unwrap();
            writeln!(&mut kernel, "}}").unwrap();
            writeln!(&mut kernel, "}}").unwrap();

            generic_kernel.set_body(kernel);

            generic_kernel
        })
    }

    pub fn run_with_query(
        &self,
        a: &TensorData,
        b: &TensorData,
        query: Option<&PerformanceQueries>,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let device = a.device();
        let a_shape = a.layout().shape();
        let b_shape = b.layout().shape();
        let output_buf = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: padded_tensor_size(
                (a_shape[0] * b_shape[1] * a.datatype().element_size()) as u64,
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_tensor = TensorData::new_from_buffer(
            device,
            output_buf,
            &[a_shape[0], b_shape[1]],
            a.datatype(),
        );
        self.run_with_query_and_out_tensor(device, a, b, query, &output_tensor, command_encoder);
        output_tensor
    }

    pub fn run_with_query_and_out_tensor(
        &self,
        device: &Device,
        a: &TensorData,
        b: &TensorData,
        query: Option<&PerformanceQueries>,
        output_tensor: &TensorData,
        command_encoder: &mut CommandEncoder,
    ) {
        assert_eq!(a.layout().shape()[1], b.layout().shape()[0]);
        let module = self.compile();

        let a_shape = a.layout().shape();
        let b_shape = b.layout().shape();
        assert_eq!(*output_tensor.layout().shape(), [a_shape[0], b_shape[1]]);

        let workgroup_dispatch_size = [
            (a_shape[0] as u32).div_ceil(WORK_GROUP_SIZE[0] * THREAD_BLOCK_M_SIZE),
            (b_shape[1] as u32).div_ceil(WORK_GROUP_SIZE[1] * THREAD_BLOCK_M_SIZE),
            1,
        ];

        module.run_with_query(
            device,
            [a.clone(), b.clone(), output_tensor.clone()],
            query,
            command_encoder,
            workgroup_dispatch_size,
        );
    }
}

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
    
    let max_size = 125;
    let iterations = if cfg!(debug_assertions) { 10 } else { 100 };

    for _ in 0..iterations {
        let size1 = rand::rng().random_range(1..max_size);
        let size2 = rand::rng().random_range(1..max_size);
        let size3 = rand::rng().random_range(1..max_size);

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
                assert_eq!(as_slice[[i, j]], dot[[i, j]]);
            }
        }
    }
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
