use std::{fmt::Write, sync::{Arc, OnceLock}};

use fusor_gguf::{GgmlType, GgufReadError, GgufTensorMetadata};
use wgpu::{util::DeviceExt, CommandEncoder};

use crate::{compute_graph::AnyComputeKey, kernel::{GenericKernel, KernelGlobalSpace, KernelInputValue}, padded_tensor_size, DataType, DataTypeEnum, Device, PerformanceQueries, Tensor, TensorData};

pub struct QMatMulOperation {
    pub(crate) input: AnyComputeKey,
    pub(crate) matrix: QMatrix,
}

impl QMatMulOperation {
    pub fn new(input: AnyComputeKey, matrix: QMatrix) -> Self {
        QMatMulOperation { input, matrix }
    }
}


impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn q_mat_mul(&self, other: &QMatrix) -> Self {
        self.add_q_mat_mul(other)
    }
}

#[derive(Clone)]
pub struct QMatrix {
    shape: Box<[usize]>,
    buffer: Arc<wgpu::Buffer>,
    datatype: GgmlType
}

impl QMatrix {
    pub(crate) fn read<R: std::io::Read + std::io::Seek>(
        device: &wgpu::Device,
        metadata: &GgufTensorMetadata,
        reader: &mut R,
        tensor_data_offset: u64,
    ) -> Result<Self, GgufReadError> {
        let bytes = metadata.read_tensor_bytes(reader, tensor_data_offset)?;
        let shape = metadata.shape.iter().map(|x| *x as usize).collect();
        let ty = metadata.ty;
        let buffer = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        }));

        Ok(QMatrix {
            shape,
            buffer,
            datatype: ty
        })
    }

    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

}


const WORK_GROUP_BLOCK_M_SIZE: u32 = 8;
const WORK_GROUP_BLOCK_N_SIZE: u32 = 8;
const WORK_GROUP_BLOCK_K_SIZE: u32 = 8;

const THREAD_BLOCK_M_SIZE: u32 = 8;

const WORK_GROUP_SIZE_ELEMENT: u32 =
    (WORK_GROUP_BLOCK_N_SIZE * WORK_GROUP_BLOCK_M_SIZE) / THREAD_BLOCK_M_SIZE;
const WORK_GROUP_SIZE: [u32; 3] = [WORK_GROUP_SIZE_ELEMENT, WORK_GROUP_SIZE_ELEMENT, 1];

pub(crate) struct UntypedQMatMul {
    #[allow(unused)]
    sparse_kernel: OnceLock<GenericKernel>,
    first_dim_dense_kernel: OnceLock<GenericKernel>,
    datatype: DataTypeEnum,
    matrix: QMatrix,
}

impl UntypedQMatMul {
    pub(crate) const fn new(datatype: DataTypeEnum, matrix: QMatrix) -> Self {
        Self {
            sparse_kernel: OnceLock::new(),
            first_dim_dense_kernel: OnceLock::new(),
            datatype,
            matrix,
        }
    }

    // 1000x1000 dense matmul time on M2 mac pro 1.4743 ms
    fn compile(&self) -> &GenericKernel {
        self.first_dim_dense_kernel.get_or_init(|| {
            // based on https://siboehm.com/articles/22/CUDA-MMM
            let mut generic_kernel = GenericKernel::new();
            generic_kernel.set_workgroup_size(WORK_GROUP_SIZE);

            let mut kernel = String::new();

            let input_tensor = generic_kernel.add_tensor_input(2, false, self.datatype);
            let input_matrix = generic_kernel.add_q_matrix_input(2, self.matrix.datatype);
            let output = generic_kernel.add_tensor_input(2, true, self.datatype);

            let cache_a = generic_kernel.add_global_array(
                KernelGlobalSpace::Workgroup,
                self.datatype,
                (WORK_GROUP_SIZE_ELEMENT * WORK_GROUP_SIZE_ELEMENT).to_string(),
            );
            let cache_b = generic_kernel.add_global_array(
                KernelGlobalSpace::Workgroup,
                self.datatype,
                (WORK_GROUP_SIZE_ELEMENT * WORK_GROUP_SIZE_ELEMENT).to_string(),
            );

            let datatype = self.datatype;
            let workgroup_index = generic_kernel.workgroup_index();
            let workgroup_local_index = generic_kernel.workgroup_local_index();

            let m_size = input_tensor.shape_binding(0);
            let n_size = input_matrix.shape_binding(1);
            let k_size = input_tensor.shape_binding(1);

            writeln!(&mut kernel, "let block_row = {workgroup_index}.y;").unwrap();
            writeln!(&mut kernel, "let block_col = {workgroup_index}.x;").unwrap();
            writeln!(&mut kernel, "let thread_col = {workgroup_local_index} % {WORK_GROUP_BLOCK_N_SIZE};").unwrap();
            writeln!(&mut kernel, "let thread_row = {workgroup_local_index} / {WORK_GROUP_BLOCK_N_SIZE};").unwrap();
            writeln!(&mut kernel, "let a_thread_col = {workgroup_local_index} % {WORK_GROUP_BLOCK_K_SIZE};").unwrap();
            writeln!(&mut kernel, "let a_thread_row = {workgroup_local_index} / {WORK_GROUP_BLOCK_K_SIZE};").unwrap();
            writeln!(&mut kernel, "let b_thread_col = {workgroup_local_index} % {WORK_GROUP_BLOCK_N_SIZE};").unwrap();
            writeln!(&mut kernel, "let b_thread_row = {workgroup_local_index} / {WORK_GROUP_BLOCK_N_SIZE};").unwrap();
            writeln!(&mut kernel, "var results: array<{datatype}, {THREAD_BLOCK_M_SIZE}>;").unwrap();
            writeln!(&mut kernel, "let a_row = {k_size} * (a_thread_row + block_row * {WORK_GROUP_SIZE_ELEMENT});").unwrap();
            writeln!(&mut kernel, "var a_col = a_thread_col;").unwrap();
            writeln!(&mut kernel, "let a_row_max = {m_size} * {k_size};").unwrap();
            writeln!(&mut kernel, "let a_col_max = {k_size};").unwrap();
            writeln!(&mut kernel, "var b_row = {n_size} * b_thread_row;").unwrap(); 
            writeln!(&mut kernel, "let b_col = b_thread_col + block_col * {WORK_GROUP_SIZE_ELEMENT};").unwrap();
            writeln!(&mut kernel, "let b_row_max = {k_size} * {n_size};").unwrap();
            writeln!(&mut kernel, "let b_col_max = {n_size};").unwrap();


            writeln!(&mut kernel, "for (var block_index = 0u; block_index < {k_size}; block_index += {WORK_GROUP_BLOCK_K_SIZE}) {{").unwrap();

            writeln!(&mut kernel, "if a_col < a_col_max && a_row < a_row_max {{").unwrap();
            writeln!(&mut kernel, "let a_index = a_row + a_col;").unwrap();
            writeln!(&mut kernel, "{cache_a}[a_thread_row * {WORK_GROUP_BLOCK_K_SIZE} + a_thread_col] = {input_tensor}[a_index];").unwrap();
            writeln!(&mut kernel, "}}").unwrap();
            writeln!(&mut kernel, "else {{").unwrap();
            writeln!(&mut kernel, "{cache_a}[a_thread_row * {WORK_GROUP_BLOCK_K_SIZE} + a_thread_col] = 0.0;").unwrap();
            writeln!(&mut kernel, "}}").unwrap();
            writeln!(&mut kernel, "if b_row < b_row_max && b_col < {n_size} {{").unwrap();
            writeln!(&mut kernel, "let b_index = b_row + b_col;").unwrap();
            writeln!(&mut kernel, "{cache_b}[b_thread_row * {WORK_GROUP_BLOCK_N_SIZE} + b_thread_col] = {input_matrix}[b_index];").unwrap(); 
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
        input: &TensorData,
        query: Option<&PerformanceQueries>,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let device = input.device();
        let a_shape = input.layout().shape();
        let b_shape = &self.matrix.shape;
        let output_buf = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: padded_tensor_size(
                (a_shape[0] * b_shape[1] * input.datatype().element_size()) as u64,
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_tensor = TensorData::new_from_buffer(
            device,
            output_buf,
            &[a_shape[0], b_shape[1]],
            input.datatype(),
        );
        self.run_with_query_and_out_tensor(device, input, query, &output_tensor, command_encoder);
        output_tensor
    }

    pub fn run_with_query_and_out_tensor(
        &self,
        device: &Device,
        input: &TensorData,
        query: Option<&PerformanceQueries>,
        output_tensor: &TensorData,
        command_encoder: &mut CommandEncoder,
    ) {
        let matrix_shape = &self.matrix.shape;
        assert_eq!(input.layout().shape()[1], matrix_shape[0]);
        let module = self.compile();

        let a_shape = input.layout().shape();
        let b_shape = matrix_shape;
        assert_eq!(*output_tensor.layout().shape(), [a_shape[0], b_shape[1]]);

        let workgroup_dispatch_size = [
            (a_shape[0] as u32).div_ceil(WORK_GROUP_SIZE[0] * THREAD_BLOCK_M_SIZE),
            (b_shape[1] as u32).div_ceil(WORK_GROUP_SIZE[1] * THREAD_BLOCK_M_SIZE),
            1,
        ];

        module.run_with_query(
            device,
            [KernelInputValue::from(input.clone()), self.matrix.clone().into(), output_tensor.clone().into()],
            query,
            command_encoder,
            workgroup_dispatch_size,
        );
    }
}
