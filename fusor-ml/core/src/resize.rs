use std::{fmt::Write, sync::OnceLock};

use wgpu::CommandEncoder;

use crate::{
    DataTypeEnum, PerformanceQueries, TILE_SIZE, Tensor, TensorData, compute_graph::AnyComputeKey,
    kernel::GenericKernel,
};

const BLOCKSIZE: u32 = 256;

pub(crate) struct ResizeOperation {
    pub(crate) input: AnyComputeKey,
    pub(crate) new_shape: Box<[usize]>,
    pub(crate) fill_shape: Box<[usize]>,
}

impl ResizeOperation {
    pub fn new(input: AnyComputeKey, new_shape: Box<[usize]>, fill_shape: Box<[usize]>) -> Self {
        Self {
            input,
            new_shape,
            fill_shape,
        }
    }
}

pub(crate) struct UntypedResizeKernel {
    new_shape: Box<[usize]>,
    fill_shape: Box<[usize]>,
    sparse_kernel: OnceLock<GenericKernel>,
}

impl UntypedResizeKernel {
    pub(crate) fn new(new_shape: &[usize], fill_shape: &[usize]) -> Self {
        Self {
            new_shape: new_shape.into(),
            fill_shape: fill_shape.into(),
            sparse_kernel: OnceLock::new(),
        }
    }

    fn kernel(&self, input_rank: u32, datatype: DataTypeEnum, tile_size: u32) -> &GenericKernel {
        self.sparse_kernel.get_or_init(|| {
            let mut kernel = GenericKernel::new();
            let mut kernel_body = String::new();
            let global_id = kernel.global_id();
            let input = kernel.add_tensor_input(input_rank, true, datatype);
            let output = kernel.add_tensor_input(self.new_shape.len() as u32, true, datatype);

            for local_index in 0..tile_size {
                writeln!(&mut kernel_body, "{{").unwrap();
                for (prefix, tensor) in [("input", &input), ("output", &output)] {
                    writeln!(
                    &mut kernel_body,
                    "var {prefix}_remaining_index = {global_id}.x * {tile_size} + {local_index};"
                )
                .unwrap();
                    for i in (0..tensor.rank()).rev() {
                        let shape_i = tensor.shape_binding(i);
                        writeln!(
                            &mut kernel_body,
                            "let {prefix}_index_{i} = {prefix}_remaining_index % {shape_i};",
                        )
                        .unwrap();
                        writeln!(&mut kernel_body, "{prefix}_remaining_index /= {shape_i};",)
                            .unwrap();
                    }
                }
                write!(kernel_body, "let input_index = ").unwrap();
                input.strided_index(&mut kernel_body, (0..).map(|i| format!("input_index_{i}")));
                writeln!(kernel_body, ";").unwrap();
                write!(kernel_body, "let output_index = ").unwrap();
                output.strided_index(&mut kernel_body, (0..).map(|i| format!("output_index_{i}")));
                writeln!(kernel_body, ";").unwrap();
                writeln!(
                    kernel_body,
                    "{output}[output_index] = {input}[input_index];"
                )
                .unwrap();

                writeln!(&mut kernel_body, "}}").unwrap();
            }
            kernel.set_body(kernel_body);
            kernel.set_workgroup_size([BLOCKSIZE, 1, 1]);
            kernel
        })
    }

    pub fn run_with_query(
        &self,
        input: &TensorData,
        query: Option<&PerformanceQueries>,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let rank = input.layout().rank();
        let datatype = input.datatype();

        let kernel = self.kernel(rank as u32, datatype, TILE_SIZE);
        let output = TensorData::new_for_shape(input.device(), &self.new_shape, datatype);
        let output_sliced =
            output.slice(&self.fill_shape.iter().map(|x| 0..*x).collect::<Vec<_>>());
        let tensors = vec![input.clone(), output_sliced];
        let workgroup_dispatch_size = [
            (input.layout().shape().iter().product::<usize>() as u32)
                .div_ceil(TILE_SIZE * BLOCKSIZE),
            1,
            1,
        ];
        kernel.run_with_query(
            input.device(),
            tensors,
            query,
            command_encoder,
            workgroup_dispatch_size,
        );
        output
    }
}

impl<const R: usize, T: crate::DataType> Tensor<R, T> {
    pub fn resize(&self, new_shape: [usize; R]) -> Tensor<R, T> {
        let new_shape = new_shape.into();
        let input = self.key();
        self.add_resize(ResizeOperation::new(
            input,
            new_shape,
            (*self.shape()).into(),
        ))
    }

    pub fn reshape<const R2: usize>(&self, new_shape: [usize; R2]) -> Tensor<R2, T> {
        assert_eq!(
            new_shape.iter().product::<usize>(),
            self.shape().iter().product::<usize>()
        );
        let new_shape: Box<[usize]> = new_shape.into();
        let input = self.key();
        self.add_resize(ResizeOperation::new(
            input,
            new_shape.clone(),
            new_shape.clone(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_resize() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.resize([30, 20]);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice[[1, 1]], 4.);
    assert_eq!(as_slice[[2, 0]], 5.);
    assert_eq!(as_slice[[2, 1]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_reshape() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.reshape([2, 3]);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[0, 2]], 3.);
    assert_eq!(as_slice[[1, 0]], 4.);
    assert_eq!(as_slice[[1, 1]], 5.);
    assert_eq!(as_slice[[1, 2]], 6.);

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.reshape([6]);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    assert_eq!(as_slice[[0]], 1.);
    assert_eq!(as_slice[[1]], 2.);
    assert_eq!(as_slice[[2]], 3.);
    assert_eq!(as_slice[[3]], 4.);
    assert_eq!(as_slice[[4]], 5.);
    assert_eq!(as_slice[[5]], 6.);
}
