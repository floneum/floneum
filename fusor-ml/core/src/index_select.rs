use crate::{
    DataTypeEnum, PerformanceQueries, TILE_SIZE, Tensor, TensorData, UntypedElementWiseKernel,
    compute_graph::AnyComputeKey, kernel::GenericKernel, padded_tensor_size,
};
use std::{fmt::Write, sync::OnceLock};
use wgpu::CommandEncoder;

pub(crate) struct IndexSelectOperation {
    pub(crate) input: AnyComputeKey,
    pub(crate) indexes: AnyComputeKey,
    pub(crate) dimension: usize,
}

impl IndexSelectOperation {
    pub fn new(input: AnyComputeKey, indexes: AnyComputeKey, dimension: usize) -> Self {
        Self {
            input,
            indexes,
            dimension,
        }
    }
}

pub(crate) struct UntypedIndexSelectKernel {
    dimension: usize,
    datatype: DataTypeEnum,
    rank: usize,
    tile_size: u32,
    pre_element_wise_input: UntypedElementWiseKernel,
    pre_element_wise_indexes: UntypedElementWiseKernel,
    sparse_kernel: OnceLock<GenericKernel>,
}

impl UntypedIndexSelectKernel {
    pub(crate) fn new(dimension: usize, datatype: DataTypeEnum, rank: usize) -> Self {
        Self {
            dimension,
            datatype,
            rank,
            tile_size: TILE_SIZE,
            pre_element_wise_input: UntypedElementWiseKernel::empty(datatype),
            pre_element_wise_indexes: UntypedElementWiseKernel::empty(DataTypeEnum::U32),
            sparse_kernel: OnceLock::new(),
        }
    }

    pub fn set_pre_element_wise_input(
        &mut self,
        pre_element_wise: UntypedElementWiseKernel,
    ) -> &mut Self {
        self.pre_element_wise_input = pre_element_wise;
        self
    }

    pub fn set_pre_element_wise_indexes(
        &mut self,
        pre_element_wise: UntypedElementWiseKernel,
    ) -> &mut Self {
        self.pre_element_wise_indexes = pre_element_wise;
        self
    }

    pub fn run_with_query(
        &self,
        value: &TensorData,
        indexes: &TensorData,
        query: Option<&PerformanceQueries>,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let device = value.device();
        let value_shape = value.layout().shape();
        let indexes_shape = indexes.layout().shape();
        let output_shape: Box<[usize]> = value_shape
            .iter()
            .enumerate()
            .map(|(i, dim)| {
                if i == self.dimension {
                    indexes_shape[0]
                } else {
                    *dim
                }
            })
            .collect();
        let output_buf = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: padded_tensor_size(
                (output_shape.iter().copied().product::<usize>() * value.datatype().element_size())
                    as u64,
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_tensor =
            TensorData::new_from_buffer(device, output_buf, &output_shape, value.datatype());
        self.run_with_query_and_out_tensor(value, indexes, query, command_encoder, &output_tensor);
        output_tensor
    }

    pub fn run_with_query_and_out_tensor(
        &self,
        value: &TensorData,
        indexes: &TensorData,
        query: Option<&PerformanceQueries>,
        command_encoder: &mut CommandEncoder,
        output_tensor: &TensorData,
    ) {
        // Make sure the output tensor has the correct shape
        assert!(
            output_tensor
                .layout()
                .shape()
                .iter()
                .zip(value.layout().shape())
                .enumerate()
                .all(|(i, (a, b))| if i == self.dimension {
                    a == &indexes.layout().shape()[0]
                } else {
                    a == b
                })
        );
        let output_layout = output_tensor.layout();
        let output_shape = output_layout.shape();

        let tensors = vec![value, indexes, output_tensor];
        let kernel = self.kernel();
        let max_blocksize = self.blocksize();
        let workgroup_dispatch_size = {
            let workgroup_size_x = output_shape
                .first()
                .map(|x| (*x as u32).div_ceil(self.tile_size * max_blocksize))
                .unwrap_or(1);
            let workgroup_size_y = output_shape
                .get(1)
                .map(|x| (*x as u32).div_ceil(self.tile_size * max_blocksize))
                .unwrap_or(1);
            let workgroup_size_z = output_shape
                .get(2)
                .map(|x| (*x as u32).div_ceil(self.tile_size * max_blocksize))
                .unwrap_or(1);
            [workgroup_size_x, workgroup_size_y, workgroup_size_z]
        };

        let device = value.device();
        kernel.run_with_query(
            device,
            tensors.iter().map(|x| (*x).clone()),
            query,
            command_encoder,
            workgroup_dispatch_size,
        );
    }

    fn kernel(&self) -> &GenericKernel {
        let create_kernel = || {
            let mut kernel = GenericKernel::new();
            let kernel_text = self.build_tiled_map_kernel(&mut kernel);
            kernel.set_body(kernel_text);
            let blocksize = self.blocksize();
            let workgroup_size =
                std::array::from_fn(|i| if self.rank as usize > i { blocksize } else { 1 });
            kernel.set_workgroup_size(workgroup_size);
            kernel
        };
        self.sparse_kernel.get_or_init(create_kernel)
    }

    fn blocksize(&self) -> u32 {
        // max_blocksize^R = 256
        (256f64.powf(1. / self.rank as f64)).floor() as u32
    }

    fn build_tiled_map_kernel(&self, kernel: &mut GenericKernel) -> String {
        assert!(
            self.rank <= 3,
            "IndexSelect only supports up to 3 rank tensors"
        );

        let tile_size = self.tile_size;
        let rank = self.rank;

        let mut kernel_body = String::new();
        let global_id = kernel.global_id();
        let input = kernel.add_tensor_input(self.rank as u32, false, self.datatype);
        let indexes = kernel.add_tensor_input(1, false, DataTypeEnum::U32);
        let output = kernel.add_tensor_input(self.rank as u32, true, self.datatype);

        let pre_element_wise_value = self.pre_element_wise_input.add_functions(kernel);
        let process_value_input = |input: &str| {
            pre_element_wise_value
                .iter()
                .fold(input.to_string(), |acc, f| f.call(vec![acc]))
        };
        let pre_element_wise_indexes = self.pre_element_wise_indexes.add_functions(kernel);
        let process_index_input = |input: &str| {
            pre_element_wise_indexes
                .iter()
                .fold(input.to_string(), |acc, f| f.call(vec![acc]))
        };

        for i in 0..self.rank as usize {
            let index = ["x", "y", "z"][i];
            writeln!(
                &mut kernel_body,
                "let tile_index_{i} = {global_id}.{index} * {tile_size};"
            )
            .unwrap();
        }
        writeln!(&mut kernel_body, "\n").unwrap();

        for i in 0..rank {
            writeln!(&mut kernel_body, "for (var local_index_{i} = 0u; local_index_{i} < {tile_size}; local_index_{i}++) {{").unwrap();
        }

        for i in 0..rank {
            writeln!(
                &mut kernel_body,
                "let merged_index_{i} = tile_index_{i} + local_index_{i};"
            )
            .unwrap();
        }

        output.check_bounds(
            &mut kernel_body,
            (0..).map(|i| format!("merged_index_{i}")),
            |kernel_body| {
                let dimension = self.dimension;
                writeln!(
                    kernel_body,
                    "let select_index_value = {indexes}[local_index_{dimension}];",
                )
                .unwrap();
                write!(kernel_body, "let select_index = ",).unwrap();
                write!(kernel_body, "{}", process_index_input("select_index_value")).unwrap();
                writeln!(kernel_body, ";").unwrap();
                write!(kernel_body, "let input_index = ",).unwrap();
                input.strided_index(
                    kernel_body,
                    (0..).map(|i| {
                        if i == self.dimension {
                            "select_index".to_string()
                        } else {
                            format!("merged_index_{i}")
                        }
                    }),
                );
                writeln!(kernel_body, ";").unwrap();

                write!(kernel_body, "let output_index = ",).unwrap();
                output.strided_index(kernel_body, (0..).map(|i| format!("merged_index_{i}")));
                writeln!(kernel_body, ";").unwrap();

                writeln!(kernel_body, "let input = {input}[input_index];",).unwrap();

                write!(kernel_body, "{output}[output_index] = ").unwrap();
                write!(kernel_body, "{}", process_value_input("input")).unwrap();
                writeln!(kernel_body, ";").unwrap();
            },
        );

        for _ in 0..rank {
            writeln!(&mut kernel_body, "}}").unwrap();
        }

        kernel_body
    }
}

impl<const R: usize, T: crate::DataType> Tensor<R, T> {
    pub fn index_select(&self, dimension: usize, indexes: &Tensor<1, u32>) -> Self {
        assert!(dimension < R);
        self.add_index_select(dimension, indexes)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_index_select_dim_0() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2., 3.], [4., 5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let indexes = Tensor::new(&device, &[1, 0]);
    let tensor = tensor.index_select(0, &indexes);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    let expected_data = [[4., 5., 6.], [1., 2., 3.]];
    let expected_tensor = Tensor::new(&device, &expected_data);
    let expected_as_slice = expected_tensor.as_slice().await.unwrap();
    assert_eq!(as_slice, expected_as_slice);
}

#[cfg(test)]
#[tokio::test]
async fn test_index_select_dim_1() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2., 3.], [4., 5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let indexes = Tensor::new(&device, &[1, 2, 0]);
    let tensor = tensor.index_select(1, &indexes);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    let expected_data = [[2., 3., 1.], [5., 6., 4.]];
    let expected_tensor = Tensor::new(&device, &expected_data);
    let expected_as_slice = expected_tensor.as_slice().await.unwrap();
    assert_eq!(as_slice, expected_as_slice);
}

#[cfg(test)]
#[tokio::test]
async fn test_index_select_fused() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2., 3.], [4., 5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let indexes = Tensor::new(&device, &[1, 0]);
    let tensor = (tensor * 3.).index_select(1, &(indexes * 2u32)) * 2.0;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    let expected_data = [[3. * 3. * 2., 1. * 3. * 2.], [6. * 3. * 2., 4. * 3. * 2.]];
    let expected_tensor = Tensor::new(&device, &expected_data);
    let expected_as_slice = expected_tensor.as_slice().await.unwrap();
    assert_eq!(as_slice, expected_as_slice);
}
