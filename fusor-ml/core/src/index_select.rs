use crate::{
    DataTypeEnum, ElementWiseFunctions, TILE_SIZE, Tensor, TensorData,
    compute_graph::AnyComputeKey,
    mir::{kernel::GenericKernel, operation::Operation},
    padded_tensor_size,
};
use std::fmt::Write;

#[derive(Debug)]
pub(crate) struct IndexSelectOperation {
    pub(crate) input: AnyComputeKey,
    pub(crate) indexes: AnyComputeKey,
    pub(crate) datatype: DataTypeEnum,
    pub(crate) dimension: usize,
    pub(crate) tile_size: u32,
    pub(crate) value_shape: Box<[usize]>,
    pub(crate) indexes_shape: Box<[usize]>,
    pub(crate) pre_element_wise_input: ElementWiseFunctions,
    pub(crate) pre_element_wise_indexes: ElementWiseFunctions,
}

impl IndexSelectOperation {
    pub fn new(
        input: AnyComputeKey,
        indexes: AnyComputeKey,
        datatype: DataTypeEnum,
        dimension: usize,
        value_shape: &[usize],
        indexes_shape: &[usize],
    ) -> Self {
        Self {
            input,
            indexes,
            datatype,
            dimension,
            tile_size: TILE_SIZE,
            value_shape: value_shape.to_vec().into_boxed_slice(),
            indexes_shape: indexes_shape.to_vec().into_boxed_slice(),
            pre_element_wise_input: ElementWiseFunctions::empty(datatype),
            pre_element_wise_indexes: ElementWiseFunctions::empty(DataTypeEnum::U32),
        }
    }

    pub(crate) fn input_datatype(&self) -> DataTypeEnum {
        self.datatype
    }

    pub(crate) fn indexes_datatype(&self) -> DataTypeEnum {
        DataTypeEnum::U32
    }

    pub(crate) fn rank(&self) -> usize {
        self.value_shape.len()
    }

    pub(crate) fn output_shape(&self) -> Box<[usize]> {
        Self::calc_output_shape(self.dimension, &self.value_shape, &self.indexes_shape)
    }

    pub(crate) fn calc_output_shape(
        dimension: usize,
        value_shape: &[usize],
        indexes_shape: &[usize],
    ) -> Box<[usize]> {
        value_shape
            .iter()
            .enumerate()
            .map(|(i, dim)| {
                if i == dimension {
                    indexes_shape[0]
                } else {
                    *dim
                }
            })
            .collect()
    }

    pub fn set_pre_element_wise_input(
        &mut self,
        pre_element_wise: ElementWiseFunctions,
    ) -> &mut Self {
        self.pre_element_wise_input = pre_element_wise;
        self
    }

    pub fn set_pre_element_wise_indexes(
        &mut self,
        pre_element_wise: ElementWiseFunctions,
    ) -> &mut Self {
        self.pre_element_wise_indexes = pre_element_wise;
        self
    }

    fn build_index_kernel(&self, kernel: &mut GenericKernel) -> String {
        assert!(
            self.rank() <= 3,
            "IndexSelect only supports up to 3 rank tensors"
        );

        let tile_size = self.tile_size;
        let rank = self.rank();

        let mut kernel_body = String::new();
        let global_id = kernel.global_id();
        let input = kernel.add_tensor_input(self.rank() as u32, false, self.datatype);
        let indexes = kernel.add_tensor_input(1, false, DataTypeEnum::U32);
        let output = kernel.add_tensor_input(self.rank() as u32, true, self.datatype);

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

        for i in 0..self.rank() {
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
                    "let select_index_value = {indexes}[merged_index_{dimension}];",
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

impl Operation for IndexSelectOperation {
    fn workgroup_shape_constraints(
        &self,
        _: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::new();
        constraints.add_constraint(1, crate::mir::workgroup_shape::Constraint::Equals(1));
        constraints.add_constraint(2, crate::mir::workgroup_shape::Constraint::Equals(1));
        constraints
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> [u32; 3] {
        let output = inputs[2].as_tensor().unwrap();
        let output_shape = output.layout().shape();
        let workgroup_shape_x = workgroup_shape.x();
        let workgroup_shape_y = workgroup_shape.y();
        let workgroup_shape_z = workgroup_shape.z();
        let workgroup_size_x = output_shape
            .first()
            .map(|x| (*x as u32).div_ceil(self.tile_size * workgroup_shape_x))
            .unwrap_or(1);
        let workgroup_size_y = output_shape
            .get(1)
            .map(|x| (*x as u32).div_ceil(self.tile_size * workgroup_shape_y))
            .unwrap_or(1);
        let workgroup_size_z = output_shape
            .get(2)
            .map(|x| (*x as u32).div_ceil(self.tile_size * workgroup_shape_z))
            .unwrap_or(1);
        [workgroup_size_x, workgroup_size_y, workgroup_size_z]
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(AnyComputeKey)) {
        f(self.input);
        f(self.indexes);
    }

    fn inputs(
        &self,
        nodes: &crate::compute_graph::ComputeGraphInner,
    ) -> Vec<crate::mir::inputs::MirValue> {
        let value = nodes.get_result(self.input).unwrap();
        let indexes = nodes.get_result(self.indexes).unwrap();
        let device = value.device();
        let value_shape = value.layout().shape();
        let indexes_shape = indexes.layout().shape();
        let output_shape: Box<[usize]> =
            IndexSelectOperation::calc_output_shape(self.dimension, value_shape, indexes_shape);
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

        vec![value.into(), indexes.into(), output_tensor.into()]
    }

    fn build_kernel(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
        kernel: &mut GenericKernel,
    ) -> crate::mir::inputs::MirValue {
        let kernel_text = self.build_index_kernel(kernel);
        kernel.push_body(&kernel_text);
        let output = inputs[2].clone();
        output
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
async fn test_index_select_large_dim_0() {
    use rand::seq::SliceRandom;

    use crate::Device;

    let device = Device::new().await.unwrap();

    const SIZE_1: usize = 100;
    const SIZE_0: usize = 100;
    let mut indexes_array: [u32; SIZE_0] = std::array::from_fn(|i| i as u32);
    indexes_array.shuffle(&mut rand::rng());
    let data: [[f32; SIZE_1]; SIZE_0] =
        std::array::from_fn(|i| std::array::from_fn(|j| (i * SIZE_1 + j) as f32));
    let tensor = Tensor::new(&device, &data);
    let indexes = Tensor::new(&device, &indexes_array);
    let tensor = tensor.index_select(0, &indexes);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    let expected_data: [[f32; SIZE_1]; SIZE_0] = std::array::from_fn(|i| {
        let index = indexes_array[i];
        data[index as usize]
    });
    let expected_tensor = Tensor::new(&device, &expected_data);
    let expected_as_slice = expected_tensor.as_slice().await.unwrap();
    assert_eq!(as_slice, expected_as_slice);
}

#[cfg(test)]
#[tokio::test]
async fn test_index_select_dim_1() {
    use crate::Device;

    let device = Device::new().await.unwrap();

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
async fn test_index_select_large_dim_1() {
    use rand::seq::SliceRandom;

    use crate::Device;

    let device = Device::new().await.unwrap();

    const SIZE_1: usize = 100;
    const SIZE_0: usize = 100;
    let mut indexes_array: [u32; SIZE_1] = std::array::from_fn(|i| i as u32);
    indexes_array.shuffle(&mut rand::rng());
    let data: [[f32; SIZE_1]; SIZE_0] =
        std::array::from_fn(|i| std::array::from_fn(|j| (i * SIZE_1 + j) as f32));
    let tensor = Tensor::new(&device, &data);
    let indexes = Tensor::new(&device, &indexes_array);
    let tensor = tensor.index_select(1, &indexes);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    let expected_data: [[f32; SIZE_1]; SIZE_0] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let index = indexes_array[j];
            data[i][index as usize]
        })
    });
    let expected_tensor = Tensor::new(&device, &expected_data);
    let expected_as_slice = expected_tensor.as_slice().await.unwrap();
    assert_eq!(as_slice, expected_as_slice);
}

#[cfg(test)]
#[tokio::test]
async fn test_index_select_fused() {
    use crate::Device;

    let device = Device::new().await.unwrap();

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
