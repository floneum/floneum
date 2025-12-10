use crate::{
    DataTypeEnum, ElementWiseFunctions, TILE_SIZE, Tensor, TensorData,
    compute_graph::NodeIndex,
    mir::{kernel::GenericKernel, operation::Operation},
    padded_tensor_size,
};
use std::fmt::Write;

#[derive(Debug, Clone)]
pub(crate) struct IndexSelectOperation {
    pub(crate) input: NodeIndex,
    pub(crate) indexes: NodeIndex,
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
        input: NodeIndex,
        indexes: NodeIndex,
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

    fn build_index_kernel(&self, kernel: &mut GenericKernel) {
        assert!(
            self.rank() <= 3,
            "IndexSelect only supports up to 3 rank tensors"
        );

        let tile_size = self.tile_size;
        let rank = self.rank();

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
                kernel,
                "let tile_index_{i} = {global_id}.{index} * {tile_size};"
            )
            .unwrap();
        }
        writeln!(kernel, "\n").unwrap();

        for i in 0..rank {
            writeln!(kernel, "for (var local_index_{i} = 0u; local_index_{i} < {tile_size}; local_index_{i}++) {{").unwrap();
        }

        for i in 0..rank {
            writeln!(
                kernel,
                "let merged_index_{i} = tile_index_{i} + local_index_{i};"
            )
            .unwrap();
        }

        output.check_bounds(
            kernel,
            (0..).map(|i| format!("merged_index_{i}")),
            |kernel| {
                write!(kernel, "let indices_memory_index = ").unwrap();
                indexes.strided_index(
                    kernel,
                    (0..rank).map(|_| format!("merged_index_{}", self.dimension)),
                );
                writeln!(kernel, ";").unwrap();
                writeln!(
                    kernel,
                    "let select_index_value = {indexes}[indices_memory_index];"
                )
                .unwrap();
                write!(kernel, "let select_index = ",).unwrap();
                write!(kernel, "{}", process_index_input("select_index_value")).unwrap();
                writeln!(kernel, ";").unwrap();
                write!(kernel, "let input_index = ",).unwrap();
                input.strided_index(
                    kernel,
                    (0..).map(|i| {
                        if i == self.dimension {
                            "select_index".to_string()
                        } else {
                            format!("merged_index_{i}")
                        }
                    }),
                );
                writeln!(kernel, ";").unwrap();

                write!(kernel, "let output_index = ",).unwrap();
                output.strided_index(kernel, (0..rank).map(|i| format!("merged_index_{i}")));
                writeln!(kernel, ";").unwrap();

                writeln!(kernel, "let input = {input}[input_index];",).unwrap();

                write!(kernel, "{output}[output_index] = ").unwrap();
                write!(kernel, "{}", process_value_input("input")).unwrap();
                writeln!(kernel, ";").unwrap();
            },
        );

        for _ in 0..rank {
            writeln!(kernel, "}}").unwrap();
        }
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

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
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
        let size = padded_tensor_size(
            (output_shape.iter().copied().product::<usize>() * value.datatype().element_size())
                as u64,
        );
        let output_buf = device.create_buffer(
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
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
        _: &[crate::mir::inputs::MirValue],
        kernel: &mut GenericKernel,
    ) {
        self.build_index_kernel(kernel);
    }

    fn output(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> crate::mir::inputs::MirValue {
        inputs[2].clone()
    }

    fn name(&self) -> String {
        format!(
            "index_select_{}_{}_{}_to_{}",
            self.dimension,
            self.datatype,
            self.value_shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x"),
            self.indexes_shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
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
    println!("{as_slice:?}");
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
    println!("{as_slice:?}");
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
    println!("{as_slice:?}");
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
    println!("{as_slice:?}");
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
async fn test_reproducing_qwen_issue() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    // Create the same rotary_embeds as in the qwen test
    let rotary_embeds_data = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.6309573, 0.39810717, 0.25118864, 0.15848932],
        [2.0, 1.2619146, 0.79621434, 0.5023773, 0.31697863],
        [3.0, 1.8928719, 1.1943215, 0.7535659, 0.47546795],
    ];
    let rotary_embeds = Tensor::new(&device, &rotary_embeds_data);

    // Create the same hpos_indices as in the qwen test: [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]
    let hpos_indices = Tensor::new(
        &device,
        &[0u32, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3],
    );

    let selected = rotary_embeds.index_select(0, &hpos_indices);
    let selected_vec = selected.to_vec2().await.unwrap();

    println!("rotary_embeds: {:?}", rotary_embeds_data);
    println!("hpos_indices: {:?}", hpos_indices.to_vec1().await.unwrap());
    println!("selected: {:?}", selected_vec);

    // Check the first few results
    assert_eq!(selected_vec[0][0], 0.0); // index 0 -> row 0
    assert_eq!(selected_vec[1][0], 0.0); // index 0 -> row 0  
    assert_eq!(selected_vec[2][0], 1.0); // index 1 -> row 1
    assert_eq!(selected_vec[3][0], 1.0); // index 1 -> row 1
    assert_eq!(selected_vec[4][0], 0.0); // index 0 -> row 0
    assert_eq!(selected_vec[5][0], 0.0); // index 0 -> row 0
    assert_eq!(selected_vec[6][0], 1.0); // index 1 -> row 1
    assert_eq!(selected_vec[7][0], 1.0); // index 1 -> row 1
    assert_eq!(selected_vec[8][0], 2.0); // index 2 -> row 2
    assert_eq!(selected_vec[9][0], 2.0); // index 2 -> row 2
}

#[cfg(test)]
#[tokio::test]
async fn test_exact_failing_scenario() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    // Create rotary_embeds using the EXACT same pattern as make_embeds
    // First create inv_freq like VisionRotaryEmbedding does
    let head_dim_half = 10;
    let rope_theta: f32 = 10000.0;

    let mut inv_freq_data = Vec::new();
    for i in 0..head_dim_half {
        let freq = 1.0 / (rope_theta.powf(i as f32 / head_dim_half as f32));
        inv_freq_data.push(freq);
    }
    let inv_freq = Tensor::new(&device, &inv_freq_data).reshape([1, head_dim_half]);

    // Create sequence tensor using the EXACT same arange pattern as make_embeds
    let seq = Tensor::arange(&device, 0.0f32, 4.0f32).reshape([4, 1]);

    // Multiply to get rotary embeddings - exactly like make_embeds
    let rotary_embeds = seq.mat_mul(&inv_freq);

    println!("rotary_embeds shape: {:?}", rotary_embeds.shape());
    let rotary_vec = rotary_embeds.to_vec2().await.unwrap();
    println!("rotary_embeds first values:");
    for i in 0..4 {
        println!("  Row {}: first={}", i, rotary_vec[i][0]);
    }

    // Create the EXACT same hpos_indices as the failing test - FULL 32 elements
    let hpos_indices = Tensor::new(
        &device,
        &[
            0u32, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3,
            2, 2, 3, 3,
        ],
    );

    println!("hpos_indices: {:?}", hpos_indices.to_vec1().await.unwrap());

    // Test the index_select that's failing
    let result = rotary_embeds.index_select(0, &hpos_indices);
    let result_vec = result.to_vec2().await.unwrap();

    println!("index_select result first values:");
    for i in 0..10 {
        println!("  Row {}: first={}", i, result_vec[i][0]);
    }

    // Check the specific failing case - test the first 10 indices like in qwen
    println!("Checking specific indices (first 10):");
    let rotary_vec = rotary_embeds.to_vec2().await.unwrap();
    for i in 0..10 {
        let idx = hpos_indices.to_vec1().await.unwrap()[i];
        let actual_first = result_vec[i][0];
        let expected_first = rotary_vec[idx as usize][0];
        let match_status = if actual_first == expected_first {
            "✅"
        } else {
            "❌"
        };
        println!(
            "Index {}: idx={}, actual={}, expected={}, {}",
            i, idx, actual_first, expected_first, match_status
        );

        if i == 2 {
            // This is the failing case from the qwen test
            assert_eq!(
                actual_first, expected_first,
                "CRITICAL FAILURE at index {}: expected {}, got {}",
                i, expected_first, actual_first
            );
        }
    }

    // Create the exact same hpos_indices as in qwen test: shape [32]
    let hpos_indices = Tensor::new(
        &device,
        &[
            0u32, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3,
            2, 2, 3, 3,
        ],
    );

    println!("rotary_embeds shape: {:?}", rotary_embeds.shape());
    println!("hpos_indices shape: {:?}", hpos_indices.shape());
    println!("hpos_indices: {:?}", hpos_indices.to_vec1().await.unwrap());

    let selected = rotary_embeds.index_select(0, &hpos_indices);
    println!("selected shape: {:?}", selected.shape());

    let selected_vec = selected.to_vec2().await.unwrap();
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
    println!("{as_slice:?}");
    let expected_data = [[3. * 3. * 2., 1. * 3. * 2.], [6. * 3. * 2., 4. * 3. * 2.]];
    let expected_tensor = Tensor::new(&device, &expected_data);
    let expected_as_slice = expected_tensor.as_slice().await.unwrap();
    assert_eq!(as_slice, expected_as_slice);
}

#[cfg(test)]
#[tokio::test]
async fn test_index_select_exact_make_embeds_pattern() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    // Use the EXACT same configuration as the failing qwen test
    let hidden_size = 1280;
    let num_heads = 16;
    let head_dim = hidden_size / num_heads; // 80
    let rotary_dim = head_dim / 2; // 40 - this is what VisionRotaryEmbedding uses
    let rope_theta = 10000.0f32;

    // Create inv_freq exactly like create_inverse_frequency: step_by(2) up to rotary_dim
    let inv_freq_data: Vec<f32> = (0..rotary_dim)
        .step_by(2)
        .map(|i| 1.0 / (rope_theta.powf(i as f32 / rotary_dim as f32)))
        .collect();
    let inv_freq = Tensor::new(&device, &inv_freq_data).reshape([1, inv_freq_data.len()]);

    // Create embed_positions exactly like make_embeds: arange(0, 32) for the test
    let embed_positions = Tensor::arange(&device, 0.0f32, 32.0f32).reshape([32, 1]);

    // Create tensor exactly like make_embeds: mat_mul
    let tensor = embed_positions.mat_mul(&inv_freq);

    println!("Created tensor with shape: {:?}", tensor.shape());
    println!(
        "rotary_dim: {}, inv_freq len: {}",
        rotary_dim,
        inv_freq_data.len()
    );
    println!("inv_freq first 4 values: {:?}", &inv_freq_data[0..4]);

    // Test with the EXACT same indices that fail in qwen: first 10 of the hpos_indices
    let indices_data = vec![0u32, 0, 1, 1, 0, 0, 1, 1, 2, 2];
    let indices = Tensor::new(&device, &indices_data);

    println!("Testing with indices: {:?}", indices_data);

    let result = tensor.index_select(0, &indices);
    let result_data = result.to_vec2().await.unwrap();

    // Print first few rows to debug
    println!("First 8 rows of result (first 4 values each):");
    for i in 0..8.min(result_data.len()) {
        println!("Row {}: {:?}", i, &result_data[i][0..4]);
    }

    // Verify that repeated indices give identical results
    for i in 0..indices_data.len() / 2 {
        let idx1 = i * 2;
        let idx2 = i * 2 + 1;
        assert_eq!(
            result_data[idx1],
            result_data[idx2],
            "CRITICAL: Repeated indices {} and {} should give identical results but got:\nRow 1: {:?}\nRow 2: {:?}",
            indices_data[idx1],
            indices_data[idx2],
            &result_data[idx1][0..4],
            &result_data[idx2][0..4]
        );
    }

    println!("✅ All repeated indices matched correctly!");
}

#[cfg(test)]
#[tokio::test]
async fn test_reproduce_exact_qwen_bug() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    // Recreate the exact VisionRotaryEmbedding::new and make_embeds pattern
    let hidden_size = 1280;
    let num_heads = 16;
    let head_dim = hidden_size / num_heads; // 80
    let rotary_dim = head_dim / 2; // 40
    let rope_theta = 10000.0f32;

    // Create inv_freq exactly like create_inverse_frequency
    let inv_freq_data: Vec<f32> = (0..rotary_dim)
        .step_by(2)
        .map(|i| 1.0 / (rope_theta.powf(i as f32 / rotary_dim as f32)))
        .collect();
    let inv_freq = Tensor::new(&device, &inv_freq_data).reshape([1, inv_freq_data.len()]);

    // Create embed_positions exactly like make_embeds(32)
    let embed_positions = Tensor::arange(&device, 0.0f32, 32.0f32).reshape([32, 1]);

    // Create rotary_embeds exactly like make_embeds
    let rotary_embeds = embed_positions.mat_mul(&inv_freq);

    println!("rotary_embeds shape: {:?}", rotary_embeds.shape());

    // Create the EXACT same pos_ids from the failing test: [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 0], [2, 1], ...]
    let pos_ids_data = vec![
        [0u32, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 0],
        [2, 1],
    ];
    let pos_ids = Tensor::new(&device, &pos_ids_data);

    // Extract the first column (hpos_indices) - this is what fails
    let hpos_indices = pos_ids.i((.., 0));

    println!("pos_ids shape: {:?}", pos_ids.shape());
    println!("hpos_indices: {:?}", hpos_indices.to_vec1().await.unwrap());

    // Test index_select - this is the exact failing operation
    let rotary_pos_emb_0 = rotary_embeds.index_select(0, &hpos_indices);
    let result_data = rotary_pos_emb_0.to_vec2().await.unwrap();

    println!("rotary_pos_emb_0 result:");
    for i in 0..result_data.len() {
        let idx = hpos_indices.to_vec1().await.unwrap()[i];
        let actual_first = result_data[i][0];
        println!("Row {}: idx={}, actual={}", i, idx, actual_first);

        // Check the specific failing case from qwen test
        if i == 2 {
            let expected_first = idx as f32; // Should be 1.0, but qwen test shows 0.0
            assert_eq!(
                actual_first, expected_first,
                "CRITICAL FAILURE at index {}: expected {}, got {}",
                i, expected_first, actual_first
            );
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_manual_stride_simulation() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    // Create the exact same data as the slice, but manually with stride
    let pos_ids_data = vec![[0u32, 100], [1, 101], [2, 102], [3, 103]];
    let pos_ids = Tensor::new(&device, &pos_ids_data);

    // Create manual tensor with the same data as the slice
    let manual_data = vec![0u32, 1, 2, 3];
    let manual_indices = Tensor::new(&device, &manual_data);

    println!("=== Manual vs Slice Comparison ===");
    println!(
        "Manual indices: {:?}",
        manual_indices.to_vec1().await.unwrap()
    );

    let slice_indices = pos_ids.i((.., 0));
    println!(
        "Slice indices: {:?}",
        slice_indices.to_vec1().await.unwrap()
    );

    // Test index_select with both
    let input_data = vec![[10.0f32, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]];
    let input_tensor = Tensor::new(&device, &input_data);

    let result_manual = input_tensor.index_select(0, &manual_indices);
    let manual_result = result_manual.to_vec2().await.unwrap();

    println!("Manual indices result:");
    for i in 0..manual_result.len() {
        println!("  Row {}: {:?}", i, manual_result[i]);
    }

    let result_slice = input_tensor.index_select(0, &slice_indices);
    let slice_result = result_slice.to_vec2().await.unwrap();

    println!("Slice indices result:");
    for i in 0..slice_result.len() {
        println!("  Row {}: {:?}", i, slice_result[i]);
    }

    // They should be identical
    assert_eq!(
        manual_result, slice_result,
        "Manual and slice results should be identical"
    );
}

#[cfg(test)]
#[tokio::test]
async fn test_debug_stride_issue() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    // Create a 2D tensor and slice it to understand the stride
    let data_2d = vec![[0u32, 100], [1, 101], [2, 102], [3, 103]];
    let tensor_2d = Tensor::new(&device, &data_2d);

    // Get the slice
    let slice = tensor_2d.i((.., 0));

    println!("=== Understanding the stride issue ===");
    println!("Original 2D tensor data: {:?}", data_2d);
    println!("Slice values: {:?}", slice.to_vec1().await.unwrap());

    // Now let's test what happens when we manually create a strided tensor
    // Create a 1D tensor with the same stride pattern
    let flat_data = vec![0u32, 100, 1, 101, 2, 102, 3, 103]; // Flattened 2D data
    let flat_tensor = Tensor::new(&device, &flat_data).reshape([4, 2]);

    // Slice this to get the same pattern
    let manual_slice = flat_tensor.i((.., 0));

    println!(
        "Manual slice values: {:?}",
        manual_slice.to_vec1().await.unwrap()
    );

    // Test index_select with both slices
    let input_data = vec![[10.0f32, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]];
    let input_tensor = Tensor::new(&device, &input_data);

    let result1 = input_tensor.index_select(0, &slice);
    let result2 = input_tensor.index_select(0, &manual_slice);

    let result1_data = result1.to_vec2().await.unwrap();
    let result2_data = result2.to_vec2().await.unwrap();

    println!("Original slice result:");
    for i in 0..result1_data.len() {
        println!("  Row {}: {:?}", i, result1_data[i]);
    }

    println!("Manual slice result:");
    for i in 0..result2_data.len() {
        println!("  Row {}: {:?}", i, result2_data[i]);
    }

    // They should be identical if the stride handling is correct
    assert_eq!(
        result1_data, result2_data,
        "Both slices should give identical results"
    );
}

#[cfg(test)]
#[tokio::test]
async fn test_debug_memory_layout() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    // Create a simple 2D tensor with known pattern
    let data = vec![[0u32, 100], [1, 101], [2, 102], [3, 103]];
    let tensor_2d = Tensor::new(&device, &data);

    println!("=== Original 2D tensor ===");
    let original_vec = tensor_2d.to_vec2().await.unwrap();
    for i in 0..original_vec.len() {
        println!("  Row {}: {:?}", i, original_vec[i]);
    }

    // Create intermediate slice (before squeeze)
    let slice_2d = tensor_2d.slice([0..4, 0..1]);
    println!("=== 2D slice [0..4, 0..1] ===");
    let slice_2d_vec = slice_2d.to_vec2().await.unwrap();
    for i in 0..slice_2d_vec.len() {
        println!("  Row {}: {:?}", i, slice_2d_vec[i]);
    }

    // Create final 1D slice (after squeeze)
    let slice_1d = tensor_2d.i((.., 0));
    println!("=== 1D slice (after squeeze) ===");
    let slice_1d_vec = slice_1d.to_vec1().await.unwrap();
    for i in 0..slice_1d_vec.len() {
        println!("  Index {}: {}", i, slice_1d_vec[i]);
    }

    // The values should be [0, 1, 2, 3] for all cases
    let expected = vec![0u32, 1, 2, 3];

    println!("=== Verification ===");
    println!("Expected: {:?}", expected);

    // Check 2D slice values
    let slice_2d_flat: Vec<u32> = slice_2d_vec.iter().map(|row| row[0]).collect();
    println!("2D slice flat: {:?}", slice_2d_flat);
    assert_eq!(slice_2d_flat, expected, "2D slice should match expected");

    // Check 1D slice values
    println!("1D slice: {:?}", slice_1d_vec);
    assert_eq!(slice_1d_vec, expected, "1D slice should match expected");

    // Now test index_select
    println!("=== Index Select Test ===");
    let input_data = vec![[10.0f32, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]];
    let input_tensor = Tensor::new(&device, &input_data);

    // Direct indices
    let direct_indices = Tensor::new(&device, &expected);
    let result_direct = input_tensor.index_select(0, &direct_indices);
    let direct_result = result_direct.to_vec2().await.unwrap();

    println!("Direct indices result:");
    for i in 0..direct_result.len() {
        println!("  Row {}: {:?}", i, direct_result[i]);
    }

    // Slice indices
    let result_slice = input_tensor.index_select(0, &slice_1d);
    let slice_result = result_slice.to_vec2().await.unwrap();

    println!("Slice indices result:");
    for i in 0..slice_result.len() {
        println!("  Row {}: {:?}", i, slice_result[i]);
    }

    // They should be identical
    assert_eq!(
        direct_result, slice_result,
        "Direct and slice results should be identical"
    );
}

#[cfg(test)]
#[tokio::test]
async fn test_compare_slice_vs_direct_indices() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    // Create the same rotary_embeds as the failing test
    let hidden_size = 1280;
    let num_heads = 16;
    let head_dim = hidden_size / num_heads; // 80
    let rotary_dim = head_dim / 2; // 40
    let rope_theta = 10000.0f32;

    let inv_freq_data: Vec<f32> = (0..rotary_dim)
        .step_by(2)
        .map(|i| 1.0 / (rope_theta.powf(i as f32 / rotary_dim as f32)))
        .collect();
    let inv_freq = Tensor::new(&device, &inv_freq_data).reshape([1, inv_freq_data.len()]);
    let embed_positions = Tensor::arange(&device, 0.0f32, 32.0f32).reshape([32, 1]);
    let rotary_embeds = embed_positions.mat_mul(&inv_freq);

    // Test 1: Create indices directly (this should work)
    let direct_indices = Tensor::new(&device, &[0u32, 0, 1, 1, 0, 0, 1, 1, 2, 2]);
    let result_direct = rotary_embeds.index_select(0, &direct_indices);
    let direct_data = result_direct.to_vec2().await.unwrap();

    println!("Direct indices result:");
    for i in 0..direct_data.len() {
        println!("Row {}: actual={}", i, direct_data[i][0]);
    }

    // Test 2: Extract indices from a slice (this fails)
    let pos_ids_data = vec![
        [0u32, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 0],
        [2, 1],
    ];
    let pos_ids = Tensor::new(&device, &pos_ids_data);
    let slice_indices = pos_ids.i((.., 0));
    let result_slice = rotary_embeds.index_select(0, &slice_indices);
    let slice_data = result_slice.to_vec2().await.unwrap();

    println!("Slice indices result:");
    for i in 0..slice_data.len() {
        println!("Row {}: actual={}", i, slice_data[i][0]);
    }

    // Compare results - they should be identical but aren't
    println!("Comparing results:");
    for i in 0..direct_data.len() {
        if direct_data[i][0] != slice_data[i][0] {
            println!(
                "❌ Mismatch at row {}: direct={}, slice={}",
                i, direct_data[i][0], slice_data[i][0]
            );
        } else {
            println!("✅ Row {} matches: {}", i, direct_data[i][0]);
        }
    }

    // The critical failure
    assert_eq!(
        direct_data[2][0], slice_data[2][0],
        "Slice vs direct indices differ at row 2: direct={}, slice={}",
        direct_data[2][0], slice_data[2][0]
    );
}
