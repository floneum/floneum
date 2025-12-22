use crate::{DataTypeEnum, Tensor, compute_graph::NodeIndex};

#[derive(Debug, Clone)]
pub(crate) struct IndexSelectOperation {
    pub(crate) input: NodeIndex,
    pub(crate) indexes: NodeIndex,
    pub(crate) datatype: DataTypeEnum,
    pub(crate) dimension: usize,
    pub(crate) value_shape: Box<[usize]>,
    pub(crate) indexes_shape: Box<[usize]>,
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
            value_shape: value_shape.to_vec().into_boxed_slice(),
            indexes_shape: indexes_shape.to_vec().into_boxed_slice(),
        }
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

    let device = Device::test_instance();

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

    let device = Device::test_instance();

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

    let device = Device::test_instance();

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

    let device = Device::test_instance();

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
async fn test_multiply_before_index_select() {
    use crate::Device;

    let device = Device::test_instance();

    // Test that multiply works correctly
    let data = [[1., 2., 3.], [4., 5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor * 3.;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("multiply result: {as_slice:?}");
    let expected_data = [[3., 6., 9.], [12., 15., 18.]];
    let expected_tensor = Tensor::new(&device, &expected_data);
    let expected_as_slice = expected_tensor.as_slice().await.unwrap();
    assert_eq!(as_slice, expected_as_slice);
}

#[cfg(test)]
#[tokio::test]
async fn test_index_select_prefused() {
    use crate::Device;

    let device = Device::test_instance();

    // Test just tensor * 3. -> index_select (pre-fusion only)
    let data = [[1., 2., 3.], [4., 5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let indexes = Tensor::new(&device, &[1, 0]);
    let tensor = (tensor * 3.).index_select(1, &indexes);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("prefused: {as_slice:?}");
    // tensor * 3 = [[3, 6, 9], [12, 15, 18]]
    // index_select(1, [1, 0]) -> [[6, 3], [15, 12]]
    let expected_data = [[6., 3.], [15., 12.]];
    let expected_tensor = Tensor::new(&device, &expected_data);
    let expected_as_slice = expected_tensor.as_slice().await.unwrap();
    assert_eq!(as_slice, expected_as_slice);
}

#[cfg(test)]
#[tokio::test]
async fn test_index_select_fused() {
    use crate::Device;

    let device = Device::test_instance();

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
