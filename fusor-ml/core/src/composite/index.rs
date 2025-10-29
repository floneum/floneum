use crate::{DataType, Tensor};
use std::ops::RangeFull;

impl<const R: usize, D: DataType> Tensor<R, D> {
    /// Index into a tensor with a tuple of ranges and indices
    /// This implements indexing similar to PyTorch's tensor[(..., 0, ...)] syntax
    pub fn i<I: TensorIndex<R, D>>(&self, indices: I) -> I::Output {
        indices.index(self)
    }
}

pub trait TensorIndex<const R: usize, D: DataType> {
    type Output;

    fn index(&self, tensor: &Tensor<R, D>) -> Self::Output;
}

// Support for 3D tensor with (.., idx, ..) pattern
impl<D: DataType> TensorIndex<3, D> for (RangeFull, usize, RangeFull) {
    type Output = Tensor<2, D>;

    fn index(&self, tensor: &Tensor<3, D>) -> Tensor<2, D> {
        let shape = tensor.shape();
        let (_, idx, _) = *self;

        // Create slice that selects the specific index in dimension 1
        let sliced = tensor.slice([0..shape[0], idx..(idx + 1), 0..shape[2]]);

        sliced.squeeze(1)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_index() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]];
    let tensor = Tensor::new(&device, &data);
    let indexed = tensor.i((.., 0, ..));

    let indexed_slice = indexed.as_slice().await.unwrap();
    assert_eq!(indexed_slice[[0, 0]], 1.);
    assert_eq!(indexed_slice[[0, 1]], 2.);
    assert_eq!(indexed_slice[[1, 0]], 5.);
    assert_eq!(indexed_slice[[1, 1]], 6.);
}
