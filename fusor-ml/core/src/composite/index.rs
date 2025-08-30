use crate::{DataType, Tensor};
use std::ops::RangeFull;

impl<const R: usize, D: DataType> Tensor<R, D> {
    /// Index into a tensor with a tuple of ranges and indices
    /// This implements indexing similar to PyTorch's tensor[(..., 0, ...)] syntax
    pub fn i(&self, indices: impl TensorIndex<R>) -> crate::Result<Self, crate::Error> {
        indices.index(self)
    }
}

pub trait TensorIndex<const R: usize> {
    fn index<D: DataType>(&self, tensor: &Tensor<R, D>) -> crate::Result<Tensor<R, D>, crate::Error>;
}

// Support for 3D tensor with (.., idx, ..) pattern
impl TensorIndex<3> for (RangeFull, usize, RangeFull) {
    fn index<D: DataType>(&self, tensor: &Tensor<3, D>) -> crate::Result<Tensor<3, D>, crate::Error> {
        let shape = tensor.shape();
        let (_, idx, _) = *self;
        
        if idx >= shape[1] {
            return Err(crate::Error::GgufError(fusor_gguf::GgufReadError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Index out of bounds"))));
        }
        
        // Create slice that selects the specific index in dimension 1
        let sliced = tensor.slice([
            0..shape[0],
            idx..(idx + 1),
            0..shape[2],
        ]);
        
        Ok(sliced)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_index() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]];
    let tensor = Tensor::new(&device, &data);
    let indexed = tensor.i((.., 0, ..)).unwrap();
    
    let indexed_slice = indexed.as_slice().await.unwrap();
    assert_eq!(indexed_slice[[0, 0, 0]], 1.);
    assert_eq!(indexed_slice[[0, 0, 1]], 2.);
    assert_eq!(indexed_slice[[1, 0, 0]], 5.);
    assert_eq!(indexed_slice[[1, 0, 1]], 6.);
}