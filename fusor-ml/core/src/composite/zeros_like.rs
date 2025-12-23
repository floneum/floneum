use crate::{DataType, Tensor};

impl<const R: usize, D: DataType> Tensor<R, D> {
    /// Create a tensor filled with zeros that has the same shape as this tensor
    pub fn zeros_like(&self) -> Self {
        Self::splat(self.device(), D::zero(), *self.shape())
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_zeros_like() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.]];
    let tensor = Tensor::new(&device, &data);
    let zeros = tensor.zeros_like();

    assert_eq!(zeros.shape(), tensor.shape());

    let zeros_slice = zeros.as_slice().await.unwrap();
    assert_eq!(zeros_slice[[0, 0]], 0.);
    assert_eq!(zeros_slice[[0, 1]], 0.);
    assert_eq!(zeros_slice[[1, 0]], 0.);
    assert_eq!(zeros_slice[[1, 1]], 0.);
}
