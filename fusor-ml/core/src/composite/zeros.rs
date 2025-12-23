use crate::{DataType, Device, Tensor};

impl<const R: usize, D: DataType> Tensor<R, D> {
    /// Create a tensor filled with zeros
    pub fn zeros(device: &Device, shape: [usize; R]) -> Self {
        Tensor::splat(device, D::zero(), shape)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_zeros() {
    use crate::Device;

    let device = Device::test_instance();

    let zeros = Tensor::<_, f32>::zeros(&device, [2, 2]);

    assert_eq!(zeros.shape(), &[2, 2]);

    let zeros_slice = zeros.as_slice().await.unwrap();
    assert_eq!(zeros_slice[[0, 0]], 0.);
    assert_eq!(zeros_slice[[0, 1]], 0.);
    assert_eq!(zeros_slice[[1, 0]], 0.);
    assert_eq!(zeros_slice[[1, 1]], 0.);
}
