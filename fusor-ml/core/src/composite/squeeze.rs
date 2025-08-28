use crate::{DataType, LastRank, Tensor};

fn unchecked_squeeze<const R1: usize, const R2: usize, D: DataType>(
    tensor: &Tensor<R1, D>,
    axis: usize,
) -> Tensor<R2, D> {
    let shape = tensor.shape();
    assert!(axis < R1);
    tensor.reshape(std::array::from_fn(|i| {
        if i < axis { shape[i] } else { shape[i + 1] }
    }))
}

pub trait Squeeze<const R: usize, D>: LastRank<R, D> {
    fn squeeze(&self, axis: usize) -> Self::LastRank;
}

impl<const R1: usize, const R2: usize, D: DataType> Squeeze<R2, D> for Tensor<R1, D>
where
    Self: LastRank<R2, D>,
{
    fn squeeze(&self, axis: usize) -> Self::LastRank {
        const {
            assert!(R1 == R2 + 1);
        }
        assert_eq!(self.shape()[axis], 1);
        unchecked_squeeze(self, axis)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_squeeze_dim0() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[[1., 2.], [3., 4.], [5., 6.]]];
    let tensor = Tensor::new(&device, &data);
    let unsqueezed = tensor.squeeze(0);
    let as_slice = unsqueezed.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice[[1, 1]], 4.);
    assert_eq!(as_slice[[2, 0]], 5.);
    assert_eq!(as_slice[[2, 1]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_squeeze_dim2() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[[1.], [2.]], [[3.], [4.]], [[5.], [6.]]];
    let tensor = Tensor::new(&device, &data);
    let unsqueezed = tensor.squeeze(2);
    let as_slice = unsqueezed.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice[[1, 1]], 4.);
    assert_eq!(as_slice[[2, 0]], 5.);
    assert_eq!(as_slice[[2, 1]], 6.);
}
