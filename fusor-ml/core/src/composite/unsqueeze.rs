use crate::{DataType, NextRank, Tensor};

fn unchecked_unsqueeze<const R1: usize, const R2: usize, D: DataType>(
    tensor: &Tensor<R1, D>,
    axis: usize,
) -> Tensor<R2, D> {
    let shape = tensor.shape();
    assert!(axis < R1);
    tensor.reshape(std::array::from_fn(|i| match i.cmp(&axis) {
        std::cmp::Ordering::Less => shape[i],
        std::cmp::Ordering::Equal => 1,
        std::cmp::Ordering::Greater => shape[i - 1],
    }))
}

pub trait Unsqueeze<const R: usize, D>: NextRank<R, D> {
    fn unsqueeze(&self, axis: usize) -> Self::NextRank;
}

impl<const R1: usize, const R2: usize, D: DataType> Unsqueeze<R2, D> for Tensor<R1, D>
where
    Self: NextRank<R2, D>,
{
    fn unsqueeze(&self, axis: usize) -> Self::NextRank {
        const {
            assert!(R1 + 1 == R2);
        }
        unchecked_unsqueeze(self, axis)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_unsqueeze() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let unsqueezed = tensor.unsqueeze(0);
    let as_slice = unsqueezed.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0, 0, 0]], 1.);
    assert_eq!(as_slice[[0, 0, 1]], 2.);
    assert_eq!(as_slice[[0, 1, 0]], 3.);
    assert_eq!(as_slice[[0, 1, 1]], 4.);
    assert_eq!(as_slice[[0, 2, 0]], 5.);
    assert_eq!(as_slice[[0, 2, 1]], 6.);
}
