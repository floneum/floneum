use crate::{DataType, Tensor};

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

pub trait Unsqueeze {
    type Output;

    fn unsqueeze(&self, axis: usize) -> Self::Output;
}

macro_rules! impl_unsqueeze {
    ($R:expr) => {
        impl<D: DataType> Unsqueeze for Tensor<$R, D> {
            type Output = Tensor<{ $R + 1 }, D>;

            fn unsqueeze(&self, axis: usize) -> Self::Output {
                unchecked_unsqueeze(self, axis)
            }
        }
    };
}

impl_unsqueeze!(1);
impl_unsqueeze!(2);
impl_unsqueeze!(3);
impl_unsqueeze!(4);
impl_unsqueeze!(5);
impl_unsqueeze!(6);
impl_unsqueeze!(7);
impl_unsqueeze!(8);
impl_unsqueeze!(9);
impl_unsqueeze!(10);
impl_unsqueeze!(11);
impl_unsqueeze!(12);
impl_unsqueeze!(13);
impl_unsqueeze!(14);
impl_unsqueeze!(15);
impl_unsqueeze!(16);
impl_unsqueeze!(17);
impl_unsqueeze!(18);
impl_unsqueeze!(19);
impl_unsqueeze!(20);

#[cfg(test)]
#[tokio::test]
async fn test_unsqueeze() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let unsqueezed = tensor.unsqueeze(0);
    let as_slice = unsqueezed.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    assert_eq!(as_slice[[0, 0, 0]], 1.);
    assert_eq!(as_slice[[0, 0, 1]], 2.);
    assert_eq!(as_slice[[0, 1, 0]], 3.);
    assert_eq!(as_slice[[0, 1, 1]], 4.);
    assert_eq!(as_slice[[0, 2, 0]], 5.);
    assert_eq!(as_slice[[0, 2, 1]], 6.);
}
