use std::ops::Sub;

use crate::{DataType, Tensor};

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn where_cond<D2>(self, on_true: Tensor<R, D2>, on_false: Tensor<R, D2>) -> Tensor<R, D2>
    where
        D2: DataType + Sub<Tensor<R, D2>, Output = Tensor<R, D2>>,
    {
        let is_zero: Tensor<R, D2> = self.eq(D::zero());
        is_zero.clone() * on_false + (D2::one() - is_zero) * on_true
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_where_cond() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = Tensor::arange(&device, 0., 10.);
    let even = Tensor::arange(&device, 0, 10) % 2;
    let zero = Tensor::splat(&device, 0., *data.shape());
    let data_where_even = even.where_cond(data, zero);

    let result = data_where_even.as_slice().await.unwrap();
    println!("result: {:?}", result);

    assert_eq!(result[[0]], 0.);
    assert_eq!(result[[1]], 1.);
    assert_eq!(result[[2]], 0.);
    assert_eq!(result[[3]], 3.);
    assert_eq!(result[[4]], 0.);
    assert_eq!(result[[5]], 5.);
    assert_eq!(result[[6]], 0.);
    assert_eq!(result[[7]], 7.);
    assert_eq!(result[[8]], 0.);
    assert_eq!(result[[9]], 9.);
}
