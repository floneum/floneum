use crate::{DataType, Tensor};

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn sqr(&self) -> Self {
        self * self
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_sqr() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];

    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.sqr();

    println!("{}", tensor.graphvis());

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - 1. * 1.) < 0.001);
    assert!((output[[0, 1]] - 4. * 4.) < 0.001);
    assert!((output[[1, 0]] - 9. * 9.) < 0.001);
    assert!((output[[1, 1]] - 16. * 16.) < 0.001);
    assert!((output[[2, 0]] - 25. * 25.) < 0.001);
    assert!((output[[2, 1]] - 36. * 36.) < 0.001);
}
