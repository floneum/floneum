use crate::Tensor;

impl<const R: usize> Tensor<R, f32> {
    pub fn sigmoid(&self) -> Self {
        1.0 / (1.0 + (-self.clone()).exp())
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_sigmoid() {
    use crate::Device;
    let device = Device::test_instance();

    let data = [[-2., -1.], [0., 1.], [2., 3.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.sigmoid();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    assert!((output[[0, 0]] - sigmoid(data[0][0])).abs() < 0.001);
    assert!((output[[0, 1]] - sigmoid(data[0][1])).abs() < 0.001);
    assert!((output[[1, 0]] - sigmoid(data[1][0])).abs() < 0.001);
    assert!((output[[1, 1]] - sigmoid(data[1][1])).abs() < 0.001);
    assert!((output[[2, 0]] - sigmoid(data[2][0])).abs() < 0.001);
    assert!((output[[2, 1]] - sigmoid(data[2][1])).abs() < 0.001);
}
