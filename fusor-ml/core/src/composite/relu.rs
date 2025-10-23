use crate::{FloatDataType, Tensor};

impl<const R: usize, D: FloatDataType> Tensor<R, D> {
    pub fn relu(&self) -> Self {
        // relu(x) = max(0, x)
        self.max_elementwise(D::from_f32(0.0))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_relu() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., -2.], [-3., 4.], [5., -6.]];

    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.relu();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    let relu = |x: f32| x.max(0.0);
    assert!((output[[0, 0]] - relu(data[0][0])).abs() < 0.001);
    assert!((output[[0, 1]] - relu(data[0][1])).abs() < 0.001);
    assert!((output[[1, 0]] - relu(data[1][0])).abs() < 0.001);
    assert!((output[[1, 1]] - relu(data[1][1])).abs() < 0.001);
    assert!((output[[2, 0]] - relu(data[2][0])).abs() < 0.001);
    assert!((output[[2, 1]] - relu(data[2][1])).abs() < 0.001);
}
