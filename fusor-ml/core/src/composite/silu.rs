use crate::{DataType, Tensor};

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn silu(&self) -> Self {
        // silu(x) = x / (1 + exp(-x))
        self / &(1. + (-self.clone()).exp())
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_silu() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., -2.], [-3., 4.], [5., -6.]];

    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.silu();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    let silu = |x: f32| x / (1. + (-x).exp());
    assert!((output[[0, 0]] - silu(data[0][0])).abs() < 0.001);
    assert!((output[[0, 1]] - silu(data[0][1])).abs() < 0.001);
    assert!((output[[1, 0]] - silu(data[1][0])).abs() < 0.001);
    assert!((output[[1, 1]] - silu(data[1][1])).abs() < 0.001);
    assert!((output[[2, 0]] - silu(data[2][0])).abs() < 0.001);
    assert!((output[[2, 1]] - silu(data[2][1])).abs() < 0.001);
}
