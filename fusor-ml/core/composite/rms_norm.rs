use crate::{CastTensor, DataType, Sum, Tensor};

impl<T> Tensor<2, T>
where
    T: DataType + CastTensor<f32>,
    f32: CastTensor<T>,
{
    pub fn layer_norm(self, weight: Tensor<1, T>, eps: f32) -> Self {
        let hidden_size = *self.shape().last().unwrap();
        let self_shape = *self.shape();
        let f32_self = self.cast::<f32>();
        let norm_x = f32_self.sqr().sum(1) / hidden_size as f32;
        let x_normed = f32_self / (norm_x + eps).sqrt().broadcast(self_shape);
        x_normed.cast::<T>() * weight.broadcast(self_shape)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_layer_norm() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });

    let tensor = Tensor::new(&device, &[[1., 2.], [3., 4.], [5., 6.]]);
    let weight = Tensor::new(&device, &[2., 3.]);
    let tensor = tensor.layer_norm(weight, 1e-5);
    let as_slice = tensor.as_slice().await.unwrap();
    assert!((as_slice[[0, 0]] - 1.2649086).abs() < 0.001);
    assert!((as_slice[[0, 1]] - 3.7947257).abs() < 0.001);
    assert!((as_slice[[1, 0]] - 1.6970556).abs() < 0.001);
    assert!((as_slice[[1, 1]] - 3.3941112).abs() < 0.001);
    assert!((as_slice[[2, 0]] - 1.8107147).abs() < 0.001);
    assert!((as_slice[[2, 1]] - 3.2592864).abs() < 0.001);
}
