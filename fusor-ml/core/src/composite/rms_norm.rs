use crate::{CastTensor, DataType, LastRank, MaxRank, NextRankInner, Tensor, TensorSlice};
use std::fmt::Debug;

impl<const N: usize, T> Tensor<N, T> {
    pub fn layer_norm<const R: usize, const N2: usize>(
        &self,
        weight: &Tensor<R, T>,
        bias: Option<&Tensor<R, T>>,
        eps: f32,
        remove_mean: bool,
    ) -> Self
    where
        T: DataType + CastTensor<f32>,
        f32: CastTensor<T>,
        (Tensor<N, f32>, Tensor<R, f32>): MaxRank<N, f32>,
        (Tensor<N, T>, Tensor<R, T>): MaxRank<N, T>,
        (Tensor<N, f32>, Tensor<N, f32>): MaxRank<N, f32>,
        Tensor<N, f32>: LastRank<N2, f32>,
        Tensor<N2, f32>: NextRankInner<NextRank = Tensor<N, f32>>,
        TensorSlice<N, f32>: Debug,
    {
        let self_shape = *self.shape();
        let hidden_size = self_shape.last().copied().unwrap();
        let f32_self = self.cast::<f32>();
        let f32_self = if remove_mean {
            let mean_self = f32_self.sum_keepdim(N - 1) / hidden_size as f32;
            f32_self.sub_(&mean_self)
        } else {
            f32_self
        };
        let norm_x = f32_self.sqr().sum_keepdim(N - 1) / hidden_size as f32;

        let x_normed = f32_self.div_(&(norm_x + eps).sqrt());
        let product = x_normed.cast::<T>().mul_(&weight);
        if let Some(bias) = bias {
            product.add_(&bias)
        } else {
            product
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_layer_norm() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let tensor = Tensor::new(&device, &[[1., 2.], [3., 4.], [5., 6.]]);
    let weight = Tensor::new(&device, &[2., 3.]);
    let tensor = tensor.layer_norm(&weight, None, 1e-5, false);
    let as_slice = tensor.as_slice().await.unwrap();
    assert!((as_slice[[0, 0]] - 1.2649086).abs() < 0.001);
    assert!((as_slice[[0, 1]] - 3.7947257).abs() < 0.001);
    assert!((as_slice[[1, 0]] - 1.6970556).abs() < 0.001);
    assert!((as_slice[[1, 1]] - 3.3941112).abs() < 0.001);
    assert!((as_slice[[2, 0]] - 1.8107147).abs() < 0.001);
    assert!((as_slice[[2, 1]] - 3.2592864).abs() < 0.001);
}

#[cfg(test)]
#[tokio::test]
async fn test_batched_layer_norm() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let tensor = Tensor::new(
        &device,
        &[
            [[1., 2.], [3., 4.], [5., 6.]],
            [[1., 2.], [3., 4.], [5., 6.]],
        ],
    );
    let weight = Tensor::new(&device, &[2., 3.]);
    let tensor = tensor.layer_norm(&weight, None, 1e-5, false);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    for i in 0..2 {
        assert!((as_slice[[i, 0, 0]] - 1.2649086).abs() < 0.001);
        assert!((as_slice[[i, 0, 1]] - 3.7947257).abs() < 0.001);
        assert!((as_slice[[i, 1, 0]] - 1.6970556).abs() < 0.001);
        assert!((as_slice[[i, 1, 1]] - 3.3941112).abs() < 0.001);
        assert!((as_slice[[i, 2, 0]] - 1.8107147).abs() < 0.001);
        assert!((as_slice[[i, 2, 1]] - 3.2592864).abs() < 0.001);
    }
}
