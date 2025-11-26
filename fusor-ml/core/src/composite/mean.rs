use crate::{Dim, FloatDataType, LastRank, LastRankInner, NextRankInner, Tensor};

impl<const N: usize, D: FloatDataType> Tensor<N, D> {
    pub fn mean<const O: usize>(&self, dim: impl Dim<N>) -> Tensor<O, D>
    where
        Self: LastRank<O, D>,
    {
        let dim = dim.resolve();
        let sum = self.sum(dim);
        let dim_size = self.shape()[dim] as f32;
        sum / D::from_f32(dim_size)
    }

    pub fn mean_keepdim<const O: usize>(&self, dim: impl Dim<N>) -> Self
    where
        Self: LastRank<O, D>,
        <Self as LastRankInner>::LastRank: NextRankInner<NextRank = Self>,
    {
        self.mean(dim).unsqueeze(dim)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_mean() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    let a = Tensor::new(&device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let mean0: Tensor<1, f32> = a.mean(0);
    let mean1: Tensor<1, f32> = a.mean(1);

    assert_eq!(&mean0.as_slice().await.unwrap(), &[2.5, 3.5, 4.5]);
    assert_eq!(&mean1.as_slice().await.unwrap(), &[2.0, 5.0]);
}
