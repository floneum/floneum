use crate::{Dim, FloatDataType, LastRank, LastRankInner, NextRankInner, Tensor};

impl<const N: usize, D: FloatDataType> Tensor<N, D> {
    pub fn var<const O: usize>(&self, dim: impl Dim<N>) -> Tensor<O, D>
    where
        Self: LastRank<O, D>,
        <Self as LastRankInner>::LastRank: NextRankInner<NextRank = Self>,
    {
        let mean = self.mean(dim);
        let diff = self.sub_(&mean.unsqueeze(dim));
        let sq_diff = &diff * &diff;
        sq_diff.mean(dim)
    }

    pub fn var_keepdim<const O: usize>(&self, dim: impl Dim<N>) -> Self
    where
        Self: LastRank<O, D>,
        <Self as LastRankInner>::LastRank: NextRankInner<NextRank = Self>,
    {
        self.var(dim).unsqueeze(dim)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_var() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    let a = Tensor::new(&device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let var0: Tensor<1, f32> = a.var(0);
    let var1: Tensor<1, f32> = a.var(1);

    assert_eq!(&var0.as_slice().await.unwrap(), &[2.25, 2.25, 2.25]);
    assert_eq!(&var1.as_slice().await.unwrap(), &[0.6666667, 0.6666667]);
}
