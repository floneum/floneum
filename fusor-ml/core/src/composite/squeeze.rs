use crate::{DataType, LastRank, LastRankInner, SmallerRank, SmallerRankInner, Tensor};

fn unchecked_squeeze<const R1: usize, const R2: usize, const DIFF: usize, D: DataType>(
    tensor: &Tensor<R1, D>,
    mut axis: [usize; DIFF],
) -> Tensor<R2, D> {
    const {
        assert!(R1 == R2 + DIFF);
    }
    let shape = tensor.shape();
    for &ax in &axis {
        assert!(ax < R1);
        assert_eq!(shape[ax], 1);
    }
    axis.sort_unstable();
    let mut iter = axis.into_iter().peekable();
    tensor.reshape(std::array::from_fn(|i| {
        _ = iter.next_if_eq(&i);
        let remaining = iter.len();
        let past = DIFF - remaining;
        shape[i + past]
    }))
}

impl<const R1: usize, D: DataType> Tensor<R1, D> {
    pub fn squeeze<const R2: usize>(&self, axis: usize) -> <Self as LastRankInner>::LastRank
    where
        Self: LastRank<R2, D>,
    {
        unchecked_squeeze(self, [axis])
    }
}

impl<const R1: usize, D: DataType> Tensor<R1, D> {
    pub fn squeeze_dims<const DIFF: usize, const R2: usize>(
        &self,
        axises: [usize; DIFF],
    ) -> <Self as SmallerRankInner<DIFF>>::SmallerRank
    where
        Self: SmallerRank<DIFF, R2, D>,
    {
        unchecked_squeeze(self, axises)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_squeeze_dim0() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[[1., 2.], [3., 4.], [5., 6.]]];
    let tensor = Tensor::new(&device, &data);
    let unsqueezed = tensor.squeeze(0);
    let as_slice = unsqueezed.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice[[1, 1]], 4.);
    assert_eq!(as_slice[[2, 0]], 5.);
    assert_eq!(as_slice[[2, 1]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_squeeze_dim2() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[[1.], [2.]], [[3.], [4.]], [[5.], [6.]]];
    let tensor = Tensor::new(&device, &data);
    let unsqueezed = tensor.squeeze(2);
    let as_slice = unsqueezed.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice[[1, 1]], 4.);
    assert_eq!(as_slice[[2, 0]], 5.);
    assert_eq!(as_slice[[2, 1]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_squeeze_dims() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[[[1.], [2.]], [[3.], [4.]], [[5.], [6.]]]];
    let tensor = Tensor::new(&device, &data);
    let unsqueezed = tensor.squeeze_dims([3, 0]);
    let as_slice = unsqueezed.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice[[1, 1]], 4.);
    assert_eq!(as_slice[[2, 0]], 5.);
    assert_eq!(as_slice[[2, 1]], 6.);
}
