use crate::{DataType, Dim, LargerRank, LargerRankInner, NextRank, NextRankInner, Tensor};

fn unchecked_unsqueeze<const R1: usize, const R2: usize, const DIFF: usize, D: DataType>(
    tensor: &Tensor<R1, D>,
    mut axis: [usize; DIFF],
) -> Tensor<R2, D> {
    const {
        assert!(R1 + DIFF == R2);
    }
    let shape = tensor.shape();
    for &ax in &axis {
        assert!(ax < R2);
    }
    axis.sort_unstable();
    let mut iter = axis.into_iter().peekable();
    tensor.reshape(std::array::from_fn(|i| {
        if iter.next_if(|o| o <= &i).is_some() {
            return 1;
        }
        let remaining = iter.len();
        let past = DIFF - remaining;
        shape[i - past]
    }))
}

impl<const R1: usize, D: DataType> Tensor<R1, D> {
    pub fn unsqueeze<const R2: usize>(&self, axis: impl Dim<R2>) -> <Self as NextRankInner>::NextRank
    where
        Self: NextRank<R2, D>,
    {
        unchecked_unsqueeze(self, [axis.resolve()])
    }
}

impl<const R1: usize, D: DataType> Tensor<R1, D> {
    pub fn unsqueeze_dims<const DIFF: usize, const R2: usize>(
        &self,
        axises: [usize; DIFF],
    ) -> <Self as LargerRankInner<DIFF>>::LargerRank
    where
        Self: LargerRank<DIFF, R2, D>,
    {
        unchecked_unsqueeze(self, axises)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_unsqueeze() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let unsqueezed = tensor.unsqueeze(0);
    let as_slice = unsqueezed.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0, 0, 0]], 1.);
    assert_eq!(as_slice[[0, 0, 1]], 2.);
    assert_eq!(as_slice[[0, 1, 0]], 3.);
    assert_eq!(as_slice[[0, 1, 1]], 4.);
    assert_eq!(as_slice[[0, 2, 0]], 5.);
    assert_eq!(as_slice[[0, 2, 1]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_unsqueeze_dims() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let unsqueezed = tensor.unsqueeze_dims([2, 0]);
    println!("{unsqueezed:?}");
    let as_slice = unsqueezed.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0, 0, 0, 0]], 1.);
    assert_eq!(as_slice[[0, 0, 0, 1]], 2.);
    assert_eq!(as_slice[[0, 1, 0, 0]], 3.);
    assert_eq!(as_slice[[0, 1, 0, 1]], 4.);
    assert_eq!(as_slice[[0, 2, 0, 0]], 5.);
    assert_eq!(as_slice[[0, 2, 0, 1]], 6.);
}
