use crate::{DataType, LargerRank, LastRank, NextRank, SmallerRank, Tensor};

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn pool<const DIFF: usize, const R2: usize, const R3: usize, const O: usize>(
        &self,
        pools: [impl Into<PoolSize>; DIFF],
        with: fn(&Tensor<O, D>, usize) -> Tensor<R, D>,
    ) -> Self
    where
        Self: LargerRank<DIFF, R2, D>,
        Tensor<R2, D>: NextRank<R3, D>,
        Tensor<R3, D>: SmallerRank<DIFF, O, D>,
        Tensor<O, D>: LastRank<R, D>,
    {
        let pools = pools.map(|p| p.into());

        let axis_start = R - DIFF;
        let tiled = self.sliding_window_view(std::array::from_fn(|i| {
            let window = pools[i].size;
            let stride = pools[i].stride;
            [axis_start + i, window, stride]
        }));

        let flattened = tiled.unsqueeze(R2).flatten_last_n::<DIFF, _>();

        with(&flattened, O - 1)
    }

    pub fn pool_max<const DIFF: usize, const R2: usize, const R3: usize, const O: usize>(
        &self,
        pools: [impl Into<PoolSize>; DIFF],
    ) -> Self
    where
        Self: LargerRank<DIFF, R2, D>,
        Tensor<R2, D>: NextRank<R3, D>,
        Tensor<R3, D>: SmallerRank<DIFF, O, D>,
        Tensor<O, D>: LastRank<R, D>,
    {
        self.pool(pools, Tensor::max)
    }

    pub fn pool_min<const DIFF: usize, const R2: usize, const R3: usize, const O: usize>(
        &self,
        pools: [impl Into<PoolSize>; DIFF],
    ) -> Self
    where
        Self: LargerRank<DIFF, R2, D>,
        Tensor<R2, D>: NextRank<R3, D>,
        Tensor<R3, D>: SmallerRank<DIFF, O, D>,
        Tensor<O, D>: LastRank<R, D>,
    {
        self.pool(pools, Tensor::min)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PoolSize {
    size: usize,
    stride: usize,
}

impl From<usize> for PoolSize {
    fn from(size: usize) -> Self {
        Self { size, stride: size }
    }
}

impl From<(usize, usize)> for PoolSize {
    fn from((size, stride): (usize, usize)) -> Self {
        Self { size, stride }
    }
}

impl From<[usize; 2]> for PoolSize {
    fn from([size, stride]: [usize; 2]) -> Self {
        Self { size, stride }
    }
}

impl PoolSize {
    pub fn new(size: usize, stride: usize) -> Self {
        Self { size, stride }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[tokio::test]
    async fn test_pool_1d() {
        let device = Device::new().await.unwrap();

        for (pool_size, stride) in [(2, 2), (3, 1), (4, 2)] {
            let input_data = &[1, 2, 3, 4, 5, 3, 12, 3, 5, 39, 29, 1];
            let input_tensor = Tensor::new(&device, input_data);

            let output_tensor = input_tensor.pool_max([(pool_size, stride)]);

            let expected = input_data
                .windows(pool_size)
                .step_by(stride)
                .map(|window| window.iter().max().unwrap())
                .copied()
                .collect::<Vec<u32>>();

            let output_data = output_tensor.as_slice().await.unwrap();
            assert_eq!(output_data.shape()[0], expected.len());
            for i in 0..expected.len() {
                let val = output_data[[i]];
                let expected = expected[i];
                assert_eq!(
                    val, expected,
                    "Mismatch at index {}: got {}, expected {}",
                    i, val, expected
                );
            }
        }
    }
}
