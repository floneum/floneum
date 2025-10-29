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

    #[tokio::test]
    async fn test_pool_1d_vs_candle() {
        use candle_core::{Device as CandleDevice, Tensor as CandleTensor};

        let device = Device::new().await.unwrap();
        let candle_device = CandleDevice::Cpu;

        // Test various configurations
        for (pool_size, stride) in [(2, 2), (3, 2), (4, 3)] {
            // Input: (1, 1, 12) - batch=1, channels=1, length=12
            let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 12.0, 3.0, 5.0, 39.0, 29.0, 1.0];
            let input_flat = input_data.clone();

            // Fusor expects (batch, channels, length) for 1D pooling with R=3, DIFF=1
            let input_nested = vec![vec![input_data]];
            let input_tensor = Tensor::new(&device, &input_nested);

            // Fusor pooling
            let output_tensor = input_tensor.pool_max([(pool_size, stride)]);
            let output_data = output_tensor.as_slice().await.unwrap();

            // Candle pooling - reshape to 2D and use max_pool2d
            // (1, 1, 12) -> (1, 1, 1, 12) for 2D pooling
            let candle_input = CandleTensor::from_slice(&input_flat, (1, 1, 1, 12), &candle_device).unwrap();

            // Use max_pool2d with kernel (1, pool_size) and stride (1, stride)
            let candle_output = candle_input.max_pool2d_with_stride((1, pool_size), (1, stride)).unwrap();
            // Reshape back from (1, 1, 1, out_len) to (1, 1, out_len)
            let candle_shape = candle_output.shape();
            let out_len = candle_shape.dims()[3];
            let candle_output = candle_output.reshape((1, 1, out_len)).unwrap();
            let candle_result = candle_output.to_vec3::<f32>().unwrap();

            // Compare results
            let fusor_shape = output_data.shape();
            assert_eq!(fusor_shape[0], 1);
            assert_eq!(fusor_shape[1], 1);
            assert_eq!(candle_result[0][0].len(), fusor_shape[2],
                "Output length mismatch for pool_size={}, stride={}: fusor={}, candle={}",
                pool_size, stride, fusor_shape[2], candle_result[0][0].len());

            for i in 0..fusor_shape[2] {
                let fusor_val = output_data[[0, 0, i]];
                let candle_val = candle_result[0][0][i];
                assert!(
                    (fusor_val - candle_val).abs() < 1e-5,
                    "Mismatch at position {} (pool_size={}, stride={}): fusor={}, candle={}",
                    i, pool_size, stride,
                    fusor_val,
                    candle_val
                );
            }
        }
    }

    #[tokio::test]
    async fn test_pool_1d_vs_candle_multi_channel() {
        use candle_core::{Device as CandleDevice, Tensor as CandleTensor};

        let device = Device::new().await.unwrap();
        let candle_device = CandleDevice::Cpu;

        let pool_size = 3;
        let stride = 2;

        // Input: (2, 4, 16) - batch=2, channels=4, length=16
        let mut input_data = vec![];
        let mut input_nested = vec![];
        for b in 0..2 {
            let mut batch = vec![];
            for c in 0..4 {
                let mut channel = vec![];
                for i in 0..16 {
                    let val = ((b * 64 + c * 16 + i) % 13) as f32;
                    input_data.push(val);
                    channel.push(val);
                }
                batch.push(channel);
            }
            input_nested.push(batch);
        }

        let input_tensor = Tensor::new(&device, &input_nested);

        // Fusor pooling
        let output_tensor = input_tensor.pool_max([(pool_size, stride)]);
        let output_data = output_tensor.as_slice().await.unwrap();

        // Candle pooling - reshape to 4D and use max_pool2d
        // (2, 4, 16) -> (2, 4, 1, 16)
        let candle_input = CandleTensor::from_slice(&input_data, (2, 4, 1, 16), &candle_device).unwrap();
        let candle_output = candle_input.max_pool2d_with_stride((1, pool_size), (1, stride)).unwrap();
        // Reshape from (2, 4, 1, out_len) to (2, 4, out_len)
        let candle_shape = candle_output.shape();
        let out_len = candle_shape.dims()[3];
        let candle_output = candle_output.reshape((2, 4, out_len)).unwrap();
        let candle_result = candle_output.to_vec3::<f32>().unwrap();

        // Compare results
        let fusor_shape = output_data.shape();
        assert_eq!(fusor_shape[0], 2);
        assert_eq!(fusor_shape[1], 4);
        assert_eq!(candle_result.len(), 2);
        assert_eq!(candle_result[0].len(), 4);

        for b in 0..2 {
            for c in 0..4 {
                assert_eq!(fusor_shape[2], candle_result[b][c].len(),
                    "Output length mismatch at batch {} channel {}", b, c);
                for i in 0..fusor_shape[2] {
                    let fusor_val = output_data[[b, c, i]];
                    let candle_val = candle_result[b][c][i];
                    assert!(
                        (fusor_val - candle_val).abs() < 1e-5,
                        "Mismatch at [{}, {}, {}]: fusor={}, candle={}",
                        b, c, i,
                        fusor_val,
                        candle_val
                    );
                }
            }
        }
    }
}
