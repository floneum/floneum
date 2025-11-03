use crate::{DataType, LargerRank, Tensor};

impl<const R: usize, D: DataType> Tensor<R, D> {
    /// Pad a specific axis with zeros on both sides
    fn pad_axis(&self, axis: usize, padding: usize) -> Self {
        if padding == 0 {
            return self.clone();
        }

        let shape = self.shape();
        let device = self.device();

        // Create left padding shape
        let mut pad_shape = *shape;
        pad_shape[axis] = padding;
        let pad_left = Tensor::zeros(device, pad_shape);
        let pad_right = Tensor::zeros(device, pad_shape);

        // Concatenate: [pad_left, self, pad_right] along the axis
        Tensor::cat([pad_left, self.clone(), pad_right], axis)
    }

    /// Unified convolution method that handles different tensor formats:
    /// - Simple convolution (R = DIFF): element-wise convolution without channels
    /// - Multi-channel convolution (R = 2 + DIFF): (batch, channels, ...spatial) format
    ///
    /// For Conv1d: R=3, DIFF=1 gives (batch, in_channels, length) -> (batch, out_channels, out_length)
    /// For simple 1D conv: R=1, DIFF=1 gives (length) -> (out_length)
    pub fn conv<const WEIGHT_RANK: usize, const DIFF: usize, const R2: usize>(
        &self,
        weight: &Tensor<WEIGHT_RANK, D>,
        bias: Option<&Tensor<1, D>>,
        padding: [usize; DIFF],
        strides: [usize; DIFF],
    ) -> Self
    where
        Self: LargerRank<DIFF, R2, D>,
    {
        // Extract dimensions
        let input_shape = self.shape();
        let weight_shape = weight.shape();
        let spatial_start = R - DIFF;

        // Multi-channel convolution: (batch, channels, ...spatial)
        // Note: This implementation expects R = 2 + DIFF format
        assert_eq!(
            R,
            2 + DIFF,
            "Conv expects (batch, channels, ...spatial) format where R = 2 + DIFF"
        );
        let batch_axis = 0;
        let in_channels_axis = 1;

        let batch = input_shape[batch_axis];
        let in_channels = input_shape[in_channels_axis];
        let out_channels = weight_shape[0];

        // Weight shape is (out_channels, in_channels, ...kernel_dims)
        assert_eq!(
            weight_shape[1], in_channels,
            "Weight in_channels must match input in_channels"
        );

        // Step 1: Apply padding to the spatial dimensions (last DIFF dimensions)
        let padded = if padding.iter().any(|&p| p > 0) {
            let mut result = self.clone();
            for i in 0..DIFF {
                let axis = R - DIFF + i;
                if padding[i] > 0 {
                    result = result.pad_axis(axis, padding[i]);
                }
            }
            result
        } else {
            self.clone()
        };

        // Calculate output spatial dimensions
        let mut out_spatial_size = 1;
        for i in 0..DIFF {
            let padded_len = input_shape[spatial_start + i] + 2 * padding[i];
            let kernel_len = weight_shape[spatial_start + i];
            let out_len = (padded_len - kernel_len) / strides[i] + 1;
            out_spatial_size *= out_len;
        }

        // Step 2: Create sliding windows over the spatial dimensions
        // This gives us shape: (batch, in_channels, ...out_spatial..., ...kernel...)
        let windows = padded.sliding_window_view(std::array::from_fn(|i| {
            let axis = R - DIFF + i;
            let kernel_size = weight_shape[spatial_start + i];
            [axis, kernel_size, strides[i]]
        }));

        // Step 3: Prepare for matmul by reshaping and transposing
        // Windows: (batch, in_channels, ...out_spatial..., ...kernel...)
        // We need: (batch, ...out_spatial..., in_channels, ...kernel...)
        // Then flatten to: (batch * out_spatial_size, in_channels * kernel_size)

        // First, calculate kernel size
        let kernel_size: usize = weight_shape[spatial_start..].iter().product();

        // Transpose to move in_channels after spatial:
        // (batch, in_channels, ...out_spatial..., ...kernel...) -> (batch, ...out_spatial..., in_channels, ...kernel...)
        let windows_transposed = windows.transpose(in_channels_axis, spatial_start);

        // Flatten to (batch * out_spatial_size, in_channels * kernel_size)
        let windows_flat =
            windows_transposed.reshape([batch * out_spatial_size, in_channels * kernel_size]);

        // Step 4: Reshape weight for matmul
        // Weight: (out_channels, in_channels, ...kernel...) -> (out_channels, in_channels * kernel_size)
        let weight_reshaped = weight.reshape([out_channels, in_channels * kernel_size]);
        // Transpose for matmul: (in_channels * kernel_size, out_channels)
        let weight_t = weight_reshaped.t();

        // Step 5: Matrix multiplication
        // (batch * out_spatial_size, in_channels * kernel_size) @ (in_channels * kernel_size, out_channels)
        // = (batch * out_spatial_size, out_channels)
        let output = windows_flat.mat_mul(&weight_t);

        // Step 6: Reshape and transpose back to (batch, out_channels, ...out_spatial...)
        // First reshape to (batch, out_spatial_size, out_channels)
        let output_reshaped = output.reshape([batch, out_spatial_size, out_channels]);
        // Transpose to (batch, out_channels, out_spatial_size)
        let output_transposed = output_reshaped.transpose(in_channels_axis, spatial_start);

        // Reshape to (batch, out_channels, ...out_spatial_dims...)
        let mut output_shape = *input_shape;
        output_shape[in_channels_axis] = out_channels;
        for i in 0..DIFF {
            let padded_len = input_shape[spatial_start + i] + 2 * padding[i];
            let kernel_len = weight_shape[spatial_start + i];
            output_shape[spatial_start + i] = (padded_len - kernel_len) / strides[i] + 1;
        }
        let output_final = output_transposed.reshape(output_shape);

        // Step 7: Add bias if present
        if let Some(bias) = bias {
            // Bias shape: (out_channels,)
            // Need to broadcast to (batch, out_channels, ...spatial...)
            // Reshape to (1, out_channels, 1, 1, ...) for broadcasting
            let mut bias_shape = [1; R];
            bias_shape[in_channels_axis] = out_channels;
            let bias_reshaped = bias.unsqueeze(0).reshape(bias_shape);
            output_final.add_(&bias_reshaped)
        } else {
            output_final
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[tokio::test]
    async fn test_conv_1d() {
        let device = Device::new().await.unwrap();

        // Input: (batch=1, in_channels=1, length=5)
        let input_data = [[[1.0f32, 2.0, 3.0, 4.0, 5.0]]];
        let input_tensor = Tensor::new(&device, &input_data);

        // Weight: (out_channels=1, in_channels=1, kernel_size=3)
        let weight_data = [[[0.2f32, 0.5, 0.3]]];
        let weight_tensor = Tensor::new(&device, &weight_data);

        let bias_val = 0.1f32;
        let bias = Some(Tensor::splat(&device, bias_val, [1]));

        // Perform convolution with stride 1 and no padding
        // Input: (batch=1, in_channels=1, length=5), Weight: (out_channels=1, in_channels=1, kernel_size=3)
        // For R=3, DIFF=1: R2=4 (after sliding window)
        let output_tensor = input_tensor.conv(&weight_tensor, bias.as_ref(), [0], [1]);

        // Expected values for the 1D convolution
        let input_flat = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let weight_flat = [0.2f32, 0.5, 0.3];
        let expected = input_flat
            .windows(weight_flat.len())
            .map(|window| {
                window
                    .iter()
                    .zip(weight_flat.iter())
                    .map(|(x, w)| x * w)
                    .sum::<f32>()
                    + bias_val
            })
            .collect::<Vec<f32>>();

        let output_data = output_tensor.as_slice().await.unwrap();
        assert_eq!(output_data.shape(), &[1, 1, expected.len()]);
        for i in 0..expected.len() {
            let val = output_data[[0, 0, i]];
            let expected_val = expected[i];
            assert!(
                (val - expected_val).abs() < 1e-6,
                "Mismatch at index {}: got {}, expected {}",
                i,
                val,
                expected_val
            );
        }
    }

    #[tokio::test]
    async fn test_conv_1d_strided() {
        let device = Device::new().await.unwrap();

        // Input: (batch=1, in_channels=1, length=5)
        let input_data = [[[1.0f32, 2.0, 3.0, 4.0, 5.0]]];
        let input_tensor = Tensor::new(&device, &input_data);

        // Weight: (out_channels=1, in_channels=1, kernel_size=3)
        let weight_data = [[[0.2f32, 0.5, 0.3]]];
        let weight_tensor = Tensor::new(&device, &weight_data);

        let bias_val = 0.1f32;
        let bias_tensor = Some(Tensor::splat(&device, bias_val, [1]));
        let stride = 2;
        // Input: (batch=1, in_channels=1, length=5), Weight: (out_channels=1, in_channels=1, kernel_size=3)
        // For R=3, DIFF=1: R2=4 (after sliding window)
        let output_tensor = input_tensor.conv(&weight_tensor, bias_tensor.as_ref(), [0], [stride]);

        // Expected values for the strided 1D convolution
        let input_flat = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let weight_flat = [0.2f32, 0.5, 0.3];
        let expected = input_flat
            .windows(weight_flat.len())
            .step_by(stride)
            .map(|window| {
                window
                    .iter()
                    .zip(weight_flat.iter())
                    .map(|(x, w)| x * w)
                    .sum::<f32>()
                    + bias_val
            })
            .collect::<Vec<f32>>();

        let output_data = output_tensor.as_slice().await.unwrap();
        assert_eq!(output_data.shape(), &[1, 1, expected.len()]);
        for i in 0..expected.len() {
            let val = output_data[[0, 0, i]];
            let expected_val = expected[i];
            assert!(
                (val - expected_val).abs() < 1e-6,
                "Mismatch at index {}: got {}, expected {}",
                i,
                val,
                expected_val
            );
        }
    }

    #[tokio::test]
    async fn test_conv_1d_vs_candle() {
        use candle_core::{Device as CandleDevice, Tensor as CandleTensor};

        let device = Device::new().await.unwrap();
        let candle_device = CandleDevice::Cpu;

        // Input: (2, 3, 8) - batch=2, in_channels=3, length=8
        let mut input_data = vec![];
        let mut input_nested = vec![];
        for b in 0..2 {
            let mut batch = vec![];
            for c in 0..3 {
                let mut channel = vec![];
                for i in 0..8 {
                    let val = (b * 24 + c * 8 + i + 1) as f32 * 0.15;
                    input_data.push(val);
                    channel.push(val);
                }
                batch.push(channel);
            }
            input_nested.push(batch);
        }
        let input = Tensor::new(&device, &input_nested);

        // Weight: (5, 3, 4) - out_channels=5, in_channels=3, kernel_size=4
        let mut weight_data = vec![];
        let mut weight_nested = vec![];
        for o in 0..5 {
            let mut out_ch = vec![];
            for i in 0..3 {
                let mut in_ch = vec![];
                for k in 0..4 {
                    let val = ((o * 12 + i * 4 + k) % 11) as f32 * 0.1;
                    weight_data.push(val);
                    in_ch.push(val);
                }
                out_ch.push(in_ch);
            }
            weight_nested.push(out_ch);
        }
        let weight = Tensor::new(&device, &weight_nested);

        // Bias: (5,)
        let bias_data: Vec<f32> = (0..5).map(|i| i as f32 * 0.05).collect();
        let bias = Tensor::new(&device, &bias_data);

        // Fusor convolution with padding and stride
        let fusor_output = input.conv(&weight, Some(&bias), [1], [2]);
        let fusor_result = fusor_output.as_slice().await.unwrap();

        // Candle convolution
        let candle_input =
            CandleTensor::from_slice(&input_data, (2, 3, 8), &candle_device).unwrap();
        let candle_weight =
            CandleTensor::from_slice(&weight_data, (5, 3, 4), &candle_device).unwrap();
        let candle_bias = CandleTensor::from_slice(&bias_data, 5, &candle_device).unwrap();

        let candle_output = candle_input.conv1d(&candle_weight, 1, 2, 1, 1).unwrap();
        let candle_output = candle_output
            .broadcast_add(&candle_bias.reshape((1, 5, 1)).unwrap())
            .unwrap();
        let candle_result = candle_output.to_vec3::<f32>().unwrap();

        // Compare results
        let fusor_shape = fusor_result.shape();
        assert_eq!(fusor_shape[0], 2);
        assert_eq!(fusor_shape[1], 5);
        assert_eq!(candle_result.len(), 2);
        assert_eq!(candle_result[0].len(), 5);

        for b in 0..2 {
            for c in 0..5 {
                assert_eq!(
                    fusor_shape[2],
                    candle_result[b][c].len(),
                    "Output length mismatch at batch {} channel {}",
                    b,
                    c
                );
                for i in 0..fusor_shape[2] {
                    let fusor_val = fusor_result[[b, c, i]];
                    let candle_val = candle_result[b][c][i];
                    assert!(
                        (fusor_val - candle_val).abs() < 1e-3,
                        "Mismatch at [{}, {}, {}]: fusor={}, candle={}",
                        b,
                        c,
                        i,
                        fusor_val,
                        candle_val
                    );
                }
            }
        }
    }
}
