use crate::{DataType, Device, Result, Tensor, VarBuilder};

/// Configuration for Conv1d layer
#[derive(Debug, Clone, Copy)]
pub struct Conv1dConfig {
    pub padding: usize,
    pub stride: usize,
    pub groups: usize,
    pub dilation: usize,
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            groups: 1,
            dilation: 1,
        }
    }
}

/// 1D Convolution layer
///
/// Applies a 1D convolution over an input signal.
/// Input shape: (batch, in_channels, length)
/// Output shape: (batch, out_channels, out_length)
/// where out_length = (length + 2*padding - kernel_size) / stride + 1
pub struct Conv1d<T> {
    weight: Tensor<3, T>,       // (out_channels, in_channels, kernel_size)
    bias: Option<Tensor<1, T>>, // (out_channels,)
    config: Conv1dConfig,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
}

impl<T: DataType> Conv1d<T> {
    /// Create a new Conv1d layer with given weights and configuration
    pub fn new(weight: Tensor<3, T>, bias: Option<Tensor<1, T>>, config: Conv1dConfig) -> Self {
        let shape = weight.shape();
        let out_channels = shape[0];
        let in_channels = shape[1];
        let kernel_size = shape[2];

        // Validate configuration
        assert_eq!(config.groups, 1, "Only groups=1 is currently supported");
        assert_eq!(config.dilation, 1, "Only dilation=1 is currently supported");

        if let Some(ref b) = bias {
            assert_eq!(
                b.shape()[0],
                out_channels,
                "Bias shape must match out_channels"
            );
        }

        Self {
            weight,
            bias,
            config,
            in_channels,
            out_channels,
            kernel_size,
        }
    }

    /// Load Conv1d layer from VarBuilder
    ///
    /// Expects weight shape: (out_channels, in_channels, kernel_size)
    /// Expects bias shape: (out_channels,)
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: Conv1dConfig,
    ) -> Result<Self> {
        // Load and dequantize weight
        let weight = vb.get("weight", device)?.dequantize();

        // Load and dequantize bias
        let bias = vb.get("bias", device).ok().map(|b| b.dequantize());

        // Verify shapes
        let weight_shape = weight.shape();
        assert_eq!(
            weight_shape[0], out_channels,
            "Weight out_channels mismatch"
        );
        assert_eq!(weight_shape[1], in_channels, "Weight in_channels mismatch");
        assert_eq!(weight_shape[2], kernel_size, "Weight kernel_size mismatch");

        if let Some(ref b) = bias {
            assert_eq!(b.shape()[0], out_channels, "Bias shape mismatch");
        }

        Ok(Self::new(weight, bias, config))
    }

    /// Forward pass
    ///
    /// Input shape: (batch, in_channels, length)
    /// Output shape: (batch, out_channels, out_length)
    ///
    /// This is a special case of the generic conv_with_linear_channels method with DIFF=1
    /// (one spatial dimension).
    pub fn forward(&self, input: &Tensor<3, T>) -> Tensor<3, T> {
        // Conv1d is just multi-channel convolution with DIFF=1 (one spatial dimension)
        // Input: (batch, in_channels, length) - rank R=3
        // Weight: (out_channels, in_channels, kernel_size) - rank WEIGHT_RANK=3
        input.conv(
            &self.weight,
            self.bias.as_ref(),
            [self.config.padding],
            [self.config.stride],
        )
    }

    pub fn config(&self) -> &Conv1dConfig {
        &self.config
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_conv1d_simple() {
        let device = Device::test_instance();

        // Simple test: 1 input channel, 1 output channel, kernel size 3
        // Input: (1, 1, 5) - batch=1, in_channels=1, length=5
        let input_data = [[[1.0, 2.0, 3.0, 4.0, 5.0]]];
        let input = Tensor::new(&device, &input_data);

        // Weight: (1, 1, 3) - out_channels=1, in_channels=1, kernel_size=3
        let weight_data = [[[0.2, 0.5, 0.3]]];
        let weight = Tensor::new(&device, &weight_data);

        // Bias: (1,)
        let bias_data = [0.1];
        let bias = Tensor::new(&device, &bias_data);

        let config = Conv1dConfig {
            padding: 0,
            stride: 1,
            groups: 1,
            dilation: 1,
        };

        let conv = Conv1d::new(weight, Some(bias), config);
        let output = conv.forward(&input);

        // Output shape should be (1, 1, 3)
        assert_eq!(output.shape(), &[1, 1, 3]);

        let result = output.as_slice().await.unwrap();

        // Manual calculation:
        // output[0] = 1.0*0.2 + 2.0*0.5 + 3.0*0.3 + 0.1 = 0.2 + 1.0 + 0.9 + 0.1 = 2.2
        // output[1] = 2.0*0.2 + 3.0*0.5 + 4.0*0.3 + 0.1 = 0.4 + 1.5 + 1.2 + 0.1 = 3.2
        // output[2] = 3.0*0.2 + 4.0*0.5 + 5.0*0.3 + 0.1 = 0.6 + 2.0 + 1.5 + 0.1 = 4.2

        assert!((result[[0, 0, 0]] - 2.2).abs() < 1e-5);
        assert!((result[[0, 0, 1]] - 3.2).abs() < 1e-5);
        assert!((result[[0, 0, 2]] - 4.2).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_conv1d_with_padding() {
        let device = Device::test_instance();

        // Input: (1, 1, 3)
        let input_data = [[[1.0, 2.0, 3.0]]];
        let input = Tensor::new(&device, &input_data);

        // Weight: (1, 1, 3)
        let weight_data = [[[1.0, 1.0, 1.0]]];
        let weight = Tensor::new(&device, &weight_data);

        let config = Conv1dConfig {
            padding: 1,
            stride: 1,
            groups: 1,
            dilation: 1,
        };

        let conv = Conv1d::new(weight, None, config);
        let output = conv.forward(&input);

        // With padding=1, input becomes [0, 1, 2, 3, 0]
        // Output shape should be (1, 1, 3)
        assert_eq!(output.shape(), &[1, 1, 3]);

        let result = output.as_slice().await.unwrap();

        // Manual calculation:
        // output[0] = 0*1 + 1*1 + 2*1 = 3
        // output[1] = 1*1 + 2*1 + 3*1 = 6
        // output[2] = 2*1 + 3*1 + 0*1 = 5

        assert!((result[[0, 0, 0]] - 3.0).abs() < 1e-5);
        assert!((result[[0, 0, 1]] - 6.0).abs() < 1e-5);
        assert!((result[[0, 0, 2]] - 5.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_conv1d_multi_channel() {
        let device = Device::test_instance();

        // Input: (1, 2, 4) - 2 input channels
        let input_data = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]];
        let input = Tensor::new(&device, &input_data);

        // Weight: (3, 2, 2) - 3 output channels, 2 input channels, kernel size 2
        let weight_data = [
            [[1.0, 0.0], [0.0, 1.0]], // out_channel 0
            [[0.5, 0.5], [0.5, 0.5]], // out_channel 1
            [[1.0, 1.0], [1.0, 1.0]], // out_channel 2
        ];
        let weight = Tensor::new(&device, &weight_data);

        let config = Conv1dConfig::default();

        let conv = Conv1d::new(weight, None, config);
        let output = conv.forward(&input);

        // Output shape should be (1, 3, 3)
        assert_eq!(output.shape(), &[1, 3, 3]);

        let result = output.as_slice().await.unwrap();

        // Verify the mathematical convolution results
        // For position 0: in_ch0 window [1,2], in_ch1 window [5,6]
        //   out_ch 0 weights [[1,0], [0,1]]: 1*1 + 2*0 + 5*0 + 6*1 = 7
        //   out_ch 1 weights [[0.5,0.5], [0.5,0.5]]: 1*0.5 + 2*0.5 + 5*0.5 + 6*0.5 = 7
        //   out_ch 2 weights [[1,1], [1,1]]: 1*1 + 2*1 + 5*1 + 6*1 = 14
        // Similarly for positions 1 and 2

        // Out channel 0
        assert!((result[[0, 0, 0]] - 7.0).abs() < 1e-5);
        assert!((result[[0, 0, 1]] - 9.0).abs() < 1e-5);
        assert!((result[[0, 0, 2]] - 11.0).abs() < 1e-5);

        // Out channel 1
        assert!((result[[0, 1, 0]] - 7.0).abs() < 1e-5);
        assert!((result[[0, 1, 1]] - 9.0).abs() < 1e-5);
        assert!((result[[0, 1, 2]] - 11.0).abs() < 1e-5);

        // Out channel 2
        assert!((result[[0, 2, 0]] - 14.0).abs() < 1e-5);
        assert!((result[[0, 2, 1]] - 18.0).abs() < 1e-5);
        assert!((result[[0, 2, 2]] - 22.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_conv1d_vs_candle_simple() {
        use candle_core::{Device as CandleDevice, Tensor as CandleTensor};

        let device = Device::test_instance();
        let candle_device = CandleDevice::Cpu;

        // Input: (1, 1, 5) - batch=1, in_channels=1, length=5
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let input = Tensor::new(&device, &[[[1.0, 2.0, 3.0, 4.0, 5.0]]]);

        // Weight: (1, 1, 3) - out_channels=1, in_channels=1, kernel_size=3
        let weight_data = vec![0.2f32, 0.5, 0.3];
        let weight = Tensor::new(&device, &[[[0.2, 0.5, 0.3]]]);

        // Bias: (1,)
        let bias = Tensor::new(&device, &[0.1f32]);

        let config = Conv1dConfig {
            padding: 0,
            stride: 1,
            groups: 1,
            dilation: 1,
        };

        // Fusor convolution
        let conv = Conv1d::new(weight, Some(bias), config);
        let fusor_output = conv.forward(&input);
        let fusor_result = fusor_output.as_slice().await.unwrap();

        // Candle convolution
        let candle_input =
            CandleTensor::from_slice(&input_data, (1, 1, 5), &candle_device).unwrap();
        let candle_weight =
            CandleTensor::from_slice(&weight_data, (1, 1, 3), &candle_device).unwrap();
        let candle_bias = CandleTensor::from_slice(&[0.1f32], 1, &candle_device).unwrap();

        let candle_output = candle_input.conv1d(&candle_weight, 0, 1, 1, 1).unwrap();
        let candle_output = candle_output
            .broadcast_add(&candle_bias.reshape((1, 1, 1)).unwrap())
            .unwrap();
        let candle_result = candle_output.to_vec3::<f32>().unwrap();

        // Compare results
        assert_eq!(fusor_result.shape(), &[1, 1, 3]);
        for i in 0..3 {
            let fusor_val = fusor_result[[0, 0, i]];
            let candle_val = candle_result[0][0][i];
            assert!(
                (fusor_val - candle_val).abs() < 1e-4,
                "Mismatch at position {}: fusor={}, candle={}",
                i,
                fusor_val,
                candle_val
            );
        }
    }

    #[tokio::test]
    async fn test_conv1d_vs_candle_with_padding() {
        use candle_core::{Device as CandleDevice, Tensor as CandleTensor};

        let device = Device::test_instance();
        let candle_device = CandleDevice::Cpu;

        // Input: (1, 1, 5)
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let input = Tensor::new(&device, &[[[1.0, 2.0, 3.0, 4.0, 5.0]]]);

        // Weight: (1, 1, 3)
        let weight_data = vec![0.5f32, 1.0, 0.5];
        let weight = Tensor::new(&device, &[[[0.5, 1.0, 0.5]]]);

        let config = Conv1dConfig {
            padding: 1,
            stride: 1,
            groups: 1,
            dilation: 1,
        };

        // Fusor convolution
        let conv = Conv1d::new(weight, None, config);
        let fusor_output = conv.forward(&input);
        let fusor_result = fusor_output.as_slice().await.unwrap();

        // Candle convolution
        let candle_input =
            CandleTensor::from_slice(&input_data, (1, 1, 5), &candle_device).unwrap();
        let candle_weight =
            CandleTensor::from_slice(&weight_data, (1, 1, 3), &candle_device).unwrap();
        let candle_output = candle_input.conv1d(&candle_weight, 1, 1, 1, 1).unwrap();
        let candle_result = candle_output.to_vec3::<f32>().unwrap();

        // Compare results
        assert_eq!(fusor_result.shape(), &[1, 1, 5]);
        assert_eq!(candle_result[0][0].len(), 5);
        for i in 0..5 {
            let fusor_val = fusor_result[[0, 0, i]];
            let candle_val = candle_result[0][0][i];
            assert!(
                (fusor_val - candle_val).abs() < 1e-4,
                "Mismatch at position {}: fusor={}, candle={}",
                i,
                fusor_val,
                candle_val
            );
        }
    }

    #[tokio::test]
    async fn test_conv1d_vs_candle_multi_channel() {
        use candle_core::{Device as CandleDevice, Tensor as CandleTensor};

        let device = Device::test_instance();
        let candle_device = CandleDevice::Cpu;

        // Input: (2, 3, 10) - batch=2, in_channels=3, length=10
        let mut input_data = vec![];
        let mut input_nested = vec![];
        for b in 0..2 {
            let mut batch = vec![];
            for c in 0..3 {
                let mut channel = vec![];
                for i in 0..10 {
                    let val = (b * 30 + c * 10 + i + 1) as f32 * 0.1;
                    input_data.push(val);
                    channel.push(val);
                }
                batch.push(channel);
            }
            input_nested.push(batch);
        }
        let input = Tensor::new(&device, &input_nested);

        // Weight: (4, 3, 3) - out_channels=4, in_channels=3, kernel_size=3
        let mut weight_data = vec![];
        let mut weight_nested = vec![];
        for o in 0..4 {
            let mut out_ch = vec![];
            for i in 0..3 {
                let mut in_ch = vec![];
                for k in 0..3 {
                    let val = ((o * 9 + i * 3 + k) % 7) as f32 * 0.2;
                    weight_data.push(val);
                    in_ch.push(val);
                }
                out_ch.push(in_ch);
            }
            weight_nested.push(out_ch);
        }
        let weight = Tensor::new(&device, &weight_nested);

        let config = Conv1dConfig {
            padding: 1,
            stride: 2,
            groups: 1,
            dilation: 1,
        };

        // Fusor convolution
        let conv = Conv1d::new(weight, None, config);
        let fusor_output = conv.forward(&input);
        let fusor_result = fusor_output.as_slice().await.unwrap();

        // Candle convolution
        let candle_input =
            CandleTensor::from_slice(&input_data, (2, 3, 10), &candle_device).unwrap();
        let candle_weight =
            CandleTensor::from_slice(&weight_data, (4, 3, 3), &candle_device).unwrap();
        let candle_output = candle_input.conv1d(&candle_weight, 1, 2, 1, 1).unwrap();
        let candle_result = candle_output.to_vec3::<f32>().unwrap();

        // Compare results
        let fusor_shape = fusor_result.shape();
        assert_eq!(fusor_shape[0], 2); // batch
        assert_eq!(fusor_shape[1], 4); // out_channels
        assert_eq!(candle_result.len(), 2);
        assert_eq!(candle_result[0].len(), 4);

        for b in 0..2 {
            for c in 0..4 {
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
