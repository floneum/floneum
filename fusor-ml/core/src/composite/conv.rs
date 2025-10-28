use crate::{DataType, LargerRank, LastRank, MaxRank, NextRank, SmallerRank, Tensor};

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn conv<const DIFF: usize, const R2: usize, const R3: usize, const O: usize>(
        &self,
        weight: &Tensor<DIFF, D>,
        bias: Option<D>,
        strides: [usize; DIFF],
    ) -> Self
    where
        Self: LargerRank<DIFF, R2, D>,
        Tensor<R2, D>: NextRank<R3, D>,
        Tensor<R3, D>: SmallerRank<DIFF, O, D>,
        (Tensor<O, D>, Tensor<1, D>): MaxRank<O, D>,
        Tensor<O, D>: LastRank<R, D>,
    {
        let window = weight.shape();
        let axis_start = R - DIFF;
        let tiled = self.sliding_window_view(std::array::from_fn(|i| {
            [axis_start + i, window[i], strides[i]]
        }));

        let flattened_weights = weight.flatten_all();
        let flattened = tiled.unsqueeze(R2).flatten_last_n::<DIFF, _>();
        let product = flattened.mul_(&flattened_weights);

        let out = product.sum(O - 1);
        if let Some(bias) = bias {
            out + bias
        } else {
            out
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

        let input_data = &[1.0f32, 2.0, 3.0, 4.0, 5.0];
        let input_tensor = Tensor::new(&device, input_data);

        let weight_data = &[0.2f32, 0.5, 0.3];
        let weight_tensor = Tensor::new(&device, weight_data);

        let bias = Some(0.1f32);

        // Perform convolution with stride 1
        let output_tensor = input_tensor.conv(&weight_tensor, bias, [1]);

        let expected = input_data
            .windows(weight_data.len())
            .map(|window| {
                window
                    .iter()
                    .zip(weight_data.iter())
                    .map(|(x, w)| x * w)
                    .sum::<f32>()
                    + bias.unwrap_or_default()
            })
            .collect::<Vec<f32>>();

        let output_data = output_tensor.as_slice().await.unwrap();
        assert_eq!(output_data.shape()[0], expected.len());
        for i in 0..expected.len() {
            let val = output_data[[i]];
            let expected = expected[i];
            assert!(
                (val - expected).abs() < 1e-6,
                "Mismatch at index {}: got {}, expected {}",
                i,
                val,
                expected
            );
        }
    }

    #[tokio::test]
    async fn test_conv_1d_strided() {
        let device = Device::new().await.unwrap();

        let input_data = &[1.0f32, 2.0, 3.0, 4.0, 5.0];
        let input_tensor = Tensor::new(&device, input_data);

        let weight_data = &[0.2f32, 0.5, 0.3];
        let weight_tensor = Tensor::new(&device, weight_data);

        let bias = Some(0.1f32);
        let stride = 2;
        let output_tensor = input_tensor.conv(&weight_tensor, bias, [stride]);

        let expected = input_data
            .windows(weight_data.len())
            .step_by(stride)
            .map(|window| {
                window
                    .iter()
                    .zip(weight_data.iter())
                    .map(|(x, w)| x * w)
                    .sum::<f32>()
                    + bias.unwrap_or_default()
            })
            .collect::<Vec<f32>>();

        let output_data = output_tensor.as_slice().await.unwrap();
        assert_eq!(output_data.shape()[0], expected.len());
        for i in 0..expected.len() {
            let val = output_data[[i]];
            let expected = expected[i];
            assert!(
                (val - expected).abs() < 1e-6,
                "Mismatch at index {}: got {}, expected {}",
                i,
                val,
                expected
            );
        }
    }
}
