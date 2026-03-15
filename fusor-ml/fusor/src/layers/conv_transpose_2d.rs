//! 2D Transposed Convolution layer implementation.

use crate::{ConcreteTensor, Device, Tensor, VarBuilder};

/// 2D Transposed Convolution layer
///
/// Applies a 2D transposed convolution (sometimes called "deconvolution").
/// Input shape: (batch, in_channels, height, width)
/// Output shape: (batch, out_channels, height * stride[0], width * stride[1])
///
/// Note: This implementation is optimized for the case where kernel_size == stride
/// (commonly used for learned upsampling), using a matmul + pixel-shuffle approach.
/// Weight shape: (in_channels, out_channels, kernel_h, kernel_w)
pub struct ConvTranspose2d {
    weight: Tensor<4, f32, ConcreteTensor<f32, 4>>,
    bias: Option<Tensor<1, f32, ConcreteTensor<f32, 1>>>,
    stride: [usize; 2],
}

impl ConvTranspose2d {
    /// Create a new ConvTranspose2d layer
    pub fn new(
        weight: Tensor<4, f32, ConcreteTensor<f32, 4>>,
        bias: Option<Tensor<1, f32, ConcreteTensor<f32, 1>>>,
        stride: [usize; 2],
    ) -> Self {
        Self {
            weight,
            bias,
            stride,
        }
    }

    /// Forward pass using matmul + pixel-shuffle pattern.
    ///
    /// For stride==kernel (the common SAM case), each output pixel depends on exactly
    /// one input pixel, making this a simple matmul + reshape.
    pub fn forward(
        &self,
        input: &Tensor<4, f32, ConcreteTensor<f32, 4>>,
    ) -> Tensor<4, f32, ConcreteTensor<f32, 4>> {
        let shape = input.shape();
        let b = shape[0];
        let in_ch = shape[1];
        let h = shape[2];
        let w = shape[3];
        let weight_shape = self.weight.shape();
        let out_ch = weight_shape[1];
        let kh = weight_shape[2];
        let kw = weight_shape[3];
        let out_h = h * self.stride[0];
        let out_w = w * self.stride[1];

        // Reshape input to (b, in_ch, h*w) then transpose to (b, h*w, in_ch)
        // then flatten batch: (b*h*w, in_ch)
        let input_flat: Tensor<2, f32, ConcreteTensor<f32, 2>> = input
            .reshape([b, in_ch, h * w])
            .transpose(1, 2)
            .to_concrete()
            .reshape([b * h * w, in_ch])
            .to_concrete();

        // Reshape weight from (in_ch, out_ch, kh, kw) to (in_ch, out_ch * kh * kw)
        let weight_flat: Tensor<2, f32, ConcreteTensor<f32, 2>> =
            self.weight.reshape([in_ch, out_ch * kh * kw]).to_concrete();

        // Matmul: (b*h*w, in_ch) @ (in_ch, out_ch*kh*kw) -> (b*h*w, out_ch*kh*kw)
        let result = input_flat.mat_mul(&weight_flat);

        // Reshape to (b, h, w, out_ch, kh, kw)
        let result: Tensor<6, f32, ConcreteTensor<f32, 6>> =
            result.reshape([b, h, w, out_ch, kh, kw]).to_concrete();

        // Permute to (b, out_ch, h, kh, w, kw) then reshape to (b, out_ch, h*kh, w*kw)
        let result: Tensor<4, f32, ConcreteTensor<f32, 4>> = result
            .transpose(2, 3) // (b, h, out_ch, w, kh, kw)
            .to_concrete()
            .transpose(1, 2) // (b, out_ch, h, w, kh, kw)
            .to_concrete()
            .transpose(3, 4) // (b, out_ch, h, kh, w, kw)
            .to_concrete()
            .reshape([b, out_ch, out_h, out_w])
            .to_concrete();

        if let Some(bias) = &self.bias {
            // Reshape bias from [out_ch] to [1, out_ch, 1, 1] for correct channel-dim broadcasting
            let bias_4d: Tensor<4, f32, ConcreteTensor<f32, 4>> = bias
                .reshape([1, out_ch, 1, 1])
                .broadcast_as([b, out_ch, out_h, out_w])
                .to_concrete();
            (result + bias_4d).to_concrete()
        } else {
            result
        }
    }

    /// Load ConvTranspose2d layer from VarBuilder
    pub fn load(device: &Device, vb: &mut VarBuilder, stride: [usize; 2]) -> crate::Result<Self> {
        let weight: Tensor<4, f32> = vb.get("weight", device)?.dequantize();
        let bias: Option<Tensor<1, f32, ConcreteTensor<f32, 1>>> =
            vb.get("bias", device).ok().map(|b| b.dequantize());
        Ok(Self::new(weight.to_concrete(), bias, stride))
    }
}
