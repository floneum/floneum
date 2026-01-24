//! Linear layer implementation.

use crate::{ConcreteTensor, Device, QMatrix, Tensor, SimdElement, VarBuilder};
use fusor_cpu::GgmlType;

/// A linear (fully connected) layer with quantized weights.
///
/// Computes `output = input @ weight.T + bias` using quantized matrix multiplication.
pub struct Linear<T: SimdElement> {
    weight: QMatrix<2>,
    bias: Option<Tensor<1, T>>,
}

impl<T: SimdElement> Linear<T> {
    /// Create a new Linear layer with the given quantized weight and optional bias.
    ///
    /// Weight shape: (out_features, in_features)
    /// Bias shape: (out_features,)
    pub fn new(weight: QMatrix<2>, bias: Option<Tensor<1, T>>) -> Self {
        Self { weight, bias }
    }

    /// Get the quantization type of the weights.
    pub fn quantization(&self) -> GgmlType {
        self.weight.ggml_type()
    }

    /// Get the bias tensor if present.
    pub fn bias(&self) -> Option<&Tensor<1, T>> {
        self.bias.as_ref()
    }

    /// Get the input features size.
    pub fn in_features(&self) -> usize {
        self.weight.shape()[1]
    }

    /// Get the output features size.
    pub fn out_features(&self) -> usize {
        self.weight.shape()[0]
    }
}

impl Linear<f32> {
    /// Load a Linear layer from a VarBuilder.
    ///
    /// Expects:
    /// - weight: Quantized tensor with shape (out_features, in_features)
    /// - bias (optional): Tensor with shape (out_features,)
    pub fn load(device: &Device, vb: &mut VarBuilder) -> crate::Result<Self> {
        let weight = vb.get("weight", device)?;
        let bias: Option<Tensor<1, f32>> = vb.get("bias", device).ok().map(|b| {
            let dequant: Tensor<2, f32> = b.dequantize();
            // The bias is stored as 2D in GGUF, squeeze to 1D
            let shape = dequant.shape();
            if shape[1] == 1 {
                dequant.squeeze(1).to_concrete()
            } else {
                dequant.squeeze(0).to_concrete()
            }
        });
        Ok(Self { weight, bias })
    }

    /// Forward pass for 3D input (batch, seq_len, in_features)
    ///
    /// Input shape: (batch, seq_len, in_features)
    /// Output shape: (batch, seq_len, out_features)
    pub fn forward(
        &self,
        input: &Tensor<3, f32, ConcreteTensor<f32, 3>>,
    ) -> Tensor<3, f32, ConcreteTensor<f32, 3>> {
        let output = input.q_mat_mul(&self.weight);

        if let Some(bias) = &self.bias {
            output.add_(bias)
        } else {
            output
        }
    }
}
