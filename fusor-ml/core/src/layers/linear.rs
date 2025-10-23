use fusor_gguf::GgmlType;

use crate::{Device, QMatrix, Result, Tensor, VarBuilder};

pub struct Linear {
    weight: QMatrix,
    bias: Tensor<1, f32>,
}

impl Linear {
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let weight = vb.get("weight", device)?;
        let bias = vb.get("bias", device)?.dequantize();
        Ok(Self { weight, bias })
    }

    pub fn forward(&self, input: &Tensor<3, f32>) -> Tensor<3, f32> {
        let output = input.q_mat_mul(&self.weight);
        output.add_(&self.bias)
    }

    pub fn quantization(&self) -> GgmlType {
        self.weight.datatype()
    }
}
