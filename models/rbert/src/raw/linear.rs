use fusor_core::{DataType, Device, QMatrix, Result, Tensor, VarBuilder};

pub struct Linear {
    weight: QMatrix,
    bias: Tensor<1, f32>,
}

impl Linear {
    pub(crate) fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let weight = vb.get("weight", device)?;
        let bias = vb.get("bias", device)?.dequantize();
        Ok(Self { weight, bias })
    }

    pub(crate) fn forward(&self, input: &Tensor<2, f32>) -> Tensor<2, f32> {
        let output = input.q_mat_mul(&self.weight);
        output.add_(&self.bias)
    }
}
