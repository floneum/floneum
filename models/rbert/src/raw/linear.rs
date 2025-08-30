use fusor_core::{DataType, Device, QMatrix, Result, Tensor, VarBuilder};

pub struct Linear {
    weight: QMatrix,
    bias: QMatrix,
}

impl Linear {
    pub(crate) fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let weight = vb.get("weight", device)?;
        let bias = vb.get("bias", device)?;
        Ok(Self { weight, bias })
    }

    pub(crate) fn forward<D: DataType>(&self, input: &Tensor<2, D>) -> Tensor<2, D> {
        let output = input.q_mat_mul(&self.weight);
        &output + &self.bias.dequantize()
    }
}
