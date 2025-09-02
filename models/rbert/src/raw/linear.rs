use fusor_core::{DataType, Device, QMatrix, Result, Tensor, VarBuilder};
use pollster::FutureExt;

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

    pub(crate) fn forward(&self, input: &Tensor<3, f32>) -> Tensor<3, f32> {
        let output = input.q_mat_mul(&self.weight).debug_assert_real();
        output.add_(&self.bias).debug_assert_real()
    }
}
