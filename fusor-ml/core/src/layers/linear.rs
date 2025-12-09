use fusor_gguf::GgmlType;

use crate::{CastTensor, DataType, Device, QMatrix, Result, Tensor, VarBuilder};

pub struct Linear<T> {
    weight: QMatrix,
    bias: Option<Tensor<1, T>>,
}

impl<T: DataType> Linear<T> {
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let weight = vb.get("weight", device)?;
        let bias = vb.get("bias", device).ok().map(|bias| bias.dequantize());
        Ok(Self { weight, bias })
    }

    pub fn new(weight: QMatrix, bias: Option<Tensor<1, T>>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, input: &Tensor<3, T>) -> Tensor<3, T> {
        let output = input.q_mat_mul(&self.weight);
        match &self.bias {
            None => output,
            Some(bias) => output.add_(bias),
        }
    }

    pub fn quantization(&self) -> GgmlType {
        self.weight.datatype()
    }

    /// Cast the Linear layer to a different data type
    pub fn cast<U: DataType>(self) -> Linear<U>
    where
        T: CastTensor<U>,
    {
        Linear {
            weight: self.weight,
            bias: self.bias.map(|b| b.cast()),
        }
    }
}
