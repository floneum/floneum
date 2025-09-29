use std::fmt::Display;

use crate::LazyTensorData;

pub enum SharedExpression {
    Constant(u32),
    TensorDim(TensorDim),
}

impl SharedExpression {
    pub fn tensor_dim(tensor: LazyTensorData, dim: usize) -> Self {
        Self::TensorDim(TensorDim { tensor, dim })
    }

    pub fn constant(value: u32) -> Self {
        Self::Constant(value)
    }

    pub fn eval(&self) -> u32 {
        match self {
            SharedExpression::Constant(value) => *value,
            SharedExpression::TensorDim(tensor_dim) => {
                let shape = tensor_dim.tensor.shape();
                shape[tensor_dim.dim] as u32
            }
        }
    }
}

impl Display for SharedExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SharedExpression::Constant(value) => write!(f, "{value}"),
            SharedExpression::TensorDim(tensor_dim) => {
                let shape = tensor_dim.tensor.shape();
                write!(f, "{}", shape[tensor_dim.dim])
            }
        }
    }
}

struct TensorDim {
    tensor: LazyTensorData,
    dim: usize,
}
