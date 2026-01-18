//! Activation functions that work on both CPU and GPU backends.

use crate::{
    AddOp, ConcreteTensor, DivOp, ExpOp, FloatOps, GpuOr, MulOp, NegOp, SimdBinaryOp, SimdElement,
    SimdUnaryOp, TanhOp,
};
use fusor_core::{DataType, FloatDataType};

impl<const R: usize, D> GpuOr<R, D, ConcreteTensor<D, R>>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Rectified Linear Unit activation: relu(x) = max(0, x)
    pub fn relu(&self) -> Self {
        self.max_scalar(D::from_f32(0.0))
    }

    /// Sigmoid Linear Unit activation: silu(x) = x / (1 + exp(-x))
    pub fn silu(&self) -> Self
    where
        D: std::ops::Neg<Output = D>
            + std::ops::Add<Output = D>
            + std::ops::Div<Output = D>
            + std::ops::Mul<Output = D>,
        AddOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        MulOp: SimdBinaryOp<D>,
        NegOp: SimdUnaryOp<D>,
        ExpOp: SimdUnaryOp<D>,
    {
        // silu(x) = x / (1 + exp(-x))
        // = x * sigmoid(x)
        let neg_self = match self {
            GpuOr::Cpu(t) => GpuOr::Cpu((-t).eval()),
            GpuOr::Gpu(t) => GpuOr::Gpu(-t.clone()),
        };
        let exp_neg = neg_self.exp();
        let one_plus_exp = exp_neg.add_scalar(D::from_f32(1.0));
        // self / (1 + exp(-self))
        match (self, &one_plus_exp) {
            (GpuOr::Cpu(a), GpuOr::Cpu(b)) => GpuOr::Cpu((a / b).eval()),
            (GpuOr::Gpu(a), GpuOr::Gpu(b)) => GpuOr::Gpu(a / b),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        }
    }

    /// Gaussian Error Linear Unit activation (approximate).
    ///
    /// Uses the tanh approximation:
    /// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn gelu(&self) -> Self
    where
        D: std::ops::Add<Output = D> + std::ops::Mul<Output = D>,
        AddOp: SimdBinaryOp<D>,
        MulOp: SimdBinaryOp<D>,
        TanhOp: SimdUnaryOp<D>,
    {
        // Clamp for numerical stability (tanh is unstable for large inputs)
        let clamped = self.clamp(D::from_f32(-5.5), D::from_f32(5.5));

        let coeff = D::from_f32((2.0 / std::f32::consts::PI).sqrt());

        // x^2
        let x_squared = match &clamped {
            GpuOr::Cpu(t) => GpuOr::Cpu((t * t).eval()),
            GpuOr::Gpu(t) => GpuOr::Gpu(t * t),
        };

        // 1 + 0.044715 * x^2
        let inner_factor = x_squared.mul_scalar(D::from_f32(0.044715)).add_scalar(D::from_f32(1.0));

        // x * (1 + 0.044715 * x^2)
        let inner = match (&clamped, &inner_factor) {
            (GpuOr::Cpu(a), GpuOr::Cpu(b)) => GpuOr::Cpu((a * b).eval()),
            (GpuOr::Gpu(a), GpuOr::Gpu(b)) => GpuOr::Gpu(a * b),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        };

        // sqrt(2/pi) * (x * (1 + 0.044715 * x^2))
        let tanh_input = inner.mul_scalar(coeff);

        // tanh(...)
        let tanh_result = tanh_input.tanh();

        // 1 + tanh(...)
        let one_plus_tanh = tanh_result.add_scalar(D::from_f32(1.0));

        // x * (1 + tanh(...))
        let product = match (self, &one_plus_tanh) {
            (GpuOr::Cpu(a), GpuOr::Cpu(b)) => GpuOr::Cpu((a * b).eval()),
            (GpuOr::Gpu(a), GpuOr::Gpu(b)) => GpuOr::Gpu(a * b),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        };

        // 0.5 * x * (1 + tanh(...))
        product.mul_scalar(D::from_f32(0.5))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_relu_cpu() {
        let t: GpuOr<1, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([6], &[1.0, -2.0, -3.0, 4.0, 5.0, -6.0]));
        let result = t.relu();
        let slice = result.as_slice().await.unwrap();

        assert!((slice[[0]] - 1.0).abs() < 0.001);
        assert!((slice[[1]] - 0.0).abs() < 0.001);
        assert!((slice[[2]] - 0.0).abs() < 0.001);
        assert!((slice[[3]] - 4.0).abs() < 0.001);
        assert!((slice[[4]] - 5.0).abs() < 0.001);
        assert!((slice[[5]] - 0.0).abs() < 0.001);
    }

    fn silu_ref(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    #[tokio::test]
    async fn test_silu_cpu() {
        let data = [1.0f32, -2.0, -3.0, 4.0, 5.0, -6.0];
        let t: GpuOr<1, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([6], &data));
        let result = t.silu();
        let slice = result.as_slice().await.unwrap();

        for i in 0..6 {
            assert!((slice[[i]] - silu_ref(data[i])).abs() < 0.001, "Mismatch at index {}", i);
        }
    }

    fn gelu_ref(x: f32) -> f32 {
        0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }

    #[tokio::test]
    async fn test_gelu_cpu() {
        let data = [1.0f32, -2.0, -3.0, 4.0, 5.0, -5.0];
        let t: GpuOr<1, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([6], &data));
        let result = t.gelu();
        let slice = result.as_slice().await.unwrap();

        for i in 0..6 {
            assert!((slice[[i]] - gelu_ref(data[i])).abs() < 0.01, "Mismatch at index {}: got {}, expected {}", i, slice[[i]], gelu_ref(data[i]));
        }
    }
}
