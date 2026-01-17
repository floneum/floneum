//! Unified CPU/GPU tensor abstraction
//!
//! This crate provides a unified interface over `fusor-cpu` (CPU tensors with SIMD fusion)
//! and `fusor-core` (GPU tensors with compute graph batching).
//!
//! The key design is:
//! - `GpuOr<CpuT, GpuT>` is a runtime dispatch enum holding either CPU or GPU version
//! - CPU kernel fusion is preserved (expression types stay lazy)
//! - GPU laziness is preserved (compute graph batching)

mod device;
mod error;

pub use device::Device;
pub use error::Error;
use fusor_core::{MappedBuffer, TensorSlice};
use fusor_cpu::TensorBacking;

// Re-export from fusor-cpu
pub use fusor_cpu::{
    Abs, Add, ConcreteTensor, Cos, Div, Exp, Exp2, Expr, Log, Log2, Mul, Neg, ResolveTensor,
    ResolvedTensor, SimdElement, Sin, Sqrt, Sub, Tan, Tanh, Tensor as CpuTensor,
};

pub use fusor_core::Tensor as GpuTensor;

// Re-export from fusor-core for GPU types
pub use fusor_core::{DataType, FloatDataType};

/// Runtime dispatch wrapper - holds either CPU or GPU version of an operation/tensor type.
///
/// This enum enables writing generic code that works with both CPU and GPU tensors
/// while preserving the benefits of each backend:
/// - CPU: Expression types stay lazy and fuse at resolve time
/// - GPU: Operations build a compute graph that batches at resolve time
#[derive(Clone)]
pub enum GpuOr<const R: usize, D, B: TensorBacking<R, Elem=D> = fusor_cpu::ConcreteTensor<D, R>> {
    Cpu(CpuTensor<R, B>),
    Gpu(GpuTensor<R, D>),
}

impl<const R: usize, D, B> GpuOr<R , D, B> where B: TensorBacking<R, Elem=D> {
    /// Returns true if this is the CPU variant.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        matches!(self, GpuOr::Cpu(_))
    }

    /// Returns true if this is the GPU variant.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        matches!(self, GpuOr::Gpu(_))
    }

    /// Returns a reference to the CPU tensor if this is the CPU variant.
    #[inline]
    pub fn as_cpu(&self) -> Option<&CpuTensor<R, B>> {
        match self {
            GpuOr::Cpu(t) => Some(t),
            _ => None,
        }
    }

    /// Returns a reference to the GPU tensor if this is the GPU variant.
    #[inline]
    pub fn as_gpu(&self) -> Option<&GpuTensor<R, D>> {
        match self {
            GpuOr::Gpu(t) => Some(t),
            _ => None,
        }
    }

    /// Returns a mutable reference to the CPU tensor if this is the CPU variant.
    #[inline]
    pub fn as_cpu_mut(&mut self) -> Option<&mut CpuTensor<R, B>> {
        match self {
            GpuOr::Cpu(t) => Some(t),
            _ => None,
        }
    }

    /// Returns a mutable reference to the GPU tensor if this is the GPU variant.
    #[inline]
    pub fn as_gpu_mut(&mut self) -> Option<&mut GpuTensor<R, D>> {
        match self {
            GpuOr::Gpu(t) => Some(t),
            _ => None,
        }
    }

    /// Unwrap the CPU variant, panicking if this is a GPU tensor.
    #[inline]
    pub fn unwrap_cpu(self) -> CpuTensor<R, B> {
        match self {
            GpuOr::Cpu(t) => t,
            GpuOr::Gpu(_) => panic!("Expected CPU tensor, found GPU tensor"),
        }
    }

    /// Unwrap the GPU variant, panicking if this is a CPU tensor.
    #[inline]
    pub fn unwrap_gpu(self) -> GpuTensor<R, D> {
        match self {
            GpuOr::Gpu(t) => t,
            GpuOr::Cpu(_) => panic!("Expected GPU tensor, found CPU tensor"),
        }
    }

    #[inline]
    pub fn dispatch<const R2: usize, D2, B2>(self, cpu_fn: impl FnOnce(CpuTensor<R, B>) -> CpuTensor<R2, B2>, gpu_fn: impl FnOnce(GpuTensor<R, D>) -> GpuTensor<R2, D2>) -> GpuOr<R2, D2, B2> where B2: TensorBacking<R2, Elem=D2> {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(cpu_fn(t)),
            GpuOr::Gpu(t) => GpuOr::Gpu(gpu_fn(t)),
        }
    }

    pub async fn resolve(self) -> Result<TensorSlice<R, D, MappedBuffer>, Error> {
        match self {
            GpuOr::Cpu(t) => Ok(todo!()),
            GpuOr::Gpu(t) => {
                todo!()
            }
        }
    }
}

impl<const R: usize, D, B, B2> std::ops::Add for GpuOr<R, D, B> where CpuTensor<R, B>: std::ops::Add<Output=CpuTensor<R, B2>>, GpuTensor<R, D>: std::ops::Add<Output=GpuTensor<R, D>>, B: TensorBacking<R, Elem=D>, B2: TensorBacking<R, Elem=D> {
    type Output = GpuOr<R, D, B2>;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (GpuOr::Cpu(lhs), GpuOr::Cpu(rhs)) => GpuOr::Cpu(lhs + rhs),
            (GpuOr::Gpu(lhs), GpuOr::Gpu(rhs)) => GpuOr::Gpu(lhs + rhs),
            _ => panic!("Cannot add CPU tensor to GPU tensor"),
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_gpu_or_add() {
    let a_cpu: CpuTensor<1, fusor_cpu::ConcreteTensor<f32, 1>> = fusor_cpu::Tensor::from_slice([3], &[1.0, 2.0, 3.0]);
    let b_cpu: CpuTensor<1, fusor_cpu::ConcreteTensor<f32, 1>> = fusor_cpu::Tensor::from_slice([3], &[4.0, 5.0, 6.0]);
    let device = fusor_core::Device::new().await.unwrap();
    let a_gpu: GpuTensor<1, f32> = GpuTensor::new(&device, &[1.0, 2.0, 3.0]);
    let b_gpu: GpuTensor<1, f32> = GpuTensor::new(&device, &[4.0, 5.0, 6.0]);

    let a_cpu_or = GpuOr::Cpu(a_cpu);
    let b_cpu_or = GpuOr::Cpu(b_cpu);
    let a_gpu_or: GpuOr<1, f32> = GpuOr::Gpu(a_gpu);
    let b_gpu_or: GpuOr<1, f32> = GpuOr::Gpu(b_gpu);

    let c_cpu_or = a_cpu_or + b_cpu_or;
    println!("c_cpu_or: {:?}", c_cpu_or.as_cpu().unwrap().to_vec());
    let c_gpu_or = a_gpu_or + b_gpu_or;
    let c_gpu_resolved = c_gpu_or.as_gpu().unwrap().resolve().await.unwrap();
}