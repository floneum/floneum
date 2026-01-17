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
mod gpu;
mod ops;

pub use device::Device;
pub use error::Error;
pub use gpu::{GpuAdd, GpuDiv, GpuMul, GpuNeg, GpuSub, GpuTensor, GpuTensorLike, HasDevice};
pub use gpu::{GpuAbs, GpuCos, GpuExp, GpuExp2, GpuLog, GpuLog2, GpuSin, GpuSqrt, GpuTan, GpuTanh};

// Re-export from fusor-cpu
pub use fusor_cpu::{
    Abs, Add, ConcreteTensor, Cos, Div, Exp, Exp2, Expr, Log, Log2, Mul, Neg, ResolveTensor,
    ResolvedTensor, SimdElement, Sin, Sqrt, Sub, Tan, Tanh, Tensor as CpuTensor,
};

// Re-export from fusor-core for GPU types
pub use fusor_core::{DataType, FloatDataType};

/// Runtime dispatch wrapper - holds either CPU or GPU version of an operation/tensor type.
///
/// This enum enables writing generic code that works with both CPU and GPU tensors
/// while preserving the benefits of each backend:
/// - CPU: Expression types stay lazy and fuse at resolve time
/// - GPU: Operations build a compute graph that batches at resolve time
#[derive(Clone)]
pub enum GpuOr<CpuT, GpuT> {
    Cpu(CpuT),
    Gpu(GpuT),
}

impl<CpuT, GpuT> GpuOr<CpuT, GpuT> {
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
    pub fn as_cpu(&self) -> Option<&CpuT> {
        match self {
            GpuOr::Cpu(t) => Some(t),
            _ => None,
        }
    }

    /// Returns a reference to the GPU tensor if this is the GPU variant.
    #[inline]
    pub fn as_gpu(&self) -> Option<&GpuT> {
        match self {
            GpuOr::Gpu(t) => Some(t),
            _ => None,
        }
    }

    /// Returns a mutable reference to the CPU tensor if this is the CPU variant.
    #[inline]
    pub fn as_cpu_mut(&mut self) -> Option<&mut CpuT> {
        match self {
            GpuOr::Cpu(t) => Some(t),
            _ => None,
        }
    }

    /// Returns a mutable reference to the GPU tensor if this is the GPU variant.
    #[inline]
    pub fn as_gpu_mut(&mut self) -> Option<&mut GpuT> {
        match self {
            GpuOr::Gpu(t) => Some(t),
            _ => None,
        }
    }

    /// Unwrap the CPU variant, panicking if this is a GPU tensor.
    #[inline]
    pub fn unwrap_cpu(self) -> CpuT {
        match self {
            GpuOr::Cpu(t) => t,
            GpuOr::Gpu(_) => panic!("Expected CPU tensor, found GPU tensor"),
        }
    }

    /// Unwrap the GPU variant, panicking if this is a CPU tensor.
    #[inline]
    pub fn unwrap_gpu(self) -> GpuT {
        match self {
            GpuOr::Gpu(t) => t,
            GpuOr::Cpu(_) => panic!("Expected GPU tensor, found CPU tensor"),
        }
    }

    /// Maps the CPU variant using the provided function.
    #[inline]
    pub fn map_cpu<CpuT2, F: FnOnce(CpuT) -> CpuT2>(self, f: F) -> GpuOr<CpuT2, GpuT> {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(f(t)),
            GpuOr::Gpu(t) => GpuOr::Gpu(t),
        }
    }

    /// Maps the GPU variant using the provided function.
    #[inline]
    pub fn map_gpu<GpuT2, F: FnOnce(GpuT) -> GpuT2>(self, f: F) -> GpuOr<CpuT, GpuT2> {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t),
            GpuOr::Gpu(t) => GpuOr::Gpu(f(t)),
        }
    }

    /// Maps both variants using the provided functions.
    #[inline]
    pub fn map<CpuT2, GpuT2, Fc: FnOnce(CpuT) -> CpuT2, Fg: FnOnce(GpuT) -> GpuT2>(
        self,
        cpu_f: Fc,
        gpu_f: Fg,
    ) -> GpuOr<CpuT2, GpuT2> {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(cpu_f(t)),
            GpuOr::Gpu(t) => GpuOr::Gpu(gpu_f(t)),
        }
    }

    /// Returns the device this tensor is on.
    pub fn device(&self) -> Device
    where
        GpuT: gpu::HasDevice,
    {
        match self {
            GpuOr::Cpu(_) => Device::Cpu,
            GpuOr::Gpu(t) => Device::Gpu(t.gpu_device().clone()),
        }
    }
}

impl<CpuT: std::fmt::Debug, GpuT: std::fmt::Debug> std::fmt::Debug for GpuOr<CpuT, GpuT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuOr::Cpu(t) => f.debug_tuple("Cpu").field(t).finish(),
            GpuOr::Gpu(t) => f.debug_tuple("Gpu").field(t).finish(),
        }
    }
}


impl<const R: usize, T> GpuOr<fusor_cpu::ConcreteTensor<T, R>, gpu::GpuTensor<R, T>>
where
    T: fusor_cpu::SimdElement + fusor_core::DataType + Default,
{
    /// Create a CPU tensor filled with zeros.
    pub fn cpu_zeros(shape: [usize; R]) -> Self {
        GpuOr::Cpu(fusor_cpu::ConcreteTensor::zeros(shape))
    }

    /// Create a CPU tensor from a slice.
    pub fn cpu_from_slice(shape: [usize; R], data: &[T]) -> Self {
        GpuOr::Cpu(fusor_cpu::ConcreteTensor::from_slice(shape, data))
    }

    /// Create a GPU tensor filled with zeros.
    pub fn gpu_zeros(device: &fusor_core::Device, shape: [usize; R]) -> Self {
        GpuOr::Gpu(gpu::GpuTensor::zeros(device, shape))
    }

    /// Create a GPU tensor filled with a specific value.
    pub fn gpu_full(device: &fusor_core::Device, value: T, shape: [usize; R]) -> Self {
        GpuOr::Gpu(gpu::GpuTensor::full(device, value, shape))
    }
}
