//! Unified CPU/GPU quantized tensor abstraction
//!
//! This module provides `QGpuOr`, a runtime dispatch enum that holds either CPU
//! quantized tensors (`QuantizedTensor`) or GPU quantized matrices (`QMatrix`).

use crate::{Device, GpuOr};
use fusor_core::QMatrix;
use fusor_cpu::{BlockQ4K, BlockQ4_0, BlockQ5_0, BlockQ6K, BlockQ8_0, GgmlType, QuantizedTensor};

/// Unified quantized tensor type that holds either CPU or GPU quantized data.
///
/// This enum enables writing generic code that works with both CPU and GPU
/// quantized tensors while preserving the benefits of each backend.
///
/// The CPU variants are parameterized by block type at compile time, while
/// the GPU variant uses runtime type information via `GgmlType`.
#[derive(Clone)]
pub enum QGpuOr<const R: usize> {
    /// CPU quantized tensor with Q4_0 quantization (4-bit, block size 32)
    CpuQ4_0(QuantizedTensor<BlockQ4_0, R>),
    /// CPU quantized tensor with Q5_0 quantization (5-bit, block size 32)
    CpuQ5_0(QuantizedTensor<BlockQ5_0, R>),
    /// CPU quantized tensor with Q8_0 quantization (8-bit, block size 32)
    CpuQ8_0(QuantizedTensor<BlockQ8_0, R>),
    /// CPU quantized tensor with Q4K quantization (4-bit, block size 256)
    CpuQ4K(QuantizedTensor<BlockQ4K, R>),
    /// CPU quantized tensor with Q6K quantization (6-bit, block size 256)
    CpuQ6K(QuantizedTensor<BlockQ6K, R>),
    /// GPU quantized matrix (type-erased, uses runtime GgmlType)
    Gpu(QMatrix),
}

impl<const R: usize> QGpuOr<R> {
    /// Returns the quantization type (e.g., Q4_0, Q8_0, Q4K, etc.)
    pub fn ggml_type(&self) -> GgmlType {
        match self {
            QGpuOr::CpuQ4_0(_) => GgmlType::Q4_0,
            QGpuOr::CpuQ5_0(_) => GgmlType::Q5_0,
            QGpuOr::CpuQ8_0(_) => GgmlType::Q8_0,
            QGpuOr::CpuQ4K(_) => GgmlType::Q4K,
            QGpuOr::CpuQ6K(_) => GgmlType::Q6K,
            QGpuOr::Gpu(m) => m.datatype(),
        }
    }

    /// Returns true if this is the CPU variant.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        !matches!(self, QGpuOr::Gpu(_))
    }

    /// Returns true if this is the GPU variant.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        matches!(self, QGpuOr::Gpu(_))
    }

    /// Returns the shape of the quantized tensor.
    pub fn shape(&self) -> &[usize] {
        match self {
            QGpuOr::CpuQ4_0(t) => t.element_shape(),
            QGpuOr::CpuQ5_0(t) => t.element_shape(),
            QGpuOr::CpuQ8_0(t) => t.element_shape(),
            QGpuOr::CpuQ4K(t) => t.element_shape(),
            QGpuOr::CpuQ6K(t) => t.element_shape(),
            QGpuOr::Gpu(m) => m.shape(),
        }
    }

    /// Returns the device this tensor is on.
    pub fn device(&self) -> Device {
        match self {
            QGpuOr::CpuQ4_0(_)
            | QGpuOr::CpuQ5_0(_)
            | QGpuOr::CpuQ8_0(_)
            | QGpuOr::CpuQ4K(_)
            | QGpuOr::CpuQ6K(_) => Device::Cpu,
            QGpuOr::Gpu(m) => Device::Gpu(m.device().clone()),
        }
    }

    /// Create a quantized tensor from raw bytes.
    ///
    /// This dispatches to either CPU or GPU based on the device.
    ///
    /// # Arguments
    /// * `device` - The device to create the tensor on
    /// * `shape` - The logical shape in elements (not blocks)
    /// * `bytes` - Raw quantized bytes
    /// * `ty` - The quantization type
    ///
    /// # Panics
    /// Panics if the quantization type is not supported.
    pub fn from_raw_bytes(
        device: &Device,
        shape: [usize; R],
        bytes: &[u8],
        ty: GgmlType,
    ) -> Result<Self, fusor_core::GgufReadError> {
        match device {
            Device::Cpu => Ok(match ty {
                GgmlType::Q4_0 => {
                    QGpuOr::CpuQ4_0(QuantizedTensor::from_raw_bytes(shape, bytes))
                }
                GgmlType::Q5_0 => {
                    QGpuOr::CpuQ5_0(QuantizedTensor::from_raw_bytes(shape, bytes))
                }
                GgmlType::Q8_0 => {
                    QGpuOr::CpuQ8_0(QuantizedTensor::from_raw_bytes(shape, bytes))
                }
                GgmlType::Q4K => {
                    QGpuOr::CpuQ4K(QuantizedTensor::from_raw_bytes(shape, bytes))
                }
                GgmlType::Q6K => {
                    QGpuOr::CpuQ6K(QuantizedTensor::from_raw_bytes(shape, bytes))
                }
                _ => panic!("Unsupported quantization type for CPU: {:?}", ty),
            }),
            Device::Gpu(gpu_device) => {
                let boxed_shape: Box<[usize]> = shape.into();
                let q_matrix = QMatrix::from_parts(gpu_device, bytes, boxed_shape, ty)?;
                Ok(QGpuOr::Gpu(q_matrix))
            }
        }
    }

    /// Dequantize to an f32 tensor.
    ///
    /// This converts the quantized tensor to full-precision f32.
    pub fn dequantize(&self) -> GpuOr<R, f32> {
        match self {
            QGpuOr::CpuQ4_0(t) => GpuOr::Cpu(fusor_cpu::Tensor::new(t.dequantize())),
            QGpuOr::CpuQ5_0(t) => GpuOr::Cpu(fusor_cpu::Tensor::new(t.dequantize())),
            QGpuOr::CpuQ8_0(t) => GpuOr::Cpu(fusor_cpu::Tensor::new(t.dequantize())),
            QGpuOr::CpuQ4K(t) => GpuOr::Cpu(fusor_cpu::Tensor::new(t.dequantize())),
            QGpuOr::CpuQ6K(t) => GpuOr::Cpu(fusor_cpu::Tensor::new(t.dequantize())),
            QGpuOr::Gpu(m) => GpuOr::Gpu(m.dequantize()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qgpuor_cpu_creation_from_raw_bytes() {
        // Create a Q8_0 tensor on CPU using from_raw_bytes
        // Q8_0 has block size 32, so a [2, 64] tensor needs 4 blocks
        // Each Q8_0 block is 34 bytes (2 for f16 scale + 32 for i8 data)
        let shape = [2, 64];
        let num_blocks = 4;
        let block_size_bytes = 34;

        let mut raw_bytes = vec![0u8; num_blocks * block_size_bytes];
        for block_idx in 0..num_blocks {
            let offset = block_idx * block_size_bytes;
            // Set scale to 1.0 as f16
            let scale_f16 = half::f16::from_f32(1.0);
            raw_bytes[offset..offset + 2].copy_from_slice(&scale_f16.to_le_bytes());
            // Set data to sequential values
            for i in 0..32 {
                raw_bytes[offset + 2 + i] = ((block_idx * 32 + i) % 128) as u8;
            }
        }

        let qgpuor: QGpuOr<2> =
            QGpuOr::from_raw_bytes(&Device::Cpu, shape, &raw_bytes, GgmlType::Q8_0).unwrap();

        assert!(qgpuor.is_cpu());
        assert!(!qgpuor.is_gpu());
        assert_eq!(qgpuor.ggml_type(), GgmlType::Q8_0);
        assert_eq!(qgpuor.shape(), &[2, 64]);
        assert!(matches!(qgpuor.device(), Device::Cpu));
    }

    #[test]
    fn test_qgpuor_dequantize_cpu() {
        // Create a simple Q8_0 tensor and verify dequantization
        // Q8_0: scale * data[i] = output
        let shape = [1, 32];
        let block_size_bytes = 34;

        let mut raw_bytes = vec![0u8; block_size_bytes];
        // Set scale to 0.5 as f16
        let scale_f16 = half::f16::from_f32(0.5);
        raw_bytes[0..2].copy_from_slice(&scale_f16.to_le_bytes());
        // Set data to 0, 1, 2, ..., 31 (as signed i8)
        for i in 0..32 {
            raw_bytes[2 + i] = i as u8;
        }

        let qgpuor: QGpuOr<2> =
            QGpuOr::from_raw_bytes(&Device::Cpu, shape, &raw_bytes, GgmlType::Q8_0).unwrap();

        let dequantized = qgpuor.dequantize();
        assert!(dequantized.is_cpu());

        // Verify the dequantized values
        let cpu_result = dequantized.unwrap_cpu();
        for i in 0..32 {
            let expected = 0.5 * (i as f32);
            let actual = cpu_result.get([0, i]);
            assert!(
                (actual - expected).abs() < 1e-3,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }
}
