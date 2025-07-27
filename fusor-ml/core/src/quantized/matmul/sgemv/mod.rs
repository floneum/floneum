use fusor_gguf::GgmlType;

use crate::{
    mir::{
        globals::{ArrayType, KernelGlobalType, VectorType},
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    }, quantized::matmul::{
        sgemv::{
            general::general_sgemv, q4k::q4k_sgemv, q6k::q6k_sgemv, q_8_0::q_8_0_sgemv, q_n::q_n_sgemv
        }, QMatMulOperation
    }, DataTypeEnum, Device
};
use crate::{
    quantized::matmul::{
        sgemv::{
            q4k::Q4K_SGEMV_CHUNK_SIZE, q6k::Q6K_SGEMV_CHUNK_SIZE, q_8_0::Q_8_0_SGEMV_CHUNK_SIZE, q_n::Q_N_SGEMV_CHUNK_SIZE, 
        },
    }, QMatrix,
};
use std::fmt::Display;

mod general;
pub mod q4k;
pub mod q6k;
pub mod q_8_0;
pub mod q_n;

pub(crate) const SGEMV_CHUNK_SIZE: u32 = 2; // This is the size of the chunk each thread will process at a time
pub(crate) const SGEMV_VECTOR_SIZE: u32 = 4; // This is the size of the chunk we will dot at a time

fn maybe_vec_storage_type(size: u32, dtype: DataTypeEnum) -> String {
    match size {
        1 => format!("{dtype}"),
        2..=4 => format!("vec{size}<{dtype}>"),
        _ => format!("array<{dtype}, {size}u>"),
    }
}

fn maybe_vec_storage_type_enum(size: u32, dtype: DataTypeEnum) -> KernelGlobalType {
    match size {
        1 => KernelGlobalType::Value(dtype),
        2..=4 => KernelGlobalType::Vector(VectorType::new(size.to_string(), dtype)),
        _ => KernelGlobalType::Array(ArrayType::new(size.to_string(), dtype)),
    }
}

fn maybe_vec_storage_subgroup_add(size: u32, value: impl Display) -> String {
    match size {
        1..=4 => format!("subgroupAdd({value})"),
        _ => format!(
            "array({})",
            (0..size)
                .map(|i| { format!("subgroupAdd({value}[{i}])") })
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn maybe_vec_storage_index(size: u32, value: impl Display, index: impl Display) -> String {
    match size {
        0 => unreachable!(),
        1 => format!("{value}"),
        2.. => format!("{value}[{index}]"),
    }
}

pub(crate) fn sgemv(
    op: &QMatMulOperation,
    generic_kernel: &mut GenericKernel,
    workgroup_size: &WorkgroupShape,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    output: &TensorInput,
    _n_size: &str,
    // m size is always 1 for sgemv
    _m_size: &str,
    k_size: &str,
) {
    match op.matrix.datatype {
        GgmlType::Q6K => q6k_sgemv(
            op,
            generic_kernel,
            workgroup_size,
            input_a,
            input_b,
            output,
            _n_size,
            _m_size,
            k_size,
        ),
        GgmlType::Q4K => q4k_sgemv(
            op,
            generic_kernel,
            workgroup_size,
            input_a,
            input_b,
            output,
            _n_size,
            _m_size,
            k_size,
        ),
        GgmlType::Q4_0 | GgmlType::Q5_0 => q_n_sgemv(
            op,
            generic_kernel,
            workgroup_size,
            input_a,
            input_b,
            output,
            _n_size,
            _m_size,
            k_size,
        ),
        GgmlType::Q8_0 => q_8_0_sgemv(
            op,
            generic_kernel,
            workgroup_size,
            input_a,
            input_b,
            output,
            _n_size,
            _m_size,
            k_size,
        ),
        _ => general_sgemv(
            op,
            generic_kernel,
            workgroup_size,
            input_a,
            input_b,
            output,
            _n_size,
            _m_size,
            k_size,
        ),
    }
}




pub(crate) fn dispatch_size(matrix: &QMatrix, n: u32, _m: u32) -> [u32; 3] {
    if matrix.datatype == GgmlType::Q6K {
        return [(n as u32).div_ceil(Q6K_SGEMV_CHUNK_SIZE * 2), 1, 1];
    }
    if matrix.datatype == GgmlType::Q4K {
        return [(n as u32).div_ceil(Q4K_SGEMV_CHUNK_SIZE * 2), 1, 1];
    }
    if matches!(matrix.datatype, GgmlType::Q4_0 | GgmlType::Q5_0) {
        return [(n as u32).div_ceil(Q_N_SGEMV_CHUNK_SIZE * 2), 1, 1];
    }
    if matches!(matrix.datatype, GgmlType::Q8_0) {
        return [(n as u32).div_ceil(Q_8_0_SGEMV_CHUNK_SIZE * 2), 1, 1];
    }
    [(n as u32).div_ceil(SGEMV_CHUNK_SIZE * 2), 1, 1]
}



pub(crate)fn workgroup_shape_constraints(
    matrix: &QMatrix,
    device: &Device,
) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
    let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();
    let limits = device.wgpu_device().limits();
    if matrix.datatype == GgmlType::Q6K
        || matrix.datatype == GgmlType::Q4K
        || matrix.datatype == GgmlType::Q4_0
        || matrix.datatype == GgmlType::Q5_0
        || matrix.datatype == GgmlType::Q8_0
    {
        constraints.add_constraint(
            0,
            crate::mir::workgroup_shape::Constraint::equals(limits.min_subgroup_size.max(64)),
        );
    } else {
        constraints.add_constraint(
            0,
            crate::mir::workgroup_shape::Constraint::less_than(
                limits.max_compute_workgroup_size_x + 1,
            ),
        );
        constraints.add_constraint(
            0,
            crate::mir::workgroup_shape::Constraint::equals(limits.min_subgroup_size.max(16)),
        );
    }
    constraints.add_constraint(1, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints.add_constraint(2, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints
}
