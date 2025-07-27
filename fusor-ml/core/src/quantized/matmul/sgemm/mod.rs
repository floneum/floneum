use fusor_gguf::GgmlType;

use crate::{
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    }, quantized::matmul::{
        sgemm::general::general_sgemm, sgemv::{
            q4k::Q4K_SGEMV_CHUNK_SIZE, q6k::Q6K_SGEMV_CHUNK_SIZE, q_8_0::Q_8_0_SGEMV_CHUNK_SIZE, q_n::Q_N_SGEMV_CHUNK_SIZE, SGEMV_CHUNK_SIZE
        }, QMatMulOperation
    }, Device, QMatrix
};

mod general;

pub(crate) fn sgemm(
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
        _ => general_sgemm(
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
