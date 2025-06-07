use crate::{
    compute_graph::{AnyComputeKey, ComputeGraphInner}, Device
};

use super::{
    inputs::KernelInputValue,
    kernel::GenericKernel,
    workgroup_shape::{WorkgroupShape, WorkgroupShapeConstraints},
};

pub(crate) trait Operation {
    fn workgroup_shape_constraints(&self, device: &Device) -> WorkgroupShapeConstraints;

    fn dispatch_size(
        &self,
        workgroup_shape: &WorkgroupShape,
        inputs: &[KernelInputValue],
    ) -> [u32; 3];

    fn visit_dependencies(&self, f: &mut dyn FnMut(AnyComputeKey));

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<KernelInputValue>;

    fn build_kernel(
        &self,
        nodes: &ComputeGraphInner,
        workgroup_shape: &WorkgroupShape,
        inputs: &[KernelInputValue],
        kernel: &mut GenericKernel,
    ) -> KernelInputValue;
}
