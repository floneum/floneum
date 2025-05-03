use wgpu::CommandEncoder;

use crate::compute_graph::{AnyComputeKey, ComputeGraphInner};

use super::{
    inputs::KernelInputValue,
    kernel::GenericKernel,
    workgroup_shape::{WorkgroupShape, WorkgroupShapeConstraints},
};

pub(crate) trait Operation {
    fn workgroup_shape_constraints(&self) -> WorkgroupShapeConstraints;

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

    fn run(
        &self,
        nodes: &ComputeGraphInner,
        command_encoder: &mut CommandEncoder,
    ) -> KernelInputValue {
        let workgroup_shape = self.workgroup_shape_constraints().solve().unwrap();
        let mut kernel = GenericKernel::new();
        kernel.set_workgroup_size(workgroup_shape);
        let inputs = self.inputs(nodes);
        let dispatch_size = self.dispatch_size(&workgroup_shape, &inputs);
        let result = self.build_kernel(nodes, &workgroup_shape, &inputs, &mut kernel);
        kernel.run(&nodes.device, inputs, command_encoder, dispatch_size);
        result
    }
}
