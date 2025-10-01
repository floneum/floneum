use std::fmt::Write;

use crate::{
    DataTypeEnum, SmallerRank, TILE_SIZE, Tensor, TensorData,
    compute_graph::AnyComputeKey,
    mir::{
        kernel::GenericKernel,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShapeConstraints},
    },
};

const BLOCKSIZE: u32 = 256;

#[derive(Debug, Clone)]
pub(crate) struct ResizeOperation {
    pub(crate) input: AnyComputeKey,
    pub(crate) current_shape: Box<[usize]>,
    pub(crate) new_shape: Box<[usize]>,
    pub(crate) fill_shape: Box<[usize]>,
}

impl ResizeOperation {
    pub fn new(
        input: AnyComputeKey,
        current_shape: Box<[usize]>,
        new_shape: Box<[usize]>,
        fill_shape: Box<[usize]>,
    ) -> Self {
        Self {
            input,
            current_shape,
            new_shape,
            fill_shape,
        }
    }
}

impl ResizeOperation {
    fn kernel(
        &self,
        input_rank: u32,
        datatype: DataTypeEnum,
        tile_size: u32,
        kernel: &mut GenericKernel,
    ) {
        let global_id = kernel.global_id();
        let input = kernel.add_tensor_input(input_rank, true, datatype);
        let output = kernel.add_tensor_input(self.new_shape.len() as u32, true, datatype);

        for local_index in 0..tile_size {
            writeln!(kernel, "{{").unwrap();
            for (prefix, tensor) in [("input", &input), ("output", &output)] {
                writeln!(
                    kernel,
                    "var {prefix}_remaining_index = {global_id}.x * {tile_size} + {local_index};"
                )
                .unwrap();
                for i in (0..tensor.rank()).rev() {
                    let shape_i = tensor.shape_binding(i);
                    writeln!(
                        kernel,
                        "let {prefix}_index_{i} = {prefix}_remaining_index % {shape_i};",
                    )
                    .unwrap();
                    writeln!(kernel, "{prefix}_remaining_index /= {shape_i};",).unwrap();
                }
            }
            write!(kernel, "let input_index = ").unwrap();
            input.strided_index(kernel, (0..).map(|i| format!("input_index_{i}")));
            writeln!(kernel, ";").unwrap();
            write!(kernel, "let output_index = ").unwrap();
            output.strided_index(kernel, (0..).map(|i| format!("output_index_{i}")));
            writeln!(kernel, ";").unwrap();
            writeln!(kernel, "{output}[output_index] = {input}[input_index];").unwrap();

            writeln!(kernel, "}}").unwrap();
        }
    }
}

impl Operation for ResizeOperation {
    fn workgroup_shape_constraints(
        &self,
        _: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        constraints.add_constraint(0, Constraint::equals(BLOCKSIZE));
        constraints.add_constraint(1, Constraint::equals(1));
        constraints.add_constraint(2, Constraint::equals(1));
        constraints
    }

    fn dispatch_size(
        &self,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> [u32; 3] {
        let input = inputs[0].as_tensor().unwrap();
        [
            (input.layout().shape().iter().product::<usize>() as u32)
                .div_ceil(TILE_SIZE * BLOCKSIZE),
            1,
            1,
        ]
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(AnyComputeKey)) {
        f(self.input);
    }

    fn inputs(
        &self,
        nodes: &crate::compute_graph::ComputeGraphInner,
    ) -> Vec<crate::mir::inputs::MirValue> {
        let input = nodes.cached_results.get(&self.input).unwrap().clone();
        let output = TensorData::new_for_shape(input.device(), &self.new_shape, input.datatype());
        let output_sliced =
            output.slice(&self.fill_shape.iter().map(|x| 0..*x).collect::<Vec<_>>());
        vec![input.into(), output_sliced.into()]
    }

    fn build_kernel(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
        kernel: &mut GenericKernel,
    ) {
        let input = inputs[0].as_tensor().unwrap();
        let rank = input.layout().rank() as u32;
        let datatype = input.datatype();
        self.kernel(rank, datatype, TILE_SIZE, kernel);
    }

    fn output(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> crate::mir::inputs::MirValue {
        let output = inputs[1].as_tensor().unwrap();
        TensorData::new_from_buffer(
            output.device(),
            output.buffer().clone(),
            &self.new_shape,
            output.datatype(),
        )
        .into()
    }

    fn name(&self) -> String {
        format!(
            "resize_from_{}_to_{}",
            self.current_shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x"),
            self.new_shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
    }
}

impl<const R: usize, T: crate::DataType> Tensor<R, T> {
    pub fn resize(&self, new_shape: [usize; R]) -> Tensor<R, T> {
        let new_shape = new_shape.into();
        let input = self.key();
        self.add_resize(ResizeOperation::new(
            input,
            (*self.shape()).into(),
            new_shape,
            (*self.shape()).into(),
        ))
    }

    pub fn reshape<const R2: usize>(&self, new_shape: [usize; R2]) -> Tensor<R2, T> {
        assert_eq!(
            new_shape.iter().product::<usize>(),
            self.shape().iter().product::<usize>(),
            "Reshape requires the number of elements to be the same. \
            Current shape: {:?}, target shape: {:?}",
            self.shape(),
            new_shape
        );
        let new_shape: Box<[usize]> = new_shape.into();
        let input = self.key();
        self.add_resize(ResizeOperation::new(
            input,
            (*self.shape()).into(),
            new_shape.clone(),
            new_shape.clone(),
        ))
    }

    pub fn flatten_last_n<const FROM_END: usize, const O: usize>(&self) -> Tensor<O, T>
    where
        Self: SmallerRank<FROM_END, O, T>,
    {
        let new_shape = std::array::from_fn(|i| {
            if i < self.rank() - 1 - FROM_END {
                self.shape()[i]
            } else if i == self.rank() - 1 - FROM_END {
                self.shape()[i..].iter().product()
            } else {
                1
            }
        });
        self.reshape(new_shape)
    }

    pub fn flatten_first_n<const FROM_START: usize, const O: usize>(&self) -> Tensor<O, T>
    where
        Self: SmallerRank<FROM_START, O, T>,
    {
        let new_shape = std::array::from_fn(|i| {
            if i == 0 {
                self.shape()[..=FROM_START].iter().product()
            } else {
                self.shape()[i - FROM_START]
            }
        });
        self.reshape(new_shape)
    }

    pub fn flatten_all(&self) -> Tensor<1, T> {
        let size = self.shape().iter().product();
        self.reshape([size])
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_resize() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.resize([30, 20]);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice[[1, 1]], 4.);
    assert_eq!(as_slice[[2, 0]], 5.);
    assert_eq!(as_slice[[2, 1]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_reshape() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.reshape([2, 3]);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[0, 2]], 3.);
    assert_eq!(as_slice[[1, 0]], 4.);
    assert_eq!(as_slice[[1, 1]], 5.);
    assert_eq!(as_slice[[1, 2]], 6.);

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.reshape([6]);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0]], 1.);
    assert_eq!(as_slice[[1]], 2.);
    assert_eq!(as_slice[[2]], 3.);
    assert_eq!(as_slice[[3]], 4.);
    assert_eq!(as_slice[[4]], 5.);
    assert_eq!(as_slice[[5]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_transposed_reshape() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.t();
    let tensor = tensor.reshape([2, 3]);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 3.);
    assert_eq!(as_slice[[0, 2]], 5.);
    assert_eq!(as_slice[[1, 0]], 2.);
    assert_eq!(as_slice[[1, 1]], 4.);
    assert_eq!(as_slice[[1, 2]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_flatten_last_n() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.flatten_last_n::<1, _>();
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0]], 1.);
    assert_eq!(as_slice[[1]], 2.);
    assert_eq!(as_slice[[2]], 3.);
    assert_eq!(as_slice[[3]], 4.);
    assert_eq!(as_slice[[4]], 5.);
    assert_eq!(as_slice[[5]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_flatten_first_n() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.flatten_first_n::<1, _>();
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0]], 1.);
    assert_eq!(as_slice[[1]], 2.);
    assert_eq!(as_slice[[2]], 3.);
    assert_eq!(as_slice[[3]], 4.);
    assert_eq!(as_slice[[4]], 5.);
    assert_eq!(as_slice[[5]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_flatten_all() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.flatten_all();
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0]], 1.);
    assert_eq!(as_slice[[1]], 2.);
    assert_eq!(as_slice[[2]], 3.);
    assert_eq!(as_slice[[3]], 4.);
    assert_eq!(as_slice[[4]], 5.);
    assert_eq!(as_slice[[5]], 6.);
}
