use std::{
    fmt::Display,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
    sync::OnceLock,
};

use crate::{
    Tensor,
    compute_graph::{AnyComputeKey, ComputeGraphInner},
    layout::TILE_SIZE,
    mir::{function::Function, inputs::MirValue, kernel::GenericKernel, operation::Operation},
    tensor::{DataType, DataTypeEnum, TensorData},
    visit_tiled::{
        MaybeQData, build_visit_tiled_kernel, titled_map_dispatch_size,
        titled_map_workgroup_size_constraints,
    },
};

#[cfg(test)]
use crate::Device;

#[derive(Clone, Debug)]
pub(crate) struct ElementWiseFunctions {
    input_datatype: DataTypeEnum,
    functions: Vec<ElementWiseFunction>,
}

impl ElementWiseFunctions {
    pub fn new(functions: Vec<ElementWiseFunction>, input_datatype: DataTypeEnum) -> Self {
        Self {
            input_datatype,
            functions,
        }
    }

    pub fn empty(input_datatype: DataTypeEnum) -> Self {
        Self {
            input_datatype,
            functions: Vec::new(),
        }
    }

    pub fn add_functions(&self, kernel: &mut GenericKernel) -> Vec<Function> {
        let mut input_datatype = self.input_datatype;
        self.functions
            .iter()
            .rev()
            .map(|f| {
                let function = kernel.add_function(
                    f.datatype,
                    f.operation.clone(),
                    [("input".to_string(), input_datatype.to_string())],
                );
                input_datatype = f.datatype;
                function
            })
            .collect()
    }

    pub fn input_datatype(&self) -> DataTypeEnum {
        self.input_datatype
    }

    pub fn out_datatype(&self) -> DataTypeEnum {
        if let Some(first) = self.functions.first() {
            first.datatype
        } else {
            self.input_datatype
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &ElementWiseFunction> {
        self.functions.iter()
    }

    pub fn push(&mut self, function: ElementWiseFunction) {
        self.functions.push(function);
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ElementWiseOperation {
    pub(crate) value: AnyComputeKey,
    pub(crate) functions: ElementWiseFunctions,
    pub(crate) shape: Box<[usize]>,
}

impl ElementWiseOperation {
    pub fn new(
        input_datatype: DataTypeEnum,
        value: AnyComputeKey,
        functions: ElementWiseFunction,
        shape: impl Into<Box<[usize]>>,
    ) -> Self {
        Self {
            value,
            functions: ElementWiseFunctions {
                input_datatype,
                functions: vec![functions],
            },
            shape: shape.into(),
        }
    }

    pub fn from_element_wise(
        value: AnyComputeKey,
        functions: ElementWiseFunctions,
        shape: impl Into<Box<[usize]>>,
    ) -> Self {
        Self {
            value,
            functions,
            shape: shape.into(),
        }
    }

    pub fn input_datatype(&self) -> DataTypeEnum {
        self.functions.input_datatype
    }

    pub fn rank(&self) -> u32 {
        self.shape.len() as u32
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn requires_new_tensor(&self, tensor: &MaybeQData) -> bool {
        let output_type = self.functions.out_datatype();
        let re_use_allocation = self.functions.input_datatype == output_type && tensor.owned();
        !re_use_allocation
    }

    pub(crate) fn name(&self) -> String {
        self.functions
            .functions
            .iter()
            .map(|f| f.name())
            .collect::<Vec<_>>()
            .join("_")
    }
}

impl Operation for ElementWiseOperation {
    fn workgroup_shape_constraints(
        &self,
        _: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        titled_map_workgroup_size_constraints(self.rank())
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> [u32; 3] {
        let inputs: Box<[_]> = inputs
            .iter()
            .map(|input| {
                let tensor: MaybeQData = input.clone().try_into().unwrap();
                tensor
            })
            .collect();
        titled_map_dispatch_size(TILE_SIZE, *workgroup_shape, &inputs)
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(AnyComputeKey)) {
        f(self.value);
    }

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<crate::mir::inputs::MirValue> {
        let input = nodes.get_result_or_qmatrix(self.value).unwrap();
        let requires_new_tensor = self.requires_new_tensor(&input);

        if requires_new_tensor {
            let output_type = self.functions.out_datatype();
            let new_tensor =
                TensorData::new_for_shape(input.device(), input.layout().shape(), output_type)
                    .into();
            return vec![input.into(), new_tensor];
        }

        vec![input.into()]
    }

    fn output(&self, _: &ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        let tensor: MaybeQData = inputs[0].clone().try_into().unwrap();
        let output_tensor = inputs.get(1).map(|input| {
            let MirValue::Tensor(tensor) = input.clone() else {
                panic!("expected tensor input");
            };
            tensor
        });
        match output_tensor {
            Some(output_tensor) => output_tensor.into(),
            None => tensor.into(),
        }
    }

    fn build_kernel(
        &self,
        _: &ComputeGraphInner,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
        kernel: &mut GenericKernel,
    ) {
        let tensor: MaybeQData = inputs[0].clone().try_into().unwrap();
        let output_tensor = inputs.get(1).map(|input| {
            let MirValue::Tensor(tensor) = input.clone() else {
                panic!("expected tensor input");
            };
            tensor
        });
        let layout = tensor.layout();
        let shape = layout.shape();
        let output_type = self.functions.out_datatype();

        let functions = OnceLock::new();
        let mut datatypes = vec![tensor.datatype()];
        if output_tensor.is_some() {
            datatypes.push(output_type.into());
        }
        build_visit_tiled_kernel(
            shape,
            TILE_SIZE,
            datatypes,
            |kernel, indexes, tensors, values| match (indexes, tensors, values) {
                ([index], [tensor], [value]) => {
                    let result = functions
                        .get_or_init(|| self.functions.add_functions(kernel))
                        .iter()
                        .fold(value.to_string(), |acc, f| f.call(vec![acc]));
                    format!("{tensor}[{index}] = {result};")
                }
                ([_, out_index], [_, output], [value, _]) => {
                    let result = functions
                        .get_or_init(|| self.functions.add_functions(kernel))
                        .iter()
                        .fold(value.to_string(), |acc, f| f.call(vec![acc]));
                    format!("{output}[{out_index}] = {result};")
                }
                _ => panic!("invalid number of tensors"),
            },
            kernel,
        );
    }

    fn name(&self) -> String {
        let functions = self
            .functions
            .functions
            .iter()
            .map(|f| f.name())
            .collect::<Vec<_>>()
            .join("_");
        let shape = self
            .shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("x");
        format!("element_wise_{}_{}", functions, shape)
    }
}

#[derive(Clone, Debug)]
pub struct ElementWiseFunction {
    name: Option<String>,
    operation: String,
    datatype: DataTypeEnum,
}

impl ElementWiseFunction {
    pub fn new(operation: impl Display, datatype: DataTypeEnum) -> Self {
        Self {
            name: None,
            operation: operation.to_string(),
            datatype,
        }
    }

    pub(crate) fn with_name(mut self, name: impl ToString) -> Self {
        self.name = Some(name.to_string());
        self
    }

    pub(crate) fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("element_wise")
    }
}

impl<const R: usize, T: DataType> Add<T> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn add(self, rhs: T) -> Self::Output {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(format!("let output = input + {};", rhs), T::WGSL_TYPE)
                .with_name("add_const"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_add_const() {
    let device = Device::new().await.unwrap();

    let data = [
        [[1., 2.], [1., 2.]],
        [[3., 4.], [3., 4.]],
        [[5., 6.], [5., 6.]],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor + 1.0;

    let output = tensor.as_slice().await.unwrap();
    let result = [
        [[2.0, 3.0], [2.0, 3.0]],
        [[4.0, 5.0], [4.0, 5.0]],
        [[6.0, 7.0], [6.0, 7.0]],
    ];
    let result = Tensor::new(&device, &result);
    let result = result.as_slice().await.unwrap();
    println!("{:?}", output);
    println!("{:?}", result);
    assert_eq!(output, result);

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor + 1.0;

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0, 0]], 2.);
    assert_eq!(output[[0, 1]], 3.);
    assert_eq!(output[[1, 0]], 4.);
    assert_eq!(output[[1, 1]], 5.);
    assert_eq!(output[[2, 0]], 6.);
    assert_eq!(output[[2, 1]], 7.);

    let data = [1., 2.];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor + 1.0;

    let output = tensor.as_slice().await.unwrap();
    assert_eq!(output[[0]], 2.);
    assert_eq!(output[[1]], 3.);
}

#[cfg(test)]
#[tokio::test]
async fn test_add_const_4_dim() {
    let device = Device::new().await.unwrap();

    let data = [
        [
            [[1., 2.], [1., 2.]],
            [[3., 4.], [3., 4.]],
            [[5., 6.], [5., 6.]],
        ],
        [
            [[6., 2.], [1., 2.]],
            [[3., 4.], [3., 4.]],
            [[5., 6.], [5., 6.]],
        ],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor + 1.0;

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    let result = [
        [
            [[2.0, 3.0], [2.0, 3.0]],
            [[4.0, 5.0], [4.0, 5.0]],
            [[6.0, 7.0], [6.0, 7.0]],
        ],
        [
            [[7.0, 3.0], [2.0, 3.0]],
            [[4.0, 5.0], [4.0, 5.0]],
            [[6.0, 7.0], [6.0, 7.0]],
        ],
    ];
    let result = Tensor::new(&device, &result);
    assert_eq!(output, result.as_slice().await.unwrap());
}

macro_rules! impl_add {
    ($($t:ty),*) => {
        $(
            impl<const R: usize> Add<Tensor<R, $t>> for $t {
                type Output = Tensor<R, $t>;

                fn add(self, rhs: Tensor<R, $t>) -> Self::Output {
                    rhs + self
                }
            }
        )*

    };
}
impl_add!(f32, half::f16, u32);

#[cfg(test)]
#[tokio::test]
async fn test_add_const_reversed() {
    let device = Device::new().await.unwrap();

    let data = [
        [[1., 2.], [1., 2.]],
        [[3., 4.], [3., 4.]],
        [[5., 6.], [5., 6.]],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = 1.0 + tensor;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    let result = [
        [[2.0, 3.0], [2.0, 3.0]],
        [[4.0, 5.0], [4.0, 5.0]],
        [[6.0, 7.0], [6.0, 7.0]],
    ];
    let result = Tensor::new(&device, &result);
    assert_eq!(output, result.as_slice().await.unwrap());

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = 1.0 + tensor;

    let output = tensor.as_slice().await.unwrap();
    assert_eq!(output[[0, 0]], 2.);
    assert_eq!(output[[0, 1]], 3.);
    assert_eq!(output[[1, 0]], 4.);
    assert_eq!(output[[1, 1]], 5.);
    assert_eq!(output[[2, 0]], 6.);
    assert_eq!(output[[2, 1]], 7.);

    let data = [1., 2.];
    let tensor = Tensor::new(&device, &data);

    let tensor = 1.0 + tensor;

    let output = tensor.as_slice().await.unwrap();
    assert_eq!(output[[0]], 2.);
    assert_eq!(output[[1]], 3.);
}

#[cfg(test)]
#[tokio::test]
async fn test_add_const_f16() {
    let device = Device::new().await.unwrap();

    let data = [
        [
            [half::f16::from_f32(1.), half::f16::from_f32(2.)],
            [half::f16::from_f32(1.), half::f16::from_f32(2.)],
        ],
        [
            [half::f16::from_f32(3.), half::f16::from_f32(4.)],
            [half::f16::from_f32(3.), half::f16::from_f32(4.)],
        ],
        [
            [half::f16::from_f32(5.), half::f16::from_f32(6.)],
            [half::f16::from_f32(5.), half::f16::from_f32(6.)],
        ],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor + half::f16::from_f32(1.0);

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    let result = [
        [
            [half::f16::from_f32(2.0), half::f16::from_f32(3.0)],
            [half::f16::from_f32(2.0), half::f16::from_f32(3.0)],
        ],
        [
            [half::f16::from_f32(4.0), half::f16::from_f32(5.0)],
            [half::f16::from_f32(4.0), half::f16::from_f32(5.0)],
        ],
        [
            [half::f16::from_f32(6.0), half::f16::from_f32(7.0)],
            [half::f16::from_f32(6.0), half::f16::from_f32(7.0)],
        ],
    ];
    let result = Tensor::new(&device, &result);
    assert_eq!(output, result.as_slice().await.unwrap());
}

#[cfg(test)]
#[tokio::test]
async fn test_add_const_sliced() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let sliced = tensor.slice([0..3, 0..1]);

    let sliced = sliced + 1.0;

    let output = sliced.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 2.);
    assert_eq!(output[[1, 0]], 4.);
    assert_eq!(output[[2, 0]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_add_const_large() {
    let device = Device::new().await.unwrap();

    const BUF_SIZE: usize = 0x010000;
    let data = vec![10.; BUF_SIZE];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor + 1.0;

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    for i in 0..BUF_SIZE {
        assert_eq!(output[[i]], 11.0);
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_merge_add_const() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = (tensor + 1.0) * 2.0;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 4.);
    assert_eq!(output[[0, 1]], 6.);
    assert_eq!(output[[1, 0]], 8.);
    assert_eq!(output[[1, 1]], 10.);
    assert_eq!(output[[2, 0]], 12.);
    assert_eq!(output[[2, 1]], 14.);
}

impl<const R: usize, T: DataType> Sub<T> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn sub(self, rhs: T) -> Self::Output {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(format!("let output = input - {};", rhs), T::WGSL_TYPE)
                .with_name("subtract_const"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_sub_const() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor - 1.0;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 0.);
    assert_eq!(output[[0, 1]], 1.);
    assert_eq!(output[[1, 0]], 2.);
    assert_eq!(output[[1, 1]], 3.);
    assert_eq!(output[[2, 0]], 4.);
    assert_eq!(output[[2, 1]], 5.);
}

macro_rules! impl_sub {
    ($($t:ty),*) => {
        $(
            impl<const R: usize> Sub<Tensor<R, $t>> for $t {
                type Output = Tensor<R, $t>;

                fn sub(self, rhs: Tensor<R, $t>) -> Self::Output {
                    rhs.element_wise(ElementWiseOperation::new(
                        rhs.datatype(),
                        rhs.key(),
                        ElementWiseFunction::new(
                            format!("let output = {self} - input;"),
                            <$t>::WGSL_TYPE,
                        )
                        .with_name("subtract_const"),
                        rhs.shape().as_slice(),
                    ))
                }
            }
        )*
    };
}
impl_sub!(f32, half::f16, u32);

#[cfg(test)]
#[tokio::test]
async fn test_sub_const_reversed() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = 6.0 - tensor;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 5.);
    assert_eq!(output[[0, 1]], 4.);
    assert_eq!(output[[1, 0]], 3.);
    assert_eq!(output[[1, 1]], 2.);
    assert_eq!(output[[2, 0]], 1.);
    assert_eq!(output[[2, 1]], 0.);
}

impl<const R: usize, T: DataType> Mul<T> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(format!("let output = input * {};", rhs), T::WGSL_TYPE)
                .with_name("multiply_const"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_mul_const() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor * 2.0;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 2.);
    assert_eq!(output[[0, 1]], 4.);
    assert_eq!(output[[1, 0]], 6.);
    assert_eq!(output[[1, 1]], 8.);
    assert_eq!(output[[2, 0]], 10.);
    assert_eq!(output[[2, 1]], 12.);
}

macro_rules! impl_mul {
    ($($t:ty),*) => {
        $(
            impl<const R: usize> Mul<Tensor<R, $t>> for $t {
                type Output = Tensor<R, $t>;

                fn mul(self, rhs: Tensor<R, $t>) -> Self::Output {
                    rhs * self
                }
            }
        )*
    };
}
impl_mul!(f32, half::f16, u32);

#[cfg(test)]
#[tokio::test]
async fn test_mul_const_reversed() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = 2.0 * tensor;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 2.);
    assert_eq!(output[[0, 1]], 4.);
    assert_eq!(output[[1, 0]], 6.);
    assert_eq!(output[[1, 1]], 8.);
    assert_eq!(output[[2, 0]], 10.);
    assert_eq!(output[[2, 1]], 12.);
}

impl<const R: usize, T: DataType> Div<T> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn div(self, rhs: T) -> Self::Output {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(format!("let output = input / {};", rhs), T::WGSL_TYPE)
                .with_name("divide_const"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_div_const() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor / 2.0;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 0.5);
    assert_eq!(output[[0, 1]], 1.);
    assert_eq!(output[[1, 0]], 1.5);
    assert_eq!(output[[1, 1]], 2.);
    assert_eq!(output[[2, 0]], 2.5);
    assert_eq!(output[[2, 1]], 3.);
}

macro_rules! impl_div {
    ($($t:ty),*) => {
        $(
            impl<const R: usize> Div<Tensor<R, $t>> for $t {
                type Output = Tensor<R, $t>;

                fn div(self, rhs: Tensor<R, $t>) -> Self::Output {
                    rhs.element_wise(ElementWiseOperation::new(
                        rhs.datatype(),
                        rhs.key(),
                        ElementWiseFunction::new(
                            format!("let output = {} / input;", self),
                            <$t>::WGSL_TYPE,
                        )
                        .with_name("divide_const"),
                        rhs.shape().as_slice(),
                    ))
                }
            }
        )*
    };
}
impl_div!(f32, half::f16, u32);

#[cfg(test)]
#[tokio::test]
async fn test_div_const_reversed() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = 6.0 / tensor;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 6.0 / data[0][0]);
    assert_eq!(output[[0, 1]], 6.0 / data[0][1]);
    assert_eq!(output[[1, 0]], 6.0 / data[1][0]);
    assert_eq!(output[[1, 1]], 6.0 / data[1][1]);
    assert_eq!(output[[2, 0]], 6.0 / data[2][0]);
    assert_eq!(output[[2, 1]], 6.0 / data[2][1]);
}

impl<const R: usize> Rem<u32> for Tensor<R, u32> {
    type Output = Tensor<R, u32>;

    fn rem(self, rhs: u32) -> Self::Output {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(format!("let output = input % {};", rhs), u32::WGSL_TYPE)
                .with_name("mod_const"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_mod_const() {
    let device = Device::new().await.unwrap();

    let data = [[1, 2], [3, 4], [5, 6]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor % 2;

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0, 0]], 1);
    assert_eq!(output[[0, 1]], 0);
    assert_eq!(output[[1, 0]], 1);
    assert_eq!(output[[1, 1]], 0);
    assert_eq!(output[[2, 0]], 1);
    assert_eq!(output[[2, 1]], 0);
}

macro_rules! impl_mod {
    ($($t:ty),*) => {
        $(
            impl<const R: usize> Rem<Tensor<R, $t>> for $t {
                type Output = Tensor<R, $t>;

                fn rem(self, rhs: Tensor<R, $t>) -> Self::Output {
                    rhs.element_wise(ElementWiseOperation::new(
                        rhs.datatype(),
                        rhs.key(),
                        ElementWiseFunction::new(
                            format!("let output = {} % input;", self),
                            <$t>::WGSL_TYPE,
                        )
                        .with_name("mod_const"),
                        rhs.shape().as_slice(),
                    ))
                }
            }
        )*
    };
}
impl_mod!(f32, half::f16, u32);

#[cfg(test)]
#[tokio::test]
async fn test_mod_const_reversed() {
    let device = Device::new().await.unwrap();

    let data = [[1, 2], [3, 4], [5, 6]];
    let tensor = Tensor::new(&device, &data);

    let tensor = 6 % tensor;

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0, 0]], 6 % data[0][0]);
    assert_eq!(output[[0, 1]], 6 % data[0][1]);
    assert_eq!(output[[1, 0]], 6 % data[1][0]);
    assert_eq!(output[[1, 1]], 6 % data[1][1]);
    assert_eq!(output[[2, 0]], 6 % data[2][0]);
    assert_eq!(output[[2, 1]], 6 % data[2][1]);
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    /// Check if each value in the tensor is equal to the given value. Returns 1 for true and 0 for false.
    pub fn eq<D: DataType>(self, rhs: T) -> Tensor<R, D> {
        let datatype = D::WGSL_TYPE;
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(
                format!("let output = {datatype}(input == {});", rhs),
                D::WGSL_TYPE,
            )
            .with_name("equal_const"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_eq_const() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor: Tensor<2, f32> = tensor.eq(1.0);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0, 0]], 1.);
    assert_eq!(output[[0, 1]], 0.);
    assert_eq!(output[[1, 0]], 0.);
    assert_eq!(output[[1, 1]], 0.);
    assert_eq!(output[[2, 0]], 0.);
    assert_eq!(output[[2, 1]], 0.);
}

#[cfg(test)]
#[tokio::test]
async fn test_eq_const_cast() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor: Tensor<2, u32> = tensor.eq(1.0);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0, 0]], 1);
    assert_eq!(output[[0, 1]], 0);
    assert_eq!(output[[1, 0]], 0);
    assert_eq!(output[[1, 1]], 0);
    assert_eq!(output[[2, 0]], 0);
    assert_eq!(output[[2, 1]], 0);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn exp(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = exp(input);", D::WGSL_TYPE).with_name("exp"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_exp() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.exp();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].exp()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].exp()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].exp()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].exp()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].exp()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].exp()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn exp2(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = exp2(input);", D::WGSL_TYPE).with_name("exp2"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_exp2() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.exp2();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].exp2()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].exp2()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].exp2()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].exp2()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].exp2()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].exp2()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn log(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = log(input);", D::WGSL_TYPE).with_name("log"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_log() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.log();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].ln()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].ln()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].ln()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].ln()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].ln()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].ln()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn log2(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = log2(input);", D::WGSL_TYPE).with_name("log2"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_log2() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.log2();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].log2()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].log2()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].log2()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].log2()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].log2()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].log2()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn sqrt(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = sqrt(input);", D::WGSL_TYPE).with_name("sqrt"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_sqrt() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.sqrt();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].sqrt()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].sqrt()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].sqrt()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].sqrt()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].sqrt()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].sqrt()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn sin(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = sin(input);", D::WGSL_TYPE).with_name("sin"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_sin() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.sin();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].sin()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].sin()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].sin()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].sin()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].sin()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].sin()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn cos(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = cos(input);", D::WGSL_TYPE).with_name("cos"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_cos() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.cos();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].cos()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].cos()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].cos()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].cos()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].cos()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].cos()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn tan(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = tan(input);", D::WGSL_TYPE).with_name("tan"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_tan() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.tan();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].tan()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].tan()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].tan()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].tan()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].tan()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].tan()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn asin(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = asin(input);", D::WGSL_TYPE).with_name("asin"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_asin() {
    let device = Device::new().await.unwrap();

    let data = [
        [1.0f32.sin(), 2.0f32.sin()],
        [3.0f32.sin(), 4.0f32.sin()],
        [5.0f32.sin(), 6.0f32.sin()],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.asin();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].asin()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].asin()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].asin()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].asin()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].asin()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].asin()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn acos(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = acos(input);", D::WGSL_TYPE).with_name("acos"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_acos() {
    let device = Device::new().await.unwrap();

    let data = [
        [1.0f32.cos(), 2.0f32.cos()],
        [3.0f32.cos(), 4.0f32.cos()],
        [5.0f32.cos(), 6.0f32.cos()],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.acos();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].acos()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].acos()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].acos()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].acos()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].acos()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].acos()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn atan(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = atan(input);", D::WGSL_TYPE).with_name("atan"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_atan() {
    let device = Device::new().await.unwrap();

    let data = [[1. / 1., 1. / 2.], [1. / 3., 1. / 4.], [1. / 5., 1. / 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.atan();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].atan()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].atan()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].atan()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].atan()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].atan()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].atan()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn sinh(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = sinh(input);", D::WGSL_TYPE).with_name("sinh"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_sinh() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.sinh();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].sinh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].sinh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].sinh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].sinh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].sinh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].sinh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn cosh(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = cosh(input);", D::WGSL_TYPE).with_name("cosh"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_cosh() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.cosh();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].cosh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].cosh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].cosh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].cosh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].cosh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].cosh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn tanh(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = tanh(input);", D::WGSL_TYPE).with_name("tanh"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_tanh() {
    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.tanh();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].tanh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].tanh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].tanh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].tanh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].tanh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].tanh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn asinh(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = asinh(input);", D::WGSL_TYPE).with_name("asinh"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_asinh() {
    let device = Device::new().await.unwrap();

    let data = [
        [1.0f32.sinh(), 2.0f32.sinh()],
        [3.0f32.sinh(), 4.0f32.sinh()],
        [5.0f32.sinh(), 6.0f32.sinh()],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.asinh();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].asinh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].asinh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].asinh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].asinh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].asinh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].asinh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn acosh(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = acosh(input);", D::WGSL_TYPE).with_name("acosh"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_acosh() {
    let device = Device::new().await.unwrap();

    let data = [
        [1.0f32.cosh(), 2.0f32.cosh()],
        [3.0f32.cosh(), 4.0f32.cosh()],
        [5.0f32.cosh(), 6.0f32.cosh()],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.acosh();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].acosh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].acosh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].acosh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].acosh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].acosh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].acosh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn atanh(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = atanh(input);", D::WGSL_TYPE).with_name("atanh"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_atanh() {
    let device = Device::new().await.unwrap();

    let data = [
        [1.0f32.tanh(), 2.0f32.tanh()],
        [3.0f32.tanh(), 4.0f32.tanh()],
        [5.0f32.tanh(), 6.0f32.tanh()],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.atanh();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].atanh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].atanh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].atanh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].atanh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].atanh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].atanh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn abs(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = abs(input);", D::WGSL_TYPE).with_name("abs"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_abs() {
    let device = Device::new().await.unwrap();

    let data = [[1., -2.], [-3., 4.], [5., -6.]];

    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.abs();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].abs()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].abs()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].abs()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].abs()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].abs()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].abs()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Neg for Tensor<R, D> {
    type Output = Tensor<R, D>;

    fn neg(self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = -input;", D::WGSL_TYPE).with_name("neg"),
            self.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_neg() {
    let device = Device::new().await.unwrap();

    let data = [[1., -2.], [-3., 4.], [5., -6.]];

    let tensor = Tensor::new(&device, &data);

    let tensor = -tensor;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] + data[0][0]).abs() < 0.001);
    assert!((output[[0, 1]] + data[0][1]).abs() < 0.001);
    assert!((output[[1, 0]] + data[1][0]).abs() < 0.001);
    assert!((output[[1, 1]] + data[1][1]).abs() < 0.001);
    assert!((output[[2, 0]] + data[2][0]).abs() < 0.001);
    assert!((output[[2, 1]] + data[2][1]).abs() < 0.001);
}

impl<const R: usize, T> Tensor<R, T> {
    pub fn cast<T2>(self) -> Tensor<R, T2>
    where
        T: CastTensor<T2>,
    {
        T::cast(self)
    }
}

pub trait CastTensor<T>: Sized {
    /// Casts the tensor to another type
    fn cast<const R: usize>(tensor: Tensor<R, Self>) -> Tensor<R, T>;
}

impl<T> CastTensor<T> for T {
    fn cast<const R: usize>(tensor: Tensor<R, Self>) -> Tensor<R, Self> {
        tensor
    }
}

impl CastTensor<half::f16> for f32 {
    fn cast<const R: usize>(tensor: Tensor<R, Self>) -> Tensor<R, half::f16> {
        tensor.element_wise(ElementWiseOperation::new(
            tensor.datatype(),
            tensor.key(),
            ElementWiseFunction::new("let output = f16(input);", DataTypeEnum::F16)
                .with_name("cast"),
            tensor.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_f32_to_f16_cast() {
    let device = Device::new().await.unwrap();

    let data = [[1.0f32, 2.0f32], [3.0f32, 4.0f32], [5.0f32, 6.0f32]];
    let tensor = Tensor::new(&device, &data);

    let tensor: Tensor<2, half::f16> = tensor.cast();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], half::f16::from_f32(data[0][0]));
    assert_eq!(output[[0, 1]], half::f16::from_f32(data[0][1]));
    assert_eq!(output[[1, 0]], half::f16::from_f32(data[1][0]));
    assert_eq!(output[[1, 1]], half::f16::from_f32(data[1][1]));
    assert_eq!(output[[2, 0]], half::f16::from_f32(data[2][0]));
    assert_eq!(output[[2, 1]], half::f16::from_f32(data[2][1]));
}

impl CastTensor<f32> for half::f16 {
    fn cast<const R: usize>(tensor: Tensor<R, Self>) -> Tensor<R, f32> {
        tensor.element_wise(ElementWiseOperation::new(
            tensor.datatype(),
            tensor.key(),
            ElementWiseFunction::new("let output = f32(input);", DataTypeEnum::F32)
                .with_name("cast"),
            tensor.shape().as_slice(),
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_f16_to_f32_cast() {
    let device = Device::new().await.unwrap();

    let data = [
        [half::f16::from_f32(1.0), half::f16::from_f32(2.0)],
        [half::f16::from_f32(3.0), half::f16::from_f32(4.0)],
        [half::f16::from_f32(5.0), half::f16::from_f32(6.0)],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor: Tensor<2, f32> = tensor.cast();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], data[0][0].to_f32());
    assert_eq!(output[[0, 1]], data[0][1].to_f32());
    assert_eq!(output[[1, 0]], data[1][0].to_f32());
    assert_eq!(output[[1, 1]], data[1][1].to_f32());
    assert_eq!(output[[2, 0]], data[2][0].to_f32());
    assert_eq!(output[[2, 1]], data[2][1].to_f32());
}
