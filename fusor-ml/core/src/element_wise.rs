use std::{
    fmt::Display,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
    sync::OnceLock,
};

use wgpu::CommandEncoder;

use crate::{
    Tensor,
    compute_graph::AnyComputeKey,
    kernel::{Function, GenericKernel},
    layout::TILE_SIZE,
    padded_tensor_size,
    query::PerformanceQueries,
    tensor::{DataType, DataTypeEnum, TensorData},
    visit_tiled::{MaybeQData, VisitTiledKernel},
};

#[cfg(test)]
use crate::Device;

#[derive(Clone)]
pub(crate) struct ElementWiseOperation {
    pub(crate) value: AnyComputeKey,
    pub(crate) function: ElementWiseFunction,
}

impl ElementWiseOperation {
    pub fn new(value: AnyComputeKey, function: ElementWiseFunction) -> Self {
        Self { value, function }
    }
}

pub(crate) struct UntypedElementWiseKernel {
    functions: Vec<ElementWiseFunction>,
    dense_kernel: OnceLock<VisitTiledKernel>,
    sparse_kernel: OnceLock<VisitTiledKernel>,
    input_datatype: DataTypeEnum,
}

impl std::fmt::Debug for UntypedElementWiseKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UntypedElementWiseKernel")
            .field("functions", &self.functions)
            .field("input_datatype", &self.input_datatype)
            .finish()
    }
}

impl UntypedElementWiseKernel {
    pub fn new(functions: Vec<ElementWiseFunction>, input_datatype: DataTypeEnum) -> Self {
        Self {
            functions,
            dense_kernel: OnceLock::new(),
            sparse_kernel: OnceLock::new(),
            input_datatype,
        }
    }

    pub fn empty(datatype: DataTypeEnum) -> Self {
        Self {
            functions: Vec::new(),
            dense_kernel: OnceLock::new(),
            sparse_kernel: OnceLock::new(),
            input_datatype: datatype,
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

    pub fn out_datatype(&self) -> DataTypeEnum {
        if let Some(first) = self.functions.first() {
            first.datatype
        } else {
            self.input_datatype
        }
    }

    pub fn run_with_query(
        &self,
        tensor: MaybeQData,
        query: Option<&PerformanceQueries>,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let contiguous = tensor.layout().is_contiguous();
        let rank = tensor.layout().rank();
        let output_type = self.out_datatype();
        let re_use_allocation = self.input_datatype == output_type && tensor.owned();
        let requires_new_tensor = !re_use_allocation;

        let functions = OnceLock::new();
        let create_kernel = || {
            let mut datatypes = vec![tensor.datatype().into()];
            if requires_new_tensor {
                datatypes.push(output_type.into());
            }
            VisitTiledKernel::new(
                rank as u32,
                TILE_SIZE,
                contiguous,
                datatypes,
                |kernel, indexes, tensors, values| match (indexes, tensors, values) {
                    ([index], [tensor], [value]) => {
                        let result = functions
                            .get_or_init(|| self.add_functions(kernel))
                            .iter()
                            .fold(value.to_string(), |acc, f| f.call(vec![acc]));
                        format!("{tensor}[{index}] = {result};")
                    }
                    ([_, out_index], [_, output], [value, _]) => {
                        let result = functions
                            .get_or_init(|| self.add_functions(kernel))
                            .iter()
                            .fold(value.to_string(), |acc, f| f.call(vec![acc]));
                        format!("{output}[{out_index}] = {result};")
                    }
                    _ => panic!("invalid number of tensors"),
                },
            )
        };
        let kernel = if contiguous {
            self.dense_kernel.get_or_init(create_kernel)
        } else {
            self.sparse_kernel.get_or_init(create_kernel)
        };
        let mut output = None;
        let mut tensors = Vec::new();
        tensors.push(tensor.clone().into());
        if requires_new_tensor {
            let output_buf = tensor
                .device()
                .wgpu_device()
                .create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: padded_tensor_size(
                        (tensor.layout().shape().iter().product::<usize>()
                            * output_type.element_size()) as u64,
                    ),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
            let output_tensor = TensorData::new_from_buffer(
                tensor.device(),
                output_buf,
                tensor.layout().shape(),
                output_type,
            );
            tensors.push(output_tensor.clone().into());
            output = Some(output_tensor);
        }
        kernel.run_with_query(tensors, query, command_encoder);

        output.unwrap_or(match tensor {
            MaybeQData::Tensor(tensor) => tensor,
            _ => unreachable!(),
        })
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

    pub(crate) fn datatype(&self) -> DataTypeEnum {
        self.datatype
    }
}

impl<const R: usize, T: DataType> Add<T> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn add(self, rhs: T) -> Self::Output {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new(
                format!("let output = input + {};", rhs),
                T::WGSL_TYPE,
            )
            .with_name("add_const"),
        })
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
    println!("{:?}", output);
    let result = [
        [[2.0, 3.0], [2.0, 3.0]],
        [[4.0, 5.0], [4.0, 5.0]],
        [[6.0, 7.0], [6.0, 7.0]],
    ];
    let result = Tensor::new(&device, &result);
    assert_eq!(output, result.as_slice().await.unwrap());

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor + 1.0;

    let output = tensor.as_slice().await.unwrap();
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
    println!("{:?}", output);
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
    println!("{:?}", output);
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
    println!("{:?}", output);
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
    println!("{:?}", output);
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
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new(
                format!("let output = input - {};", rhs),
                T::WGSL_TYPE,
            )
            .with_name("subtract_const"),
        })
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
    println!("{:?}", output);
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
                    rhs.element_wise(ElementWiseOperation {
                        value: rhs.key(),
                        function: ElementWiseFunction::new(
                            format!("let output = {self} - input;"),
                            <$t>::WGSL_TYPE,
                        )
                        .with_name("subtract_const"),
                    })
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
    println!("{:?}", output);
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
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new(
                format!("let output = input * {};", rhs),
                T::WGSL_TYPE,
            )
            .with_name("multiply_const"),
        })
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
    println!("{:?}", output);
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
    println!("{:?}", output);
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
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new(
                format!("let output = input / {};", rhs),
                T::WGSL_TYPE,
            )
            .with_name("divide_const"),
        })
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
    println!("{:?}", output);
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
                    rhs.element_wise(ElementWiseOperation {
                        value: rhs.key(),
                        function: ElementWiseFunction::new(
                            format!("let output = {} / input;", self),
                            <$t>::WGSL_TYPE,
                        )
                        .with_name("divide_const"),
                    })
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
    println!("{:?}", output);
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
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new(
                format!("let output = input % {};", rhs),
                u32::WGSL_TYPE,
            )
            .with_name("mod_const"),
        })
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
                    rhs.element_wise(ElementWiseOperation {
                        value: rhs.key(),
                        function: ElementWiseFunction::new(
                            format!("let output = {} % input;", self),
                            <$t>::WGSL_TYPE,
                        )
                        .with_name("mod_const"),
                    })
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
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new(
                format!("let output = {datatype}(input == {});", rhs),
                D::WGSL_TYPE,
            )
            .with_name("equal_const"),
        })
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
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = exp(input);", D::WGSL_TYPE)
                .with_name("exp"),
        })
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
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].exp()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].exp()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].exp()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].exp()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].exp()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].exp()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn exp2(&self) -> Self {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = exp2(input);", D::WGSL_TYPE)
                .with_name("exp2"),
        })
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
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].exp2()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].exp2()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].exp2()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].exp2()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].exp2()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].exp2()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn log(&self) -> Self {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = log(input);", D::WGSL_TYPE)
                .with_name("log"),
        })
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
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].ln()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].ln()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].ln()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].ln()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].ln()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].ln()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn log2(&self) -> Self {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = log2(input);", D::WGSL_TYPE)
                .with_name("log2"),
        })
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
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].log2()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].log2()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].log2()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].log2()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].log2()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].log2()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn sqrt(&self) -> Self {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = sqrt(input);", D::WGSL_TYPE)
                .with_name("sqrt"),
        })
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
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].sqrt()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].sqrt()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].sqrt()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].sqrt()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].sqrt()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].sqrt()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn sin(&self) -> Self {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = sin(input);", D::WGSL_TYPE)
                .with_name("sin"),
        })
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
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].sin()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].sin()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].sin()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].sin()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].sin()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].sin()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn cos(&self) -> Self {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = cos(input);", D::WGSL_TYPE)
                .with_name("cos"),
        })
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
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].cos()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].cos()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].cos()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].cos()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].cos()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].cos()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn tan(&self) -> Self {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = tan(input);", D::WGSL_TYPE)
                .with_name("tan"),
        })
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
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].tan()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].tan()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].tan()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].tan()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].tan()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].tan()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn asin(&self) -> Self {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = asin(input);", D::WGSL_TYPE)
                .with_name("asin"),
        })
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
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].asin()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].asin()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].asin()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].asin()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].asin()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].asin()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn acos(&self) -> Self {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = acos(input);", D::WGSL_TYPE)
                .with_name("acos"),
        })
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
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].acos()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].acos()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].acos()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].acos()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].acos()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].acos()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn atan(&self) -> Self {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = atan(input);", D::WGSL_TYPE)
                .with_name("atan"),
        })
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
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].atan()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].atan()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].atan()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].atan()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].atan()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].atan()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn sinh(&self) -> Self {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = sinh(input);", D::WGSL_TYPE)
                .with_name("sinh"),
        })
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
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].sinh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].sinh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].sinh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].sinh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].sinh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].sinh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn cosh(&self) -> Self {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = cosh(input);", D::WGSL_TYPE)
                .with_name("cosh"),
        })
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
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].cosh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].cosh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].cosh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].cosh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].cosh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].cosh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn tanh(&self) -> Self {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = tanh(input);", D::WGSL_TYPE)
                .with_name("tanh"),
        })
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
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].tanh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].tanh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].tanh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].tanh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].tanh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].tanh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn asinh(&self) -> Self {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = asinh(input);", D::WGSL_TYPE)
                .with_name("asinh"),
        })
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
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].asinh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].asinh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].asinh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].asinh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].asinh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].asinh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn acosh(&self) -> Self {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = acosh(input);", D::WGSL_TYPE)
                .with_name("acosh"),
        })
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
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].acosh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].acosh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].acosh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].acosh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].acosh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].acosh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn atanh(&self) -> Self {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = atanh(input);", D::WGSL_TYPE)
                .with_name("atanh"),
        })
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
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].atanh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].atanh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].atanh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].atanh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].atanh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].atanh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn abs(&self) -> Self {
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = abs(input);", D::WGSL_TYPE)
                .with_name("abs"),
        })
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
    println!("{:?}", output);
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
        self.element_wise(ElementWiseOperation {
            value: self.key(),
            function: ElementWiseFunction::new("let output = -input;", D::WGSL_TYPE)
                .with_name("neg"),
        })
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
    println!("{:?}", output);
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
        tensor.element_wise(ElementWiseOperation {
            value: tensor.key(),
            function: ElementWiseFunction::new("let output = f16(input);", DataTypeEnum::F16)
                .with_name("cast"),
        })
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
    println!("{:?}", output);
    assert_eq!(output[[0, 0]], half::f16::from_f32(data[0][0]));
    assert_eq!(output[[0, 1]], half::f16::from_f32(data[0][1]));
    assert_eq!(output[[1, 0]], half::f16::from_f32(data[1][0]));
    assert_eq!(output[[1, 1]], half::f16::from_f32(data[1][1]));
    assert_eq!(output[[2, 0]], half::f16::from_f32(data[2][0]));
    assert_eq!(output[[2, 1]], half::f16::from_f32(data[2][1]));
}

impl CastTensor<f32> for half::f16 {
    fn cast<const R: usize>(tensor: Tensor<R, Self>) -> Tensor<R, f32> {
        tensor.element_wise(ElementWiseOperation {
            value: tensor.key(),
            function: ElementWiseFunction::new("let output = f32(input);", DataTypeEnum::F32)
                .with_name("cast"),
        })
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
    println!("{:?}", output);
    assert_eq!(output[[0, 0]], data[0][0].to_f32());
    assert_eq!(output[[0, 1]], data[0][1].to_f32());
    assert_eq!(output[[1, 0]], data[1][0].to_f32());
    assert_eq!(output[[1, 1]], data[1][1].to_f32());
    assert_eq!(output[[2, 0]], data[2][0].to_f32());
    assert_eq!(output[[2, 1]], data[2][1].to_f32());
}
