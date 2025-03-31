use std::{
    fmt::{Display, Write},
    ops::{Add, Div, Mul, Sub},
    sync::OnceLock,
};

use wgpu::CommandEncoder;

use crate::{
    compute_graph::AnyComputeKey, kernel::{Function, GenericKernel}, layout::TILE_SIZE, query::PerformanceQueries, tensor::{DataType, DataTypeEnum, TensorData}, visit_tiled::{MaybeQData, VisitTiledKernel}, ElementWiseFunction, Tensor, UntypedElementWiseKernel
};

#[derive(Clone)]
pub(crate) struct PairWiseOperation {
    pub(crate) first: AnyComputeKey,
    pub(crate) second: AnyComputeKey,
    pub(crate) function: PairWiseFunction,
}

impl PairWiseOperation {
    pub fn new(function: PairWiseFunction, first: AnyComputeKey, second: AnyComputeKey) -> Self {
        Self {
            function,
            first,
            second,
        }
    }
}

pub(crate) struct UntypedPairWiseKernel {
    pre_element_wise: [UntypedElementWiseKernel; 2],
    function: PairWiseFunction,
    post_element_wise: UntypedElementWiseKernel,
    dense_kernel: OnceLock<VisitTiledKernel>,
    sparse_kernel: OnceLock<VisitTiledKernel>,
}

impl UntypedPairWiseKernel {
    pub fn new(function: PairWiseFunction) -> Self {
        let datatype = function.datatype;
        Self {
            pre_element_wise: [
                UntypedElementWiseKernel::empty(datatype),
                UntypedElementWiseKernel::empty(datatype),
            ],
            function,
            post_element_wise: UntypedElementWiseKernel::empty(datatype),
            dense_kernel: OnceLock::new(),
            sparse_kernel: OnceLock::new(),
        }
    }

    pub fn set_post_element_wise(&mut self, element_wise: UntypedElementWiseKernel) {
        self.post_element_wise = element_wise;
    }

    pub fn set_pre_element_wise(&mut self, element_wise: [UntypedElementWiseKernel; 2]) {
        self.pre_element_wise = element_wise;
    }

    fn output_datatype(&self) -> DataTypeEnum {
        self.post_element_wise.out_datatype()
    }

    pub fn add_function(&self, kernel: &mut GenericKernel) -> Function {
        kernel.add_function(
            self.function.datatype,
            self.function.operation.clone(),
            ["a", "b"].iter().enumerate().map(|(i, x)| {
                (
                    x.to_string(),
                    self.pre_element_wise[i].out_datatype().to_string(),
                )
            }),
        )
    }

    pub fn run_with_query(
        &self,
        first: MaybeQData,
        second: MaybeQData,
        query: Option<&PerformanceQueries>,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        assert_eq!(first.layout().shape(), second.layout().shape());
        let contiguous = first.layout().is_contiguous() && second.layout().is_contiguous();
        let rank = first.layout().rank();
        let re_used_allocation_index = if first.datatype() == self.output_datatype().into()
            && first.owned()
            && !first.layout().allocation_overlaps()
        {
            Some(0)
        } else if second.datatype() == self.output_datatype().into()
            && second.owned()
            && !second.layout().allocation_overlaps()
        {
            Some(1)
        } else {
            None
        };
        let output_tensor_index = re_used_allocation_index.unwrap_or(2);
        let requires_new_tensor = re_used_allocation_index.is_none();

        let pre_element_wise_functions: OnceLock<[Vec<Function>; 2]> = OnceLock::new();
        let pair_wise_function = OnceLock::new();
        let post_element_wise_functions = OnceLock::new();
        let create_kernel = || {
            let mut datatypes = vec![first.datatype().into(), second.datatype().into()];

            if requires_new_tensor {
                datatypes.push(self.output_datatype().into());
            }

            VisitTiledKernel::new(
                rank as u32,
                TILE_SIZE,
                contiguous,
                datatypes,
                |kernel, indexes, tensors, values| {
                    let first_value = &values[0];
                    let second_value = &values[1];
                    let output_index = &indexes[output_tensor_index];
                    let out_tensor = &tensors[output_tensor_index];
                    let mut kernel_text = String::new();
                    let pre_element_wise_functions = pre_element_wise_functions.get_or_init(|| {
                        std::array::from_fn(|i| self.pre_element_wise[i].add_functions(kernel))
                    });
                    let first_value = pre_element_wise_functions[0]
                        .iter()
                        .fold(first_value.to_string(), |acc, f| f.call(vec![acc]));
                    writeln!(&mut kernel_text, "let a = {first_value};").unwrap();
                    let second_value = pre_element_wise_functions[1]
                        .iter()
                        .fold(second_value.to_string(), |acc, f| f.call(vec![acc]));
                    writeln!(&mut kernel_text, "let b = {second_value};").unwrap();
                    let pair_wise_function =
                        pair_wise_function.get_or_init(|| self.add_function(kernel));
                    let result = pair_wise_function.call(vec!["a".to_string(), "b".to_string()]);
                    let post_element_wise_functions = post_element_wise_functions
                        .get_or_init(|| self.post_element_wise.add_functions(kernel));
                    let result = post_element_wise_functions
                        .iter()
                        .fold(result, |acc, f| f.call(vec![acc]));
                    writeln!(&mut kernel_text, "{out_tensor}[{output_index}] = {result};").unwrap();
                    kernel_text
                },
            )
        };
        let kernel = if contiguous {
            self.dense_kernel.get_or_init(create_kernel)
        } else {
            self.sparse_kernel.get_or_init(create_kernel)
        };
        let mut tensors: Vec<crate::visit_tiled::MaybeQData> =
            vec![first.clone().into(), second.into()];
        if requires_new_tensor {
            let output_tensor = TensorData::new_for_shape(
                first.device(),
                first.layout().shape(),
                self.output_datatype(),
            );
            tensors.push(output_tensor.into());
        }
        let output = match tensors[output_tensor_index].clone() {
            crate::visit_tiled::MaybeQData::Tensor(tensor) => tensor,
            crate::visit_tiled::MaybeQData::QMatrix(_) => unreachable!(),
        };
        kernel.run_with_query(tensors, query, command_encoder);
        output
    }
}

#[derive(Clone)]
pub struct PairWiseFunction {
    name: Option<String>,
    operation: String,
    datatype: DataTypeEnum,
}

impl PairWiseFunction {
    pub fn new(operation: impl Display, datatype: DataTypeEnum) -> Self {
        Self {
            name: None,
            operation: operation.to_string(),
            datatype,
        }
    }

    pub fn with_name(mut self, name: impl ToString) -> Self {
        self.name = Some(name.to_string());
        self
    }

    pub(crate) fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("pair_wise")
    }

    /// Lower the function to an element-wise function where the a and b inputs are
    /// the same.
    pub(crate) fn lower_to_element_wise(self) -> ElementWiseFunction {
        ElementWiseFunction::new(
            format!("let a = input;\nlet b = input;\n{}", self.operation),
            self.datatype,
        )
        .with_name(self.name())
    }
}

impl<const R: usize, T: DataType> Add<Tensor<R, T>> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn add(self, rhs: Tensor<R, T>) -> Self::Output {
        &self + &rhs
    }
}

impl<const R: usize, T: DataType> Add<&Tensor<R, T>> for &Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn add(self, rhs: &Tensor<R, T>) -> Self::Output {
        self.pair_wise(
            rhs,
            PairWiseFunction::new("let output = a + b;".to_string(), T::WGSL_TYPE).with_name("add"),
        )
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    
    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a + &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert_eq!(as_slice[[0, 0]], 1. + 1.);
    assert_eq!(as_slice[[0, 1]], 2. + 2.);
    assert_eq!(as_slice[[1, 0]], 3. + 3.);
    assert_eq!(as_slice[[1, 1]], 4. + 4.);
    assert_eq!(as_slice[[2, 0]], 5. + 5.);
    assert_eq!(as_slice[[2, 1]], 6. + 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add_f16() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    
    let data_a = [
        [half::f16::from_f32(1.), half::f16::from_f32(2.)],
        [half::f16::from_f32(3.), half::f16::from_f32(4.)],
        [half::f16::from_f32(5.), half::f16::from_f32(6.)],
    ];
    let data_b = [
        [half::f16::from_f32(1.), half::f16::from_f32(2.)],
        [half::f16::from_f32(3.), half::f16::from_f32(4.)],
        [half::f16::from_f32(5.), half::f16::from_f32(6.)],
    ];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a + &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert_eq!(as_slice[[0, 0]], half::f16::from_f32(1. + 1.));
    assert_eq!(as_slice[[0, 1]], half::f16::from_f32(2. + 2.));
    assert_eq!(as_slice[[1, 0]], half::f16::from_f32(3. + 3.));
    assert_eq!(as_slice[[1, 1]], half::f16::from_f32(4. + 4.));
    assert_eq!(as_slice[[2, 0]], half::f16::from_f32(5. + 5.));
    assert_eq!(as_slice[[2, 1]], half::f16::from_f32(6. + 6.));
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add_u32() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    
    let data_a = [[1_u32, 2_u32], [3_u32, 4_u32], [5_u32, 6_u32]];
    let data_b = [[1_u32, 2_u32], [3_u32, 4_u32], [5_u32, 6_u32]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a + &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert_eq!(as_slice[[0, 0]], 1 + 1);
    assert_eq!(as_slice[[0, 1]], 2 + 2);
    assert_eq!(as_slice[[1, 0]], 3 + 3);
    assert_eq!(as_slice[[1, 1]], 4 + 4);
    assert_eq!(as_slice[[2, 0]], 5 + 5);
    assert_eq!(as_slice[[2, 1]], 6 + 6);
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add_const_mul_const_add_fused() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    
    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &(tensor_a + 1.) + &(tensor_b * 2.);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert_eq!(as_slice[[0, 0]], (1. + 1.) + (1. * 2.));
    assert_eq!(as_slice[[0, 1]], (2. + 1.) + (2. * 2.));
    assert_eq!(as_slice[[1, 0]], (3. + 1.) + (3. * 2.));
    assert_eq!(as_slice[[1, 1]], (4. + 1.) + (4. * 2.));
    assert_eq!(as_slice[[2, 0]], (5. + 1.) + (5. * 2.));
    assert_eq!(as_slice[[2, 1]], (6. + 1.) + (6. * 2.));
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add_sub_const_fused() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    
    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = (&tensor_a + &tensor_b) - 1.;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert_eq!(as_slice[[0, 0]], 1. + 1. - 1.);
    assert_eq!(as_slice[[0, 1]], 2. + 2. - 1.);
    assert_eq!(as_slice[[1, 0]], 3. + 3. - 1.);
    assert_eq!(as_slice[[1, 1]], 4. + 4. - 1.);
    assert_eq!(as_slice[[2, 0]], 5. + 5. - 1.);
    assert_eq!(as_slice[[2, 1]], 6. + 6. - 1.);
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add_sparse() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    
    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_a = tensor_a.slice([0..3, 0..1]);
    let tensor_b = Tensor::new(&device, &data_b);
    let tensor_b = tensor_b.slice([0..3, 0..1]);

    let tensor = &tensor_a + &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert_eq!(as_slice[[0, 0]], 1. + 1.);
    assert_eq!(as_slice[[1, 0]], 3. + 3.);
    assert_eq!(as_slice[[2, 0]], 5. + 5.);
}

impl<const R: usize, T: DataType> Sub<Tensor<R, T>> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn sub(self, rhs: Tensor<R, T>) -> Self::Output {
        &self - &rhs
    }
}

impl<const R: usize, T: DataType> Sub<&Tensor<R, T>> for &Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn sub(self, rhs: &Tensor<R, T>) -> Self::Output {
        self.pair_wise(
            rhs,
            PairWiseFunction::new("let output = a - b;".to_string(), T::WGSL_TYPE).with_name("sub"),
        )
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_sub() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    
    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a - &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert_eq!(as_slice[[0, 0]], 1. - 1.);
    assert_eq!(as_slice[[0, 1]], 2. - 2.);
    assert_eq!(as_slice[[1, 0]], 3. - 3.);
    assert_eq!(as_slice[[1, 1]], 4. - 4.);
    assert_eq!(as_slice[[2, 0]], 5. - 5.);
    assert_eq!(as_slice[[2, 1]], 6. - 6.);
}

impl<const R: usize, T: DataType> Mul<Tensor<R, T>> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn mul(self, rhs: Tensor<R, T>) -> Self::Output {
        &self * &rhs
    }
}

impl<const R: usize, T: DataType> Mul<&Tensor<R, T>> for &Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn mul(self, rhs: &Tensor<R, T>) -> Self::Output {
        self.pair_wise(
            rhs,
            PairWiseFunction::new("let output = a * b;".to_string(), T::WGSL_TYPE).with_name("mul"),
        )
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_mul() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    
    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a * &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert_eq!(as_slice[[0, 0]], 1. * 1.);
    assert_eq!(as_slice[[0, 1]], 2. * 2.);
    assert_eq!(as_slice[[1, 0]], 3. * 3.);
    assert_eq!(as_slice[[1, 1]], 4. * 4.);
    assert_eq!(as_slice[[2, 0]], 5. * 5.);
    assert_eq!(as_slice[[2, 1]], 6. * 6.);
}

impl<const R: usize, T: DataType> Div<Tensor<R, T>> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn div(self, rhs: Tensor<R, T>) -> Self::Output {
        &self / &rhs
    }
}

impl<const R: usize, T: DataType> Div<&Tensor<R, T>> for &Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn div(self, rhs: &Tensor<R, T>) -> Self::Output {
        self.pair_wise(
            rhs,
            PairWiseFunction::new("let output = a / b;".to_string(), T::WGSL_TYPE).with_name("div"),
        )
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_div() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    
    let data_a = [[1., 4.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a / &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert_eq!(as_slice[[0, 0]], 1. / 1.);
    assert_eq!(as_slice[[0, 1]], 4. / 2.);
    assert_eq!(as_slice[[1, 0]], 3. / 3.);
    assert_eq!(as_slice[[1, 1]], 4. / 4.);
    assert_eq!(as_slice[[2, 0]], 5. / 5.);
    assert_eq!(as_slice[[2, 1]], 6. / 6.);
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn pow(&self, other: &Self) -> Self {
        self.pair_wise(
            other,
            PairWiseFunction::new("let output = pow(a, b);".to_string(), T::WGSL_TYPE)
                .with_name("pow"),
        )
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_pow() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    
    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a.pow(&tensor_b);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert!((as_slice[[0, 0]] - 1_f32.powf(1.)) < 0.001);
    assert!((as_slice[[0, 1]] - 2_f32.powf(2.)) < 0.001);
    assert!((as_slice[[1, 0]] - 3_f32.powf(3.)) < 0.001);
    assert!((as_slice[[1, 1]] - 4_f32.powf(4.)) < 0.001);
    assert!((as_slice[[2, 0]] - 5_f32.powf(5.)) < 0.001);
    assert!((as_slice[[2, 1]] - 6_f32.powf(6.)) < 0.001);
}
