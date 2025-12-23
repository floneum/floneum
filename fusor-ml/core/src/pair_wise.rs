use std::{
    fmt::Display,
    ops::{Add, Div, Mul, Sub},
};

use crate::{
    ElementWiseFunction, MaxRank, Tensor,
    compute_graph::NodeIndex,
    tensor::{DataType, DataTypeEnum},
};

#[derive(Clone, Debug)]
pub(crate) struct PairWiseOperation {
    pub(crate) first: NodeIndex,
    pub(crate) second: NodeIndex,
    pub(crate) function: PairWiseFunction,
    shape: Box<[usize]>,
}

impl PairWiseOperation {
    pub fn new(
        function: PairWiseFunction,
        first: NodeIndex,
        second: NodeIndex,
        shape: &[usize],
    ) -> Self {
        Self {
            function,
            first,
            second,
            shape: shape.into(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

#[derive(Clone, Debug)]
pub struct PairWiseFunction {
    pub(crate) name: Option<String>,
    pub(crate) operation: String,
    pub(crate) datatype: DataTypeEnum,
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

    pub(crate) fn to_nary_function(
        &self,
        input_a_type: DataTypeEnum,
        input_b_type: DataTypeEnum,
    ) -> crate::nary_wise::NaryFunction {
        crate::nary_wise::NaryFunction {
            name: self.name.clone(),
            operation: self.operation.clone(),
            input_names: vec!["a".to_string(), "b".to_string()],
            input_types: vec![input_a_type, input_b_type],
            output_type: self.datatype,
        }
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

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn add_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2, T>) -> Tensor<R3, T>
    where
        (Tensor<R, T>, Tensor<R2, T>): MaxRank<R3, T>,
    {
        Self::broadcast_then_elementwise_op(self, second, |a, b| a + b)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add() {
    use crate::Device;

    let device = Device::test_instance();

    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a + &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

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

    let device = Device::test_instance();
    if !device.f16_supported() {
        return;
    }

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
    println!("{as_slice:?}");

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

    let device = Device::test_instance();

    let data_a = [[1_u32, 2_u32], [3_u32, 4_u32], [5_u32, 6_u32]];
    let data_b = [[1_u32, 2_u32], [3_u32, 4_u32], [5_u32, 6_u32]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a + &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

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

    let device = Device::test_instance();

    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &(tensor_a + 1.) + &(tensor_b * 2.);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

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

    let device = Device::test_instance();

    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = (&tensor_a + &tensor_b) - 1.;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

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

    let device = Device::test_instance();

    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_a = tensor_a.slice([0..3, 0..1]);
    let tensor_b = Tensor::new(&device, &data_b);
    let tensor_b = tensor_b.slice([0..3, 0..1]);

    let tensor = &tensor_a + &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

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

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn sub_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2, T>) -> Tensor<R3, T>
    where
        (Tensor<R, T>, Tensor<R2, T>): MaxRank<R3, T>,
    {
        Self::broadcast_then_elementwise_op(self, second, |a, b| a - b)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_sub() {
    use crate::Device;

    let device = Device::test_instance();

    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a - &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

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

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn mul_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2, T>) -> Tensor<R3, T>
    where
        (Tensor<R, T>, Tensor<R2, T>): MaxRank<R3, T>,
    {
        Self::broadcast_then_elementwise_op(self, second, |a, b| a * b)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_mul() {
    use crate::Device;

    let device = Device::test_instance();

    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a * &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

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

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn div_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2, T>) -> Tensor<R3, T>
    where
        (Tensor<R, T>, Tensor<R2, T>): MaxRank<R3, T>,
    {
        Self::broadcast_then_elementwise_op(self, second, |a, b| a / b)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_div() {
    use crate::Device;

    let device = Device::test_instance();

    let data_a = [[1., 4.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a / &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

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

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn pow_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2, T>) -> Tensor<R3, T>
    where
        (Tensor<R, T>, Tensor<R2, T>): MaxRank<R3, T>,
    {
        Self::broadcast_then_elementwise_op(self, second, |a, b| a.pow(&b))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_pow() {
    use crate::Device;

    let device = Device::test_instance();

    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a.pow(&tensor_b);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert!((as_slice[[0, 0]] - 1_f32.powf(1.)) < 0.001);
    assert!((as_slice[[0, 1]] - 2_f32.powf(2.)) < 0.001);
    assert!((as_slice[[1, 0]] - 3_f32.powf(3.)) < 0.001);
    assert!((as_slice[[1, 1]] - 4_f32.powf(4.)) < 0.001);
    assert!((as_slice[[2, 0]] - 5_f32.powf(5.)) < 0.001);
    assert!((as_slice[[2, 1]] - 6_f32.powf(6.)) < 0.001);
}
