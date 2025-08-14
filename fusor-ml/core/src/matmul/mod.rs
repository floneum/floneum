use crate::mir::operation::Operation;
use crate::{
    Device, ElementWiseFunctions, Tensor,
    compute_graph::AnyComputeKey,
    mir::kernel::GenericKernel,
    tensor::{DataType, DataTypeEnum, TensorData},
};

mod gemm;

#[derive(Debug, Clone)]
pub(crate) struct MatMulOperation {
    pub(crate) datatype: DataTypeEnum,
    pub(crate) first: AnyComputeKey,
    pub(crate) second: AnyComputeKey,
    pub(crate) first_shape: Box<[usize]>,
    pub(crate) second_shape: Box<[usize]>,
    pub(crate) out_shape: Box<[usize]>,
    pub(crate) pre_element_wise: [ElementWiseFunctions; 2],
    pub(crate) post_element_wise: ElementWiseFunctions,
}

impl MatMulOperation {
    pub fn new(
        datatype: DataTypeEnum,
        first: AnyComputeKey,
        second: AnyComputeKey,
        first_shape: &[usize],
        second_shape: &[usize],
    ) -> Self {
        let last_dim = first_shape.len() - 1;
        let second_to_last_dim = first_shape.len() - 2;
        let mut out_shape = first_shape.to_vec();
        out_shape[second_to_last_dim] = first_shape[second_to_last_dim];
        out_shape[last_dim] = second_shape[last_dim];
        assert_eq!(first_shape[last_dim], second_shape[second_to_last_dim]);
        assert!(
            first_shape
                .iter()
                .rev()
                .skip(2)
                .zip(second_shape.iter().rev().skip(2))
                .all(|(a, b)| a == b)
        );

        Self {
            first,
            second,
            first_shape: first_shape.into(),
            second_shape: second_shape.into(),
            out_shape: out_shape.into(),
            datatype,
            pre_element_wise: [
                ElementWiseFunctions::empty(datatype),
                ElementWiseFunctions::empty(datatype),
            ],
            post_element_wise: ElementWiseFunctions::empty(datatype),
        }
    }

    pub fn rank(&self) -> u32 {
        self.out_shape.len() as u32
    }

    pub(crate) fn set_pre_element_wise(&mut self, pre_element_wise: [ElementWiseFunctions; 2]) {
        self.pre_element_wise = pre_element_wise;
    }

    pub(crate) fn set_post_element_wise(&mut self, post_element_wise: ElementWiseFunctions) {
        self.post_element_wise = post_element_wise;
    }
}

impl Operation for MatMulOperation {
    fn workgroup_shape_constraints(
        &self,
        device: &Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        gemm::workgroup_shape_constraints(self, device)
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> [u32; 3] {
        let [input_a, input_b, _output] = inputs else {
            panic!("MatMulOperation requires 3 inputs");
        };
        let input_a = input_a.as_tensor().unwrap();
        let input_b = input_b.as_tensor().unwrap();
        let a_shape = input_a.layout().shape();
        let b_shape = input_b.layout().shape();
        let last_dim = self.rank() as usize - 1;
        let last_dim_size = b_shape[last_dim];
        let second_to_last_dim = self.rank() as usize - 2;
        let second_to_last_dim_size = a_shape[second_to_last_dim];
        let batch_size = a_shape.iter().rev().skip(2).product::<usize>();

        gemm::dispatch_size(
            last_dim_size,
            second_to_last_dim_size,
            batch_size,
            workgroup_shape,
        )
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(AnyComputeKey)) {
        f(self.first);
        f(self.second);
    }

    fn inputs(
        &self,
        nodes: &crate::compute_graph::ComputeGraphInner,
    ) -> Vec<crate::mir::inputs::MirValue> {
        let a = nodes.get_result(self.first).unwrap();
        let b = nodes.get_result(self.second).unwrap();
        let last_dim = self.rank() as usize - 1;
        let second_to_last_dim = self.rank() as usize - 2;
        let device = a.device();
        let a_shape = a.layout().shape();
        let b_shape = b.layout().shape();
        let mut out_shape = a_shape.to_vec();
        out_shape[second_to_last_dim] = a_shape[second_to_last_dim];
        out_shape[last_dim] = b_shape[last_dim];
        let output_tensor = TensorData::new_for_shape(device, &out_shape, a.datatype());
        vec![a.into(), b.into(), output_tensor.into()]
    }

    // 1000x1000 dense matmul time on M2 mac pro 1.4743 ms
    fn build_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
        generic_kernel: &mut GenericKernel,
    ) {
        gemm::build_kernel(self, graph, workgroup_shape, inputs, generic_kernel);
    }

    fn output(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> crate::mir::inputs::MirValue {
        let output_tensor = inputs[2].as_tensor().unwrap().clone();
        output_tensor.into()
    }

    fn name(&self) -> String {
        format!(
            "matmul_{}_{}_by_{}",
            self.datatype,
            self.first_shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("x"),
            self.second_shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
    }
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn mat_mul(&self, other: &Self) -> Self {
        self.add_mat_mul(other)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_matmul() {
    let device = Device::new().await.unwrap();

    let data_a = [[1.], [3.]];
    let data_b = [[1., 2.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);
    let tensor = tensor_a.mat_mul(&tensor_b);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice[[1, 1]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_matmul_fused() {
    let device = Device::new().await.unwrap();

    let data_a = [[1.], [3.]];
    let data_b = [[1., 2.]];
    let tensor_a = Tensor::new(&device, &data_a) * 2.;
    let tensor_b = Tensor::new(&device, &data_b);
    let tensor = tensor_a.mat_mul(&tensor_b) / 4.;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], 1. / 2.);
    assert_eq!(as_slice[[0, 1]], 2. / 2.);
    assert_eq!(as_slice[[1, 0]], 3. / 2.);
    assert_eq!(as_slice[[1, 1]], 6. / 2.);
}

#[cfg(test)]
#[tokio::test]
async fn test_transposed_matmul() {
    let device = Device::new().await.unwrap();

    let data_a = [[1.], [3.]];
    let data_b = [[1., 2.]];
    let tensor_a = Tensor::new(&device, &data_a).t();
    let tensor_b = Tensor::new(&device, &data_b).t();
    let tensor = tensor_a.mat_mul(&tensor_b);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], 7.);
}

#[cfg(test)]
#[tokio::test]
async fn test_batched_matmul() {
    let device = Device::new().await.unwrap();

    let data_a = [[[1.], [3.]], [[2.], [6.]]];
    let data_b = [[[1., 2.]], [[2., 4.]]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);
    let tensor = tensor_a.mat_mul(&tensor_b);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0, 0]], 1.);
    assert_eq!(as_slice[[0, 0, 1]], 2.);
    assert_eq!(as_slice[[0, 1, 0]], 3.);
    assert_eq!(as_slice[[0, 1, 1]], 6.);

    assert_eq!(as_slice[[1, 0, 0]], 4.);
    assert_eq!(as_slice[[1, 0, 1]], 8.);
    assert_eq!(as_slice[[1, 1, 0]], 12.);
    assert_eq!(as_slice[[1, 1, 1]], 24.);
}

#[cfg(test)]
#[tokio::test]
async fn test_matmul_f16() {
    let device = Device::new().await.unwrap();

    let data_a = [[half::f16::from_f32(1.)], [half::f16::from_f32(3.)]];
    let data_b = [[half::f16::from_f32(1.), half::f16::from_f32(2.)]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = tensor_a.mat_mul(&tensor_b);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], half::f16::from_f32(1.));
    assert_eq!(as_slice[[0, 1]], half::f16::from_f32(2.));
    assert_eq!(as_slice[[1, 0]], half::f16::from_f32(3.));
    assert_eq!(as_slice[[1, 1]], half::f16::from_f32(6.));
}

#[cfg(test)]
#[tokio::test]
async fn fuzz_matmul() {
    use rand::Rng;

    let device = Device::new().await.unwrap();

    let min_size = 1;
    let max_size = 512;
    let iterations = if cfg!(debug_assertions) { 10 } else { 100 };

    for _ in 0..iterations {
        let size1 = rand::rng().random_range(min_size..max_size);
        let size2 = rand::rng().random_range(min_size..max_size);
        let size3 = rand::rng().random_range(min_size..max_size);

        let data_a: Vec<Vec<f32>> = (0..size1)
            .map(|_| (0..size2).map(|_| rand::random()).collect())
            .collect();
        let data_b: Vec<Vec<f32>> = (0..size2)
            .map(|_| (0..size3).map(|_| rand::random()).collect())
            .collect();

        let tensor_a = Tensor::new(&device, &data_a);
        let tensor_b = Tensor::new(&device, &data_b);

        let mut ndarray_a = ndarray::Array2::zeros((size1, size2));
        for i in 0..size1 {
            for j in 0..size2 {
                ndarray_a[[i, j]] = data_a[i][j];
            }
        }
        let mut ndarray_b = ndarray::Array2::zeros((size2, size3));
        for i in 0..size2 {
            for j in 0..size3 {
                ndarray_b[[i, j]] = data_b[i][j];
            }
        }

        let dot = ndarray_a.dot(&ndarray_b);

        let tensor = tensor_a.mat_mul(&tensor_b);
        let as_slice = tensor.as_slice().await.unwrap();
        for i in 0..size1 {
            for j in 0..size3 {
                if (as_slice[[i, j]] - dot[[i, j]]).abs() > 0.001 {
                    println!(
                        "Mismatch at ({}, {}): {} != {}",
                        i,
                        j,
                        as_slice[[i, j]],
                        dot[[i, j]]
                    );
                    panic!("fuzz failed with size ({size1}x{size2})*({size2}x{size3})");
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn fuzz_batched_matmul() {
    use rand::Rng;
    let device = Device::new().await.unwrap();

    let min_batch_size = 2;
    let max_batch_size = 20;
    let min_size = 1;
    let max_size = 512;
    let iterations = if cfg!(debug_assertions) { 10 } else { 100 };

    for _ in 0..iterations {
        let batch_size = rand::rng().random_range(min_batch_size..max_batch_size);
        let size1 = rand::rng().random_range(min_size..max_size);
        let size2 = rand::rng().random_range(min_size..max_size);
        let size3 = rand::rng().random_range(min_size..max_size);

        let data_a: Vec<Vec<Vec<f32>>> = (0..batch_size)
            .map(|_| {
                (0..size1)
                    .map(|_| (0..size2).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let data_b: Vec<Vec<Vec<f32>>> = (0..batch_size)
            .map(|_| {
                (0..size2)
                    .map(|_| (0..size3).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();

        let tensor_a = Tensor::new(&device, &data_a);
        let tensor_b = Tensor::new(&device, &data_b);

        let ndarray_a = (0..batch_size)
            .map(|i| {
                let mut array = ndarray::Array2::zeros((size1, size2));
                for j in 0..size1 {
                    for k in 0..size2 {
                        array[[j, k]] = data_a[i][j][k];
                    }
                }
                array
            })
            .collect::<Vec<_>>();

        let ndarray_b = (0..batch_size)
            .map(|i| {
                let mut array = ndarray::Array2::zeros((size2, size3));
                for j in 0..size2 {
                    for k in 0..size3 {
                        array[[j, k]] = data_b[i][j][k];
                    }
                }
                array
            })
            .collect::<Vec<_>>();
        let dot = ndarray_a
            .iter()
            .zip(ndarray_b.iter())
            .map(|(a, b)| a.dot(b))
            .collect::<Vec<_>>();

        let tensor = tensor_a.mat_mul(&tensor_b);
        let as_slice = tensor.as_slice().await.unwrap();
        for batch in 0..batch_size {
            for i in 0..size1 {
                for j in 0..size3 {
                    if (as_slice[[batch, i, j]] - dot[batch][[i, j]]).abs() > 0.001 {
                        println!(
                            "Mismatch at ({}, {}): {} != {}",
                            i,
                            j,
                            as_slice[[batch, i, j]],
                            dot[batch][[i, j]]
                        );
                        panic!("fuzz failed with size ({size1}x{size2})*({size2}x{size3})");
                    }
                }
            }
        }
    }
}
