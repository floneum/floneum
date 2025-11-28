use crate::{DataType, Dim, NextRank, Tensor};

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn cat(vectors: impl IntoIterator<Item = Self>, dim: impl Dim<R>) -> Self {
        let dim = dim.resolve();
        let vectors = vectors.into_iter().collect::<Vec<_>>();
        let mut shape = [0; R];
        for (i, v) in vectors[0].shape().iter().enumerate() {
            if i != dim {
                shape[i] = *v;
            }
        }
        for vector in &vectors {
            let vector_shape = vector.shape();
            for (i, shape) in shape.iter_mut().enumerate() {
                if i == dim {
                    *shape += vector_shape[i];
                } else {
                    assert_eq!(*shape, vector_shape[i]);
                }
            }
        }
        let mut iter = vectors.into_iter();

        let mut index = 0;
        let first = iter.next().unwrap();
        let mut larger = first.resize(shape);
        index += first.shape()[dim];
        for vector in iter {
            let length = vector.shape()[dim];
            let slice = std::array::from_fn(|i| {
                if i == dim {
                    index..(index + length)
                } else {
                    0..shape[i]
                }
            });
            larger = larger.slice_assign(slice, &vector);
            index += length;
        }
        larger
    }
}

impl<const R1: usize, D: DataType> Tensor<R1, D> {
    pub fn stack<const R2: usize>(
        vectors: impl IntoIterator<Item = Self>,
        dim: impl Dim<R2>,
    ) -> Tensor<R2, D>
    where
        Self: NextRank<R2, D>,
    {
        let dim = dim.resolve();
        Tensor::cat(vectors.into_iter().map(|t| t.unsqueeze(dim)), dim)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_cat() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data1 = [[1., -2.], [-3., 4.], [5., -6.]];
    let tensor1 = Tensor::new(&device, &data1);
    let data2 = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor2 = Tensor::new(&device, &data2);

    let tensor = Tensor::cat([tensor1, tensor2], 1);
    assert_eq!(*tensor.shape(), [3, 4]);

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 1.);
    assert_eq!(output[[0, 1]], -2.);
    assert_eq!(output[[0, 2]], 1.);
    assert_eq!(output[[0, 3]], 2.);

    assert_eq!(output[[1, 0]], -3.);
    assert_eq!(output[[1, 1]], 4.);
    assert_eq!(output[[1, 2]], 3.);
    assert_eq!(output[[1, 3]], 4.);

    assert_eq!(output[[2, 0]], 5.);
    assert_eq!(output[[2, 1]], -6.);
    assert_eq!(output[[2, 2]], 5.);
    assert_eq!(output[[2, 3]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_multi_dim_cat() {
    use crate::{D, Device};

    let device = Device::new().await.unwrap();

    let data1 = vec![vec![vec![1f32; 32]; 11]; 3];
    let tensor1 = Tensor::new(&device, &data1).reshape([1, 3, 11, 32, 1]);
    let data2 = vec![vec![vec![2f32; 32]; 11]; 3];
    let tensor2 = Tensor::new(&device, &data2).reshape([1, 3, 11, 32, 1]);

    let tensor = Tensor::cat([tensor1, tensor2], D::Minus1);
    println!("tensor shape: {:?}", tensor.shape());

    assert_eq!(*tensor.shape(), [1, 3, 11, 32, 2]);

    let output = tensor.i((0usize, .., .., .., ..)).as_slice().await.unwrap();
    println!("{output:?}");

    for i in 0..3 {
        for j in 0..11 {
            for k in 0..32 {
                for l in 0..2 {
                    let value = output[[i, j, k, l]];
                    let expected = if l == 0 { 1f32 } else { 2f32 };
                    assert_eq!(value, expected);
                }
            }
        }
    }
}
