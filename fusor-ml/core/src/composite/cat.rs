use crate::{DataType, Tensor};

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn cat(vectors: impl IntoIterator<Item = Self>, dim: usize) -> Self {
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
    println!("{:?}", output);
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
