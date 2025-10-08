use crate::{DataType, Tensor, TensorSlice};

impl<D: DataType> Tensor<2, D> {
    /// Convert a 2D tensor to a `Vec<Vec<D>>`
    pub async fn to_vec2(&self) -> crate::Result<Vec<Vec<D>>, crate::Error> {
        let slice = self.as_slice().await?;
        Ok(slice.to_vec2())
    }
}

impl<D: DataType> Tensor<3, D> {
    /// Convert a 3D tensor to a `Vec<Vec<Vec<D>>>`
    pub async fn to_vec3(&self) -> crate::Result<Vec<Vec<Vec<D>>>, crate::Error> {
        let slice = self.as_slice().await?;
        Ok(slice.to_vec3())
    }
}

impl<D: DataType> TensorSlice<2, D> {
    /// Convert a 2D tensor slice to a `Vec<Vec<D>>`
    pub fn to_vec2(&self) -> Vec<Vec<D>> {
        let shape = self.shape();
        let rows = shape[0];
        let cols = shape[1];

        let mut result = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for j in 0..cols {
                row.push(self[[i, j]]);
            }
            result.push(row);
        }
        result
    }
}

impl<D: DataType> TensorSlice<3, D> {
    /// Convert a 3D tensor slice to a `Vec<Vec<Vec<D>>>`
    pub fn to_vec3(&self) -> Vec<Vec<Vec<D>>> {
        let shape = self.shape();
        let dim0 = shape[0];
        let dim1 = shape[1];
        let dim2 = shape[2];

        let mut result = Vec::with_capacity(dim0);
        for i in 0..dim0 {
            let mut layer = Vec::with_capacity(dim1);
            for j in 0..dim1 {
                let mut row = Vec::with_capacity(dim2);
                for k in 0..dim2 {
                    row.push(self[[i, j, k]]);
                }
                layer.push(row);
            }
            result.push(layer);
        }
        result
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_to_vec2() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., 2.], [3., 4.]];
    let tensor = Tensor::new(&device, &data);
    let vec2 = tensor.to_vec2().await.unwrap();

    assert_eq!(vec2.len(), 2);
    assert_eq!(vec2[0].len(), 2);
    assert_eq!(vec2[0][0], 1.);
    assert_eq!(vec2[0][1], 2.);
    assert_eq!(vec2[1][0], 3.);
    assert_eq!(vec2[1][1], 4.);
}
