use crate::{DataType, Tensor};

impl<const R: usize, D: DataType> Tensor<R, D> {
    /// Split a tensor into chunks of specified size along a given dimension
    pub fn chunk(&self, chunk_size: usize, dim: usize) -> crate::Result<Vec<Self>, crate::Error> {
        let shape = self.shape();
        let dim_size = shape[dim];
        
        if chunk_size == 0 {
            return Err(crate::Error::GgufError(fusor_gguf::GgufReadError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, "chunk_size cannot be zero"))));
        }
        
        let mut chunks = Vec::new();
        let mut start = 0;
        
        while start < dim_size {
            let end = (start + chunk_size).min(dim_size);
            let length = end - start;
            let chunk = self.narrow(dim, start, length);
            chunks.push(chunk);
            start = end;
        }
        
        Ok(chunks)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_chunk() {
    use crate::Device;

    let device = Device::new().await.unwrap();

    let data = [[1., 2., 3., 4.], [5., 6., 7., 8.]];
    let tensor = Tensor::new(&device, &data);
    let chunks = tensor.chunk(2, 1).unwrap();
    
    assert_eq!(chunks.len(), 2);
    
    let first_chunk = chunks[0].as_slice().await.unwrap();
    assert_eq!(first_chunk[[0, 0]], 1.);
    assert_eq!(first_chunk[[0, 1]], 2.);
    assert_eq!(first_chunk[[1, 0]], 5.);
    assert_eq!(first_chunk[[1, 1]], 6.);
    
    let second_chunk = chunks[1].as_slice().await.unwrap();
    assert_eq!(second_chunk[[0, 0]], 3.);
    assert_eq!(second_chunk[[0, 1]], 4.);
    assert_eq!(second_chunk[[1, 0]], 7.);
    assert_eq!(second_chunk[[1, 1]], 8.);
}