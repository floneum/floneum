use crate::{Device, NextRank, QMatrix, Result, Tensor, VarBuilder};

/// Embedding layer for token/position embeddings
///
/// Maps integer indices to dense vectors.
/// Embedding table shape: (num_embeddings, embedding_dim)
#[derive(Clone, Debug)]
pub struct Embedding {
    embeddings_quantized: Option<QMatrix>,
    embeddings: Tensor<2, f32>,
    num_embeddings: usize,
    embedding_dim: usize,
}

impl Embedding {
    /// Create a new embedding layer with the given embedding table
    pub fn new(embeddings_quantized: QMatrix) -> Self {
        let embeddings = embeddings_quantized.dequantize();
        let shape = embeddings.shape();
        let num_embeddings = shape[0];
        let embedding_dim = shape[1];

        Self {
            embeddings_quantized: Some(embeddings_quantized),
            embeddings,
            num_embeddings,
            embedding_dim,
        }
    }

    /// Create a new embedding layer with the given embedding table
    pub fn new_from_tensor(embeddings: Tensor<2, f32>) -> Self {
        let shape = embeddings.shape();
        let num_embeddings = shape[0];
        let embedding_dim = shape[1];

        Self {
            embeddings_quantized: None,
            embeddings,
            num_embeddings,
            embedding_dim,
        }
    }

    /// Load an embedding layer from VarBuilder
    ///
    /// Expects weight tensor with shape: (num_embeddings, embedding_dim)
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let embeddings = vb.get("weight", device)?;
        Ok(Self::new(embeddings))
    }

    /// Load an embedding layer with explicit shape
    pub fn load_with_shape(
        device: &Device,
        vb: &mut VarBuilder,
        num_embeddings: usize,
        embedding_dim: usize,
    ) -> Result<Self> {
        let embeddings = vb.get("weight", device)?;

        // Verify shape
        let shape = embeddings.shape();
        assert_eq!(
            shape[0], num_embeddings,
            "Embedding num_embeddings mismatch: expected {}, got {}",
            num_embeddings, shape[0]
        );
        assert_eq!(
            shape[1], embedding_dim,
            "Embedding embedding_dim mismatch: expected {}, got {}",
            embedding_dim, shape[1]
        );

        Ok(Self::new(embeddings))
    }

    /// Forward pass: lookup embeddings for the given indices
    ///
    /// Input: indices tensor of any rank N
    /// Output: embeddings tensor of rank M = N + 1
    ///
    /// Example:
    /// - Input: [batch, seq_len] with indices
    /// - Output: [batch, seq_len, embedding_dim] with embeddings
    pub fn forward<const N: usize, const M: usize>(
        &self,
        indexes: &Tensor<N, u32>,
    ) -> Tensor<M, f32>
    where
        Tensor<N, u32>: NextRank<M, u32>,
    {
        // Calculate final output dimensions: input_dims + [embedding_dim]
        let final_dims = std::array::from_fn(|i| {
            if i < N {
                indexes.shape()[i]
            } else {
                self.embedding_dim
            }
        });

        // Flatten indices to 1D, lookup, then reshape
        let indexes = indexes.flatten_all();
        let values = self.embeddings.index_select(0, &indexes);
        values.reshape(final_dims)
    }

    /// Get the embedding table
    pub fn embeddings(&self) -> &Tensor<2, f32> {
        &self.embeddings
    }

    /// Get the quantized embedding table
    pub fn embeddings_quantized(&self) -> &QMatrix {
        self.embeddings_quantized.as_ref().unwrap()
    }

    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[tokio::test]
    async fn test_embedding_1d_input() {
        let device = Device::new().await.unwrap();

        // Create embedding table: (3, 2) - 3 embeddings, each of dimension 2
        let emb_data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let embeddings = Tensor::new(&device, &emb_data);

        let embedding_layer = Embedding::new_from_tensor(embeddings);

        // Input: indices [0, 2, 1]
        let indices_data = [0u32, 2, 1];
        let indices = Tensor::new(&device, &indices_data);

        // Forward pass
        let result: Tensor<2, f32> = embedding_layer.forward(&indices);

        // Expected shape: (3, 2)
        assert_eq!(result.shape(), &[3, 2]);

        let output = result.as_slice().await.unwrap();

        // Verify lookups
        assert_eq!(output[[0, 0]], 1.0); // index 0
        assert_eq!(output[[0, 1]], 2.0);
        assert_eq!(output[[1, 0]], 5.0); // index 2
        assert_eq!(output[[1, 1]], 6.0);
        assert_eq!(output[[2, 0]], 3.0); // index 1
        assert_eq!(output[[2, 1]], 4.0);
    }

    #[tokio::test]
    async fn test_embedding_2d_input() {
        let device = Device::new().await.unwrap();

        // Create embedding table: (3, 2) - 3 embeddings, each of dimension 2
        let emb_data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let embeddings = Tensor::new(&device, &emb_data);

        let embedding_layer = Embedding::new_from_tensor(embeddings);

        // Input: 2D indices [[0, 1], [2, 0]]
        let indices_data = [[0u32, 1], [2, 0]];
        let indices = Tensor::new(&device, &indices_data);

        // Forward pass
        let result: Tensor<3, f32> = embedding_layer.forward(&indices);

        // Expected shape: (2, 2, 2)
        assert_eq!(result.shape(), &[2, 2, 2]);

        let output = result.as_slice().await.unwrap();

        // Verify lookups
        assert_eq!(output[[0, 0, 0]], 1.0); // index 0
        assert_eq!(output[[0, 0, 1]], 2.0);
        assert_eq!(output[[0, 1, 0]], 3.0); // index 1
        assert_eq!(output[[0, 1, 1]], 4.0);
        assert_eq!(output[[1, 0, 0]], 5.0); // index 2
        assert_eq!(output[[1, 0, 1]], 6.0);
        assert_eq!(output[[1, 1, 0]], 1.0); // index 0
        assert_eq!(output[[1, 1, 1]], 2.0);
    }

    #[tokio::test]
    async fn test_embedding_properties() {
        let device = Device::new().await.unwrap();

        let emb_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let embeddings = Tensor::new(&device, &emb_data);

        let embedding_layer = Embedding::new_from_tensor(embeddings);

        assert_eq!(embedding_layer.num_embeddings(), 2);
        assert_eq!(embedding_layer.embedding_dim(), 3);
    }
}
