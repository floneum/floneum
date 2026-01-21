//! Embedding layer implementation.

use crate::{ConcreteTensor, Device, QMatrix, Tensor, SimdElement, VarBuilder};
use fusor_core::DataType;

/// Embedding layer for token/position embeddings.
///
/// Maps integer indices to dense vectors.
/// Embedding table shape: (num_embeddings, embedding_dim)
#[derive(Clone)]
pub struct Embedding<D: SimdElement> {
    embeddings_quantized: Option<QMatrix<2>>,
    embeddings: Tensor<2, D, ConcreteTensor<D, 2>>,
    num_embeddings: usize,
    embedding_dim: usize,
}

impl<D> Embedding<D>
where
    D: SimdElement + DataType + Default,
{
    /// Create a new embedding layer with the given embedding table (no quantization).
    pub fn new_from_tensor(embeddings: Tensor<2, D, ConcreteTensor<D, 2>>) -> Self {
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

    /// Forward pass: lookup embeddings for the given indices.
    ///
    /// Input: indices tensor of shape (batch,) or (batch, seq_len)
    /// Output: embeddings tensor with an additional dimension for embedding_dim
    ///
    /// For 1D indices (batch,): returns (batch, embedding_dim)
    /// For 2D indices (batch, seq_len): returns (batch, seq_len, embedding_dim)
    pub fn forward_1d(
        &self,
        indices: &Tensor<1, u32, ConcreteTensor<u32, 1>>,
    ) -> Tensor<2, D, ConcreteTensor<D, 2>> {
        self.embeddings.index_select(0, indices)
    }

    /// Forward pass for 2D indices.
    ///
    /// Input shape: (batch, seq_len)
    /// Output shape: (batch, seq_len, embedding_dim)
    pub fn forward(
        &self,
        indices: &Tensor<2, u32, ConcreteTensor<u32, 2>>,
    ) -> Tensor<3, D, ConcreteTensor<D, 3>> {
        let [batch, seq_len] = indices.shape();

        // Flatten indices to 1D
        let indices_flat: Tensor<1, u32, _> = indices.flatten_all();

        // Lookup
        let values = self.embeddings.index_select(0, &indices_flat);

        // Reshape to (batch, seq_len, embedding_dim)
        values.reshape([batch, seq_len, self.embedding_dim])
    }

    /// Get the dequantized embedding table.
    pub fn embeddings(&self) -> &Tensor<2, D, ConcreteTensor<D, 2>> {
        &self.embeddings
    }

    /// Get the number of embeddings.
    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

impl Embedding<f32> {
    /// Create a new embedding layer from a quantized matrix.
    ///
    /// The quantized matrix is dequantized to f32 for efficient lookup.
    pub fn new(embeddings_quantized: QMatrix<2>) -> Self {
        let embeddings: Tensor<2, f32> = embeddings_quantized.dequantize();
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

    /// Load an embedding layer from a VarBuilder.
    ///
    /// Expects weight tensor with shape: (num_embeddings, embedding_dim)
    pub fn load(device: &Device, vb: &mut VarBuilder) -> crate::Result<Self> {
        let embeddings = vb.get("weight", device)?;
        Ok(Self::new(embeddings))
    }

    /// Load an embedding layer with explicit shape verification.
    pub fn load_with_shape(
        device: &Device,
        vb: &mut VarBuilder,
        num_embeddings: usize,
        embedding_dim: usize,
    ) -> crate::Result<Self> {
        let embeddings = vb.get("weight", device)?;
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

    /// Get the quantized embedding table if available.
    pub fn embeddings_quantized(&self) -> &QMatrix<2> {
        self.embeddings_quantized.as_ref().expect("No quantized embeddings available")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embedding_1d() {
        // Create embedding table: (3, 2) - 3 embeddings, each of dimension 2
        let emb_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let embeddings: Tensor<2, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([3, 2], &emb_data));

        let embedding_layer = Embedding::new_from_tensor(embeddings);

        // Input: indices [0, 2, 1]
        let indices_data = [0u32, 2, 1];
        let indices: Tensor<1, u32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &indices_data));

        let result = embedding_layer.forward_1d(&indices);

        assert_eq!(result.shape(), [3, 2]);

        let output = result.as_slice().await.unwrap();

        // index 0 -> [1, 2]
        assert_eq!(output[[0, 0]], 1.0);
        assert_eq!(output[[0, 1]], 2.0);
        // index 2 -> [5, 6]
        assert_eq!(output[[1, 0]], 5.0);
        assert_eq!(output[[1, 1]], 6.0);
        // index 1 -> [3, 4]
        assert_eq!(output[[2, 0]], 3.0);
        assert_eq!(output[[2, 1]], 4.0);
    }

    #[tokio::test]
    async fn test_embedding_2d() {
        // Create embedding table: (3, 2) - 3 embeddings, each of dimension 2
        let emb_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let embeddings: Tensor<2, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([3, 2], &emb_data));

        let embedding_layer = Embedding::new_from_tensor(embeddings);

        // Input: 2D indices [[0, 1], [2, 0]]
        let indices_data = [0u32, 1, 2, 0];
        let indices: Tensor<2, u32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 2], &indices_data));

        let result = embedding_layer.forward(&indices);

        assert_eq!(result.shape(), [2, 2, 2]);

        let output = result.as_slice().await.unwrap();

        // batch 0, seq 0 -> index 0 -> [1, 2]
        assert_eq!(output[[0, 0, 0]], 1.0);
        assert_eq!(output[[0, 0, 1]], 2.0);
        // batch 0, seq 1 -> index 1 -> [3, 4]
        assert_eq!(output[[0, 1, 0]], 3.0);
        assert_eq!(output[[0, 1, 1]], 4.0);
        // batch 1, seq 0 -> index 2 -> [5, 6]
        assert_eq!(output[[1, 0, 0]], 5.0);
        assert_eq!(output[[1, 0, 1]], 6.0);
        // batch 1, seq 1 -> index 0 -> [1, 2]
        assert_eq!(output[[1, 1, 0]], 1.0);
        assert_eq!(output[[1, 1, 1]], 2.0);
    }

    #[tokio::test]
    async fn test_embedding_properties() {
        let emb_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let embeddings: Tensor<2, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &emb_data));

        let embedding_layer = Embedding::new_from_tensor(embeddings);

        assert_eq!(embedding_layer.num_embeddings(), 2);
        assert_eq!(embedding_layer.embedding_dim(), 3);
    }
}
