//! Embedding Layer.
use fusor_core::{Device, NextRank, Result, Tensor};

#[derive(Clone, Debug)]
pub struct Embedding {
    embeddings: Tensor<2, f32>,
    hidden_size: usize,
}

impl Embedding {
    fn new(embeddings: Tensor<2, f32>) -> Self {
        let hidden_size = embeddings.shape()[1];
        Self {
            embeddings,
            hidden_size,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn embeddings(&self) -> &Tensor<2, f32> {
        &self.embeddings
    }
}

impl Embedding {
    pub fn forward<const N: usize, const M: usize>(
        &self,
        indexes: &Tensor<N, u32>,
    ) -> Tensor<M, f32>
    where
        Tensor<N, u32>: NextRank<M, u32>,
    {
        let final_dims = std::array::from_fn(|i| {
            if i < N {
                indexes.shape()[i]
            } else {
                self.hidden_size
            }
        });
        let indexes = indexes.flatten_all();
        let values = self.embeddings.index_select(0, &indexes);
        values.reshape(final_dims)
    }
}

pub fn embedding(device: &Device, vb: &mut crate::VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get("weight", device)?.dequantize();
    Ok(Embedding::new(embeddings))
}
