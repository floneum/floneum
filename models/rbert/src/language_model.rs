pub use crate::Bert;
use kalosm_language_model::Embedding;
use kalosm_language_model::VectorSpace;
use kalosm_language_model::{CreateModel, Embedder};

#[async_trait::async_trait]
impl CreateModel for Bert {
    async fn start() -> Self {
        Self::default()
    }

    fn requires_download() -> bool {
        true
    }
}

#[async_trait::async_trait]
impl Embedder for Bert {
    type VectorSpace = BertSpace;

    async fn embed(&self, input: &str) -> anyhow::Result<Embedding<BertSpace>> {
        let tensor = self.embed_batch_raw(&[input])?.pop().unwrap();
        Ok(Embedding::new(tensor))
    }

    async fn embed_batch(&self, inputs: &[&str]) -> anyhow::Result<Vec<Embedding<BertSpace>>> {
        let tensors = self.embed_batch_raw(inputs)?;

        let mut embeddings = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            embeddings.push(Embedding::new(tensor));
        }

        Ok(embeddings)
    }
}

/// A vector space for BERT sentence embeddings.
pub struct BertSpace;

impl VectorSpace for BertSpace {}
