pub use crate::Bert;
use crate::BertBuilder;
use kalosm_language_model::Embedding;
use kalosm_language_model::ModelLoadingProgress;
use kalosm_language_model::VectorSpace;
use kalosm_language_model::{Embedder, ModelBuilder};

#[async_trait::async_trait]
impl ModelBuilder for BertBuilder {
    type Model = Bert;
    
    async fn start_with_loading_handler(
        self,
        loading_handler: impl FnMut(ModelLoadingProgress) + Send + 'static,
    ) -> anyhow::Result<Self::Model> {
        self.build_with_loading_handler(loading_handler).await
    }

    fn requires_download(&self) -> bool {
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
