pub use crate::Bert;
use crate::BertBuilder;
use crate::Pooling;
use kalosm_common::*;
use kalosm_language_model::Embedding;
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

impl Bert {
    /// Embed a sentence with a specific pooling strategy.
    pub fn embed_with_pooling(
        &self,
        input: &str,
        pooling: Pooling,
    ) -> anyhow::Result<Embedding<BertSpace>> {
        let mut tensors = self.embed_batch_raw(std::iter::once(input), pooling)?;

        Ok(Embedding::new(tensors.pop().unwrap()))
    }

    /// Embed a batch of sentences with a specific pooling strategy.
    pub fn embed_batch_with_pooling(
        &self,
        inputs: &[&str],
        pooling: Pooling,
    ) -> anyhow::Result<Vec<Embedding<BertSpace>>> {
        let tensors = self.embed_batch_raw(inputs.iter().copied(), pooling)?;

        let mut embeddings = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            embeddings.push(Embedding::new(tensor));
        }

        Ok(embeddings)
    }
}

#[async_trait::async_trait]
impl Embedder for Bert {
    type VectorSpace = BertSpace;

    async fn embed(&self, input: &str) -> anyhow::Result<Embedding<BertSpace>> {
        let input = input.to_string();
        let self_clone = self.clone();
        Ok(
            tokio::task::spawn_blocking(move || {
                self_clone.embed_with_pooling(&input, Pooling::CLS)
            })
            .await??,
        )
    }

    async fn embed_batch(&self, inputs: &[&str]) -> anyhow::Result<Vec<Embedding<BertSpace>>> {
        let inputs = inputs
            .iter()
            .map(|input| input.to_string())
            .collect::<Vec<_>>();
        let self_clone = self.clone();
        Ok(tokio::task::spawn_blocking(move || {
            let inputs_borrowed = inputs.iter().map(|s| s.as_str()).collect::<Vec<_>>();
            self_clone.embed_batch_with_pooling(&inputs_borrowed, Pooling::CLS)
        })
        .await??)
    }
}

/// A vector space for BERT sentence embeddings.
pub struct BertSpace;

impl VectorSpace for BertSpace {}
