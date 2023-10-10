use crate::embedding::VectorSpace;
use crate::local::Embedding;
use crate::model::{CreateModel, Embedder};
pub use rbert::Bert;

#[async_trait::async_trait]
impl CreateModel for Bert {
    async fn start() -> Self {
        Self::default()
    }

    fn requires_download() -> bool {
        !Bert::downloaded()
    }
}

#[async_trait::async_trait]
impl Embedder<BertSpace> for Bert {
    async fn embed(&self, input: &str) -> anyhow::Result<Embedding<BertSpace>> {
        let tensor = self
            .load(Default::default())?
            .embed(&[input])?
            .pop()
            .unwrap();
        Ok(Embedding::new(tensor))
    }

    async fn embed_batch(&self, inputs: &[&str]) -> anyhow::Result<Vec<Embedding<BertSpace>>> {
        let tensors = Bert::builder()
            .build()?
            .load(Default::default())?
            .embed(inputs)?;

        let mut embeddings = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            embeddings.push(Embedding::new(tensor));
        }

        Ok(embeddings)
    }
}

pub struct BertSpace;

impl VectorSpace for BertSpace {}
