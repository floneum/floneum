use crate::embedding::VectorSpace;
use crate::local::Embedding;
use crate::model::Embedder;
use rbert::Bert;

pub struct LocalBert {}

#[async_trait::async_trait]
impl Embedder<BertSpace> for LocalBert {
    async fn embed(input: &str) -> anyhow::Result<Embedding<BertSpace>> {
        let tensor = Bert::new(Default::default())?
            .load(Default::default())?
            .embed(&[input])?
            .pop()
            .unwrap();
        Ok(Embedding::new(tensor))
    }

    async fn embed_batch(inputs: &[&str]) -> anyhow::Result<Vec<Embedding<BertSpace>>> {
        let tensors = Bert::new(Default::default())?
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
