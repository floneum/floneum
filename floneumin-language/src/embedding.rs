use std::marker::PhantomData;

use candle_core::{Device, Tensor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub trait VectorSpace {}

pub struct Embedding<S: VectorSpace> {
    embedding: Tensor,
    model: PhantomData<S>,
}

impl<S: VectorSpace> std::fmt::Debug for Embedding<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedding")
            .field("embedding", &self.embedding)
            .field("model", &std::any::type_name::<S>())
            .finish()
    }
}

impl<S: VectorSpace> Clone for Embedding<S> {
    fn clone(&self) -> Self {
        Embedding {
            embedding: self.embedding.clone(),
            model: PhantomData,
        }
    }
}

impl<S: VectorSpace> Serialize for Embedding<S> {
    fn serialize<Ser: Serializer>(&self, _serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        todo!()
    }
}

impl<'de, S: VectorSpace> Deserialize<'de> for Embedding<S> {
    fn deserialize<Des: Deserializer<'de>>(_deserializer: Des) -> Result<Self, Des::Error> {
        todo!()
    }
}

impl<S: VectorSpace, I: IntoIterator<Item = f32>> From<I> for Embedding<S> {
    fn from(iter: I) -> Self {
        let data: Vec<f32> = iter.into_iter().collect();
        let shape = [data.len()];
        Embedding {
            embedding: Tensor::from_vec(data, &shape, &Device::Cpu).unwrap(),
            model: PhantomData,
        }
    }
}

impl<S: VectorSpace> Embedding<S> {
    pub fn new(embedding: Tensor) -> Self {
        Embedding {
            embedding,
            model: PhantomData,
        }
    }

    pub fn vector(&self) -> &Tensor {
        &self.embedding
    }
}

pub fn get_embeddings<S: VectorSpace>(model: &dyn llm::Model, embed: &str) -> Embedding<S> {
    let mut session = model.start_session(Default::default());
    let mut output_request = llm::OutputRequest {
        all_logits: None,
        embeddings: Some(Vec::new()),
    };
    let _ = session.feed_prompt(model, embed, &mut output_request, |_| {
        Ok::<_, std::convert::Infallible>(llm::InferenceFeedback::Halt)
    });
    Embedding::from(output_request.embeddings.unwrap())
}
