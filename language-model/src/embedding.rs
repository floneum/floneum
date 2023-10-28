use std::marker::PhantomData;

use candle_core::{Device, Tensor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// An untyped vector space that is not associated with a model. This can be used to erase the vector type from an embedding.
pub struct UnknownVectorSpace;

impl VectorSpace for UnknownVectorSpace {}

/// The type of a vector space marks what model the vector space is from. You should only combine vector spaces that come from the same model.
///
/// For example, the Llama model has a different vector space than the Bert model. Comparing these two vector spaces would not make sense because different parts of each vector encode different information. This trait allows you to mark an embedding with the type of vector space it comes from to avoid problems combing vector spaces.
///
/// If you want to cast an embedding from one vector space to another, you can use the [`Embedding::cast`] method. You can cast to the UnknownVectorSpace to erase the vector space type.
pub trait VectorSpace {}

/// An embedding represents something about the meaning of data. It can be used to compare the meaning of different pieces of data, cluster data, or as input to a machine learning model.
pub struct Embedding<S: VectorSpace> {
    embedding: Tensor,
    model: PhantomData<S>,
}

impl<S: VectorSpace> Embedding<S> {
    /// Compute the cosine similarity between this embedding and another embedding.
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let sum_ij = (&other.embedding * &self.embedding)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let sum_i2 = (&other.embedding * &other.embedding)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let sum_j2 = (&self.embedding * &self.embedding)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        sum_ij / (sum_i2 * sum_j2).sqrt()
    }
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

impl<S1: VectorSpace> Embedding<S1> {
    /// Cast this embedding to a different vector space.
    pub fn cast<S2: VectorSpace>(self) -> Embedding<S2> {
        Embedding {
            embedding: self.embedding,
            model: PhantomData,
        }
    }
}

impl<S: VectorSpace> Embedding<S> {
    /// Create a new embedding from a tensor.
    pub fn new(embedding: Tensor) -> Self {
        Embedding {
            embedding,
            model: PhantomData,
        }
    }

    /// Get the tensor that represents this embedding.
    pub fn vector(&self) -> &Tensor {
        &self.embedding
    }

    /// Get the tensor that represents this embedding as a Vec of floats.
    pub fn to_vec(&self) -> Vec<f32> {
        self.embedding.to_vec1::<f32>().unwrap()
    }
}
