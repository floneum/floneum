use std::{
    marker::PhantomData,
    ops::{Add, Div, Mul, Sub},
};

use candle_core::{Device, Tensor};
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "cache")]
mod cache;
#[cfg(feature = "cache")]
pub use cache::*;
mod model;
pub use model::*;

/// An untyped vector space that is not associated with a model. This can be used to erase the vector type from an embedding.
pub struct UnknownVectorSpace;

impl VectorSpace for UnknownVectorSpace {}

/// The type of a vector space marks what model the vector space is from. You should only combine vector spaces that come from the same model.
///
/// For example, the Llama model has a different vector space than the Bert model. Comparing these two vector spaces would not make sense because different parts of each vector encode different information. This trait allows you to mark an embedding with the type of vector space it comes from to avoid problems combing vector spaces.
///
/// If you want to cast an embedding from one vector space to another, you can use the [`Embedding::cast`] method. You can cast to the UnknownVectorSpace to erase the vector space type.
pub trait VectorSpace: Sync + Send + 'static {}

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

impl<S: VectorSpace> Add for Embedding<S> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let embedding = (self.embedding + other.embedding).unwrap();
        Embedding {
            embedding,
            model: PhantomData,
        }
    }
}

impl<S: VectorSpace> Sub for Embedding<S> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let embedding = (self.embedding - other.embedding).unwrap();
        Embedding {
            embedding,
            model: PhantomData,
        }
    }
}

impl<S: VectorSpace> Mul<f64> for Embedding<S> {
    type Output = Self;

    fn mul(self, other: f64) -> Self::Output {
        let embedding = (self.embedding * other).unwrap();
        Embedding {
            embedding,
            model: PhantomData,
        }
    }
}

impl<S: VectorSpace> Div<f64> for Embedding<S> {
    type Output = Self;

    fn div(self, other: f64) -> Self::Output {
        let embedding = (self.embedding / other).unwrap();
        Embedding {
            embedding,
            model: PhantomData,
        }
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

#[cfg(feature = "serde")]
impl<S: VectorSpace> Serialize for Embedding<S> {
    fn serialize<Ser: Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        let bytes = safetensors::tensor::serialize([("data", &self.embedding)], &None)
            .map_err(|e| serde::ser::Error::custom(e.to_string()))?;

        bytes.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, S: VectorSpace> Deserialize<'de> for Embedding<S> {
    fn deserialize<Des: Deserializer<'de>>(deserializer: Des) -> Result<Self, Des::Error> {
        let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
        let tensor = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| serde::de::Error::custom(e.to_string()))?;
        let tensor = tensor
            .tensor("data")
            .map_err(|e| serde::de::Error::custom(e.to_string()))?;
        let shape = tensor.shape();
        let data = tensor.data();
        let dtype = candle_core::DType::try_from(tensor.dtype())
            .map_err(|e| serde::de::Error::custom(e.to_string()))?;
        let embedding = Tensor::from_raw_buffer(data, dtype, shape, &Device::Cpu).unwrap();
        Ok(Embedding {
            embedding,
            model: PhantomData,
        })
    }
}

#[cfg(feature = "cache")]
#[test]
fn embedding_serialization() {
    let embedding = Embedding::<UnknownVectorSpace>::from(vec![0.0, 1.0, 2.0, 3.0]);
    let first_float: Vec<f32> = embedding.vector().to_vec1().unwrap();
    let bytes = postcard::to_stdvec(&embedding).unwrap();
    let embedding: Embedding<UnknownVectorSpace> = postcard::from_bytes(&bytes).unwrap();
    let second_float: Vec<f32> = embedding.vector().to_vec1().unwrap();
    assert_eq!(first_float, second_float);
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
        let flattened = embedding.flatten_all().unwrap();
        Embedding {
            embedding: flattened,
            model: PhantomData,
        }
    }

    /// Get the tensor that represents this embedding.
    pub fn vector(&self) -> &Tensor {
        &self.embedding
    }

    /// Get the tensor that represents this embedding as a Vec of floats.
    pub fn to_vec(&self) -> Vec<f32> {
        if self.embedding.dims().len() != 1 {
            self.embedding
                .flatten_to(1)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        } else {
            self.embedding.to_vec1::<f32>().unwrap()
        }
    }
}
