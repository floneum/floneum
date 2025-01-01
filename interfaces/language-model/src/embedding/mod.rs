use std::ops::{Add, Div, Mul, Sub};

use candle_core::{Device, Tensor};
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "cache")]
mod cache;
#[cfg(feature = "cache")]
pub use cache::*;
mod model;
pub use model::*;
mod into_embedding;
pub use into_embedding::*;

#[doc = include_str!("../../docs/embedding.md")]
pub struct Embedding {
    embedding: Tensor,
}

impl Embedding {
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

impl Add for Embedding {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let embedding = (self.embedding + other.embedding).unwrap();
        Embedding { embedding }
    }
}

impl Sub for Embedding {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let embedding = (self.embedding - other.embedding).unwrap();
        Embedding { embedding }
    }
}

impl Mul<f64> for Embedding {
    type Output = Self;

    fn mul(self, other: f64) -> Self::Output {
        let embedding = (self.embedding * other).unwrap();
        Embedding { embedding }
    }
}

impl Div<f64> for Embedding {
    type Output = Self;

    fn div(self, other: f64) -> Self::Output {
        let embedding = (self.embedding / other).unwrap();
        Embedding { embedding }
    }
}

impl std::fmt::Debug for Embedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedding")
            .field("embedding", &self.embedding)
            .finish()
    }
}

impl Clone for Embedding {
    fn clone(&self) -> Self {
        Embedding {
            embedding: self.embedding.clone(),
        }
    }
}

#[cfg(feature = "serde")]
impl Serialize for Embedding {
    fn serialize<Ser: Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        let values = self.to_vec();
        values.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Embedding {
    fn deserialize<Des: Deserializer<'de>>(deserializer: Des) -> Result<Self, Des::Error> {
        let values: Vec<f32> = Deserialize::deserialize(deserializer)?;
        Ok(Embedding::from(values))
    }
}

#[cfg(feature = "cache")]
#[test]
fn embedding_serialization() {
    let embedding = Embedding::<UnknownVectorSpace>::from(vec![0.0, 1.0, 2.0, 3.0]);
    let first_float: Vec<f32> = embedding.vector().to_vec1().unwrap();
    let bytes = postcard::to_stdvec(&embedding).unwrap();
    let embedding: Embedding = postcard::from_bytes(&bytes).unwrap();
    let second_float: Vec<f32> = embedding.vector().to_vec1().unwrap();
    assert_eq!(first_float, second_float);
}

impl<I: IntoIterator<Item = f32>> From<I> for Embedding {
    fn from(iter: I) -> Self {
        let data: Vec<f32> = iter.into_iter().collect();
        let shape = [data.len()];
        Embedding {
            embedding: Tensor::from_vec(data, &shape, &Device::Cpu).unwrap(),
        }
    }
}

impl Embedding {
    /// Create a new embedding from a tensor.
    pub fn new(embedding: Tensor) -> Self {
        let flattened = embedding.flatten_all().unwrap();
        Embedding {
            embedding: flattened,
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
