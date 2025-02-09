use std::ops::{Add, Div, Mul, Sub};

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
    embedding: Box<[f32]>,
}

impl Embedding {
    /// Compute the cosine similarity between this embedding and another embedding.
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let sum_ij = self
            .embedding
            .iter()
            .zip(other.embedding.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
        let sum_i2 = other.embedding.iter().map(|a| a * a).sum::<f32>();
        let sum_j2 = self.embedding.iter().map(|a| a * a).sum::<f32>();
        sum_ij / (sum_i2 * sum_j2).sqrt()
    }
}

impl Add for Embedding {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let embedding = self
            .embedding
            .iter()
            .zip(other.embedding.iter())
            .map(|(a, b)| a + b)
            .collect();
        Embedding { embedding }
    }
}

impl Sub for Embedding {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let embedding = self
            .embedding
            .iter()
            .zip(other.embedding.iter())
            .map(|(a, b)| a - b)
            .collect();
        Embedding { embedding }
    }
}

impl Mul<f32> for Embedding {
    type Output = Self;

    fn mul(self, other: f32) -> Self::Output {
        let embedding = self.embedding.iter().map(|a| *a * other).collect();
        Embedding { embedding }
    }
}

impl Div<f32> for Embedding {
    type Output = Self;

    fn div(self, other: f32) -> Self::Output {
        let embedding = self.embedding.iter().map(|a| *a / other).collect();
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
        let values = self.vector();
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
    let embedding = Embedding::from(vec![0.0, 1.0, 2.0, 3.0]);
    let first_float: Vec<f32> = embedding.vector().to_vec();
    let bytes = postcard::to_stdvec(&embedding).unwrap();
    let embedding: Embedding = postcard::from_bytes(&bytes).unwrap();
    let second_float: Vec<f32> = embedding.vector().to_vec();
    assert_eq!(first_float, second_float);
}

impl<I: IntoIterator<Item = f32>> From<I> for Embedding {
    fn from(iter: I) -> Self {
        Embedding {
            embedding: iter.into_iter().collect(),
        }
    }
}

impl Embedding {
    /// Create a new embedding from a tensor.
    pub fn new(embedding: Box<[f32]>) -> Self {
        Embedding { embedding }
    }

    /// Get the tensor that represents this embedding.
    pub fn vector(&self) -> &[f32] {
        &self.embedding
    }
}
