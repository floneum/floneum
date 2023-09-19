use std::fmt::Debug;

use crate::embedding::{Embedding, VectorSpace};
use candle_core::Tensor;
use instant_distance::{Builder, HnswMap, Search};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct VectorDB<T, S: VectorSpace> {
    model: HnswMap<Point<S>, T>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T, S: VectorSpace> std::fmt::Debug for VectorDB<T, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorDB").finish()
    }
}

impl<T: Clone + PartialEq + Debug, S: VectorSpace + Sync> VectorDB<T, S>
where
    Self: Sync + Send,
{
    #[tracing::instrument]
    pub fn new(points: Vec<Embedding<S>>, values: Vec<T>) -> Self {
        let points = points.into_iter().map(|e| Point(e)).collect();
        let model = Builder::default().build(points, values);

        VectorDB {
            model,
            _phantom: std::marker::PhantomData,
        }
    }

    #[tracing::instrument]
    pub fn add_embedding(&mut self, embedding: Embedding<S>, value: T) {
        let already_exists = self
            .model
            .search(&Point(embedding.clone()), &mut Search::default())
            .next()
            .filter(|result| result.distance < f32::EPSILON && result.value == &value)
            .is_some();
        if already_exists {
            return;
        }
        let mut new_points = vec![embedding];
        let mut new_values = vec![value];
        for (value_id, point) in self.model.iter() {
            new_points.push(point.0.clone());
            let value = self.model.values[value_id.into_inner() as usize].clone();
            new_values.push(value);
        }
        *self = Self::new(new_points, new_values);
    }

    #[tracing::instrument]
    pub fn get_closest(&self, embedding: Embedding<S>, n: usize) -> Vec<T> {
        let mut search = Search::default();
        self.model
            .search(&Point(embedding), &mut search)
            .take(n)
            .map(|result| result.value.clone())
            .collect()
    }

    #[tracing::instrument]
    pub fn get_within(&self, embedding: Embedding<S>, distance: f32) -> Vec<T> {
        let mut search = Search::default();
        self.model
            .search(&Point(embedding), &mut search)
            .map_while(|result| (result.distance < distance).then(|| result.value.clone()))
            .collect()
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Point<S: VectorSpace>(Embedding<S>);

impl<S: VectorSpace> Clone for Point<S> {
    fn clone(&self) -> Self {
        Point(self.0.clone())
    }
}

impl<S: VectorSpace> instant_distance::Point for Point<S>
where
    Self: Clone + Sync,
{
    fn distance(&self, other: &Self) -> f32 {
        self.try_distance(other).unwrap()
    }
}

impl<S: VectorSpace> Point<S> {
    fn try_distance(&self, other: &Self) -> anyhow::Result<f32> {
        cosine_similarity(self.0.vector(), other.0.vector())
    }
}

fn cosine_similarity(v1: &Tensor, v2: &Tensor) -> anyhow::Result<f32> {
    let sum_ij = (v1 * v2)?.sum_all()?.to_scalar::<f32>()?;
    let sum_i2 = (v1 * v1)?.sum_all()?.to_scalar::<f32>()?;
    let sum_j2 = (v2 * v2)?.sum_all()?.to_scalar::<f32>()?;
    let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
    Ok(1. - cosine_similarity)
}
