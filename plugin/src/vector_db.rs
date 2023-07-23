use std::fmt::Debug;

use crate::plugins::main::types::Embedding;
use instant_distance::{Builder, HnswMap, Search};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct VectorDB<T> {
    model: HnswMap<Point, T>,
}

impl<T> Debug for VectorDB<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorDB").finish()
    }
}

impl<T: Clone + PartialEq + Debug> VectorDB<T> {
    #[tracing::instrument]
    pub fn new(points: Vec<Embedding>, values: Vec<T>) -> Self {
        let points = points.into_iter().map(|e| Point(e.vector)).collect();
        let model = Builder::default().build(points, values);

        VectorDB { model }
    }

    #[tracing::instrument]
    pub fn add_embedding(&mut self, embedding: Embedding, value: T) {
        let already_exists = self
            .model
            .search(&Point(embedding.vector.clone()), &mut Search::default())
            .next()
            .filter(|result| result.distance < f32::EPSILON && result.value == &value)
            .is_some();
        if already_exists {
            return;
        }
        let mut new_points = vec![embedding];
        let mut new_values = vec![value];
        for (value_id, point) in self.model.iter() {
            new_points.push(Embedding {
                vector: point.0.clone(),
            });
            let value = self.model.values[value_id.into_inner() as usize].clone();
            new_values.push(value);
        }
        *self = Self::new(new_points, new_values);
    }

    #[tracing::instrument]
    pub fn get_closest(&self, embedding: Embedding, n: usize) -> Vec<T> {
        let mut search = Search::default();
        self.model
            .search(&Point(embedding.vector), &mut search)
            .take(n)
            .map(|result| result.value.clone())
            .collect()
    }

    #[tracing::instrument]
    pub fn get_within(&self, embedding: Embedding, distance: f32) -> Vec<T> {
        let mut search = Search::default();
        self.model
            .search(&Point(embedding.vector), &mut search)
            .map_while(|result| (result.distance < distance).then(|| result.value.clone()))
            .collect()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Point(Vec<f32>);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        1. - cosine_similarity(&self.0, &other.0)
    }
}

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot_product = dot(v1, v2);
    let magnitude1 = magnitude(v1);
    let magnitude2 = magnitude(v2);

    dot_product / (magnitude1 * magnitude2)
}

fn dot(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(&x, &y)| x * y).sum()
}

fn magnitude(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}
