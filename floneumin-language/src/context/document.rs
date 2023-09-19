use std::ops::Range;

use crate::embedding::{Embedding, VectorSpace};

enum ChunkStrategy {
    Paragraph,
    Sentence,
    Words(usize),
}

pub struct Document<S: VectorSpace> {
    raw: String,
    chunks: Vec<Chunk<S>>,
}

impl<S: VectorSpace> Document<S> {
    pub fn new(raw: String) -> Self {
        let chunks = Vec::new();
        Self { raw, chunks }
    }
}

pub struct Chunk<S: VectorSpace> {
    byte_range: Range<usize>,
    embedding: Embedding<S>,
}
