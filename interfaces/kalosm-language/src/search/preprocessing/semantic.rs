// Semantic embedding ideas:
// - Word embeddings for first level pass?
// - Keywords?
// - Start with very small chunks, add the most similar sentences together, repeat until you have a good chunk?
// - Try to optimize embedding chunks to be as different as possible from each other?
// - Chunk small sentences with the previous sentence?
// - Chunk parentheses, and quotes together?

use kalosm_language_model::*;
use rbert::{Bert, BertSpace};
use crate::prelude::*;

pub struct SemanticChunkerConfig {
    target_score: f32,
}

impl SemanticChunkerConfig {
    pub fn new(target_score: f32) -> Self {
        Self { target_score }
    }
}

#[derive(Debug, Clone)]
struct SemanticChunk<S: VectorSpace> {
    range: std::ops::Range<usize>,
    sentences: usize,
    embedding: Embedding<S>,
    distance_to_next: Option<f32>,
}

impl<S: VectorSpace> SemanticChunk<S> {
    fn merge_next_score(&self, next: &Self) -> f32 {
        // Score higher if one of the chunks is very short
        let short_chunk_merge_bonus = (10. / self.range.len().min(next.range.len()) as f32).powi(2);
        // Score lower if the chunks are very long
        let large_chunk_penalty = (self.sentences.max(next.sentences) as f32).powf(1.5) / -200.;
        // Score higher if the similarity is high
        let similarity = self.distance_to_next.unwrap();
        similarity + short_chunk_merge_bonus + large_chunk_penalty
    }
}

pub struct SemanticChunker {
    config: SemanticChunkerConfig,
}

impl SemanticChunker {
    pub fn new( config: SemanticChunkerConfig) -> Self {
        Self {  config }
    }
}

impl Chunker for SemanticChunker {
    async fn chunk<E: Embedder + Send>(
        &self,
        document: &Document,
        embedder: &E,
    ) -> anyhow::Result<Vec<Chunk<E::VectorSpace>>> {
        let text = document.body();

        let mut current_chunks = Vec::new();

        // First chunk by sentences
        let chunker = ChunkStrategy::Sentence {
            sentence_count: 1,
            overlap: 0,
        };

        let mut initial_chunks = Vec::new();
        for chunk in chunker.chunk_str(text) {
            let trimmed = text[chunk.clone()].trim();
            if !trimmed.is_empty() {
                current_chunks.push(chunk);
                initial_chunks.push(trimmed);
            }
        }
        
        let embeddings = embedder.embed_batch(&initial_chunks).await?;

        let mut chunks = Vec::new();

        // Find the chain of distances between sequential embeddings
        for (i, chunk) in current_chunks.iter().enumerate() {
            if i == current_chunks.len() - 1 {
                chunks.push(SemanticChunk {
                    range: chunk.clone(),
                    sentences: 1,
                    embedding: embeddings[i].clone(),
                    distance_to_next: None,
                });
                break;
            }
            let first = &embeddings[i];
            let second = &embeddings[i + 1];
            let distance_to_next = first.cosine_similarity(second);
            let chunk = SemanticChunk {
                range: chunk.clone(),
                sentences: 1,
                embedding: first.clone(),
                distance_to_next: Some(distance_to_next),
            };
            chunks.push(chunk);
        }


        // Now loop until we have the right number of chunks merging the two closest chunks
        // Find the lowest distance chunk
        while let Some((index, first_chunk)) = chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.distance_to_next.is_some())
            .max_by(|(index, c1), (index2, c2)| {
                // Score higher if the similarity is high or if the text size of both is small
                let c1_score = c1.merge_next_score(&chunks[index + 1]);
                let c2_score = c2.merge_next_score(&chunks[index2 + 1]);

                c1_score.partial_cmp(&c2_score).unwrap()
            })
        {
            let second_chunk = &chunks[index + 1];

            let highest_similarity = first_chunk.merge_next_score(second_chunk);
            if highest_similarity < self.config.target_score {
                break;
            }

            // Merge the two chunks
            let range = first_chunk.range.start..second_chunk.range.end;
            let sentences = first_chunk.sentences + second_chunk.sentences;

            let new_text = text[range.clone()].trim();
            let embedding = embedder.embed(new_text).await?;

            // Calculate the distance to the next chunk
            let distance_to_next = chunks
                .get(index + 2)
                .map(|chunk_after_merge| embedding.cosine_similarity(&chunk_after_merge.embedding));

            // Recalculate the distance to the previous chunk
            if let Some(prev_chunk) = index.checked_sub(1).and_then(|index| chunks.get_mut(index)) {
                let distance_to_prev = prev_chunk.embedding.cosine_similarity(&embedding);
                prev_chunk.distance_to_next = Some(distance_to_prev);
            }

            let new_chunk = SemanticChunk {
                range,
                sentences,
                embedding,
                distance_to_next,
            };

            // Remove the last chunk
            chunks.remove(index + 1);

            // Add our merged chunk to the chunks
            chunks[index] = new_chunk;
        }

        let mut final_chunks = Vec::new();
        for chunk in chunks {
            let SemanticChunk { range, embedding, .. } = chunk;
            final_chunks.push(Chunk{
                byte_range: range,
                embeddings: vec![embedding],
            });
        }

        Ok(final_chunks)
    }
}