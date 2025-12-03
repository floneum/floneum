use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use surrealdb::{Connection, RecordIdKey, Surreal};

use kalosm_language::prelude::*;

use crate::language::{
    DocumentTable, DocumentTableBuilder, DocumentTableCreationError, DocumentTableSearchError,
};
use crate::EmbeddedIndexedTableError;

#[derive(Debug, thiserror::Error)]
pub enum HybridSearchError {
    #[error("Failed to enable hybrid search: {0}")]
    SetupError(String),

    #[error("Keyword search failed: {0}")]
    KeywordSearchError(String),

    #[error("Database error: {0}")]
    DatabaseError(#[from] surrealdb::Error),

    #[error("Document table creation failed: {0}")]
    TableCreationError(#[from] DocumentTableCreationError),

    #[error("No results found")]
    NoResults,

    #[error("Semantic search failed: {0}")]
    SemanticSearchError(#[from] EmbeddedIndexedTableError),

    #[error("Semantic query failed: {0}")]
    SemanticQueryError(String),
}

impl<E> From<DocumentTableSearchError<E>> for HybridSearchError
where
    E: std::fmt::Debug + std::fmt::Display + 'static,
{
    fn from(error: DocumentTableSearchError<E>) -> Self {
        match error {
            DocumentTableSearchError::SearchTable(err) => Self::SemanticSearchError(err),
            DocumentTableSearchError::EmbedQuery(err) => {
                Self::SemanticQueryError(format!("Embedding failed: {}", err))
            }
        }
    }
}

/// Normalize scores to 0-1 range using max normalization
///
/// Returns None if all scores are zero or negative
pub fn normalize_scores(scores: &[f32]) -> Option<Vec<f32>> {
    if scores.is_empty() {
        return Some(vec![]);
    }

    let max_score = scores.iter().copied().fold(0.0f32, f32::max);

    if max_score <= 0.0 {
        // All scores are zero or negative - cannot normalize
        return None;
    }

    Some(scores.iter().map(|&s| s / max_score).collect())
}

/// Convert distance to similarity score (1 - normalized_distance)
///
/// Returns None if max_distance is zero or negative
pub fn distance_to_similarity(distance: f32, max_distance: f32) -> Option<f32> {
    if max_distance <= 0.0 {
        return None;
    }

    Some(1.0 - (distance / max_distance).min(1.0))
}

/// Calculate weighted combination of two scores
pub fn calculate_weighted_score(
    semantic_score: f32,
    keyword_score: f32,
    semantic_weight: f32,
    keyword_weight: f32,
) -> f32 {
    semantic_score * semantic_weight + keyword_score * keyword_weight
}

/// Calculate RRF score for a given rank
pub fn calculate_rrf_score(rank: usize, k: f32) -> f32 {
    1.0 / (k + rank as f32 + 1.0)
}

/// Combine RRF scores from multiple sources
pub fn combine_rrf_scores(
    semantic_rank: Option<usize>,
    keyword_rank: Option<usize>,
    k: f32,
) -> f32 {
    let semantic_score = semantic_rank
        .map(|r| calculate_rrf_score(r, k))
        .unwrap_or(0.0);
    let keyword_score = keyword_rank
        .map(|r| calculate_rrf_score(r, k))
        .unwrap_or(0.0);
    semantic_score + keyword_score
}

/// Create a hash-based key from record content for deduplication
pub fn create_content_hash<T: Hash>(record: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    record.hash(&mut hasher);
    hasher.finish()
}

/// Merge and deduplicate results from two sources
pub fn merge_results<R>(
    semantic_map: HashMap<u64, (RecordIdKey, R, f32)>,
    keyword_map: HashMap<u64, (R, f32)>,
    combine_scores: impl Fn(f32, f32) -> f32,
) -> Vec<(u64, RecordIdKey, R, f32, f32, f32)>
where
    R: Clone,
{
    let mut combined_results = Vec::new();
    let mut seen_keys = HashSet::new();

    let all_keys: Vec<u64> = semantic_map
        .keys()
        .chain(keyword_map.keys())
        .copied()
        .collect();

    for key in all_keys {
        if seen_keys.insert(key) {
            let semantic_entry = semantic_map.get(&key);
            let keyword_entry = keyword_map.get(&key);

            let semantic_score = semantic_entry.map(|(_, _, s)| *s).unwrap_or(0.0);
            let keyword_score = keyword_entry.map(|(_, s)| *s).unwrap_or(0.0);

            let combined_score = combine_scores(semantic_score, keyword_score);

            if let Some((record_id, record, _)) = semantic_entry {
                combined_results.push((
                    key,
                    record_id.clone(),
                    record.clone(),
                    combined_score,
                    semantic_score,
                    keyword_score,
                ));
            } else if let Some((record, _)) = keyword_entry {
                combined_results.push((
                    key,
                    RecordIdKey::from("keyword_only"),
                    record.clone(),
                    combined_score,
                    semantic_score,
                    keyword_score,
                ));
            }
        }
    }

    combined_results
}

pub trait HybridSearchSetupExt<C: Connection> {
    /// Enable hybrid search on a table by creating the necessary full-text index.
    ///
    /// # Arguments
    /// * `table_name` - The name of the table to enable hybrid search on
    /// * `field_name` - The field to index for full-text search
    async fn enable_hybrid_search(
        &self,
        table_name: &str,
        field_name: &str,
    ) -> Result<(), HybridSearchError>;
}

impl<C: Connection> HybridSearchSetupExt<C> for Surreal<C> {
    async fn enable_hybrid_search(
        &self,
        table_name: &str,
        field_name: &str,
    ) -> Result<(), HybridSearchError> {
        let index_name = format!("{}_fulltext_idx", table_name);

        let query = format!(
            "DEFINE INDEX {} ON TABLE {} FIELDS {} SEARCH ANALYZER simple BM25",
            index_name, table_name, field_name
        );

        self.query(query).await?;
        Ok(())
    }
}

/// Builder for creating document tables with hybrid search capability
pub struct HybridDocumentTableBuilder<C: Connection, E = Bert, K: Chunker = SemanticChunker> {
    inner: DocumentTableBuilder<C, E, K>,
    field_name: String,
}

impl<C: Connection, E, K: Chunker> HybridDocumentTableBuilder<C, E, K> {
    /// Create a new hybrid document table builder
    ///
    /// # Arguments
    /// * `inner` - The base document table builder
    /// * `field_name` - The field to index for full-text search (commonly "body", "content", "text")
    pub(crate) fn new(inner: DocumentTableBuilder<C, E, K>, field_name: impl Into<String>) -> Self {
        Self {
            inner,
            field_name: field_name.into(),
        }
    }

    /// Specify which field to use for full-text search
    pub fn with_search_field(mut self, field: impl Into<String>) -> Self {
        self.field_name = field.into();
        self
    }

    /// Set the location of the vector database
    pub fn at(mut self, location: impl AsRef<std::path::Path>) -> Self {
        self.inner = self.inner.at(location);
        self
    }

    /// Set the embedding model
    pub fn with_embedding_model<E2>(self, model: E2) -> HybridDocumentTableBuilder<C, E2, K> {
        HybridDocumentTableBuilder {
            inner: self.inner.with_embedding_model(model),
            field_name: self.field_name,
        }
    }

    /// Set the chunking strategy
    pub fn with_chunker<K2: Chunker>(self, chunker: K2) -> HybridDocumentTableBuilder<C, E, K2> {
        HybridDocumentTableBuilder {
            inner: self.inner.with_chunker(chunker),
            field_name: self.field_name,
        }
    }

    /// Build the document table with hybrid search enabled
    pub async fn build<R: Serialize + DeserializeOwned>(
        self,
    ) -> Result<DocumentTable<C, R, E, K>, HybridSearchError>
    where
        E: Embedder,
    {
        let table = self.inner.build().await?;

        // Enable hybrid search on the table
        table
            .table()
            .db()
            .enable_hybrid_search(table.table().table(), &self.field_name)
            .await?;

        Ok(table)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchResult<Doc> {
    pub record: Doc,
    pub id: RecordIdKey,
    pub score: f32,
    pub semantic_score: f32,
    pub keyword_score: f32,
}

pub struct HybridSearchBuilder<'a, Conn: Connection, Doc, Model: Embedder, Chkr: Chunker> {
    table: &'a DocumentTable<Conn, Doc, Model, Chkr>,
    query: String,
    results: usize,
    semantic_weight: f32,
    keyword_weight: f32,
    field_name: String,
}

#[derive(Deserialize)]
struct KeywordResult<R> {
    #[serde(flatten)]
    record: R,
    pub id: RecordIdKey,
    keyword_score: f32,
}

impl<'a, C: Connection, R, M: Embedder, K: Chunker> HybridSearchBuilder<'a, C, R, M, K> {
    pub(crate) fn new(
        table: &'a DocumentTable<C, R, M, K>,
        query: impl Into<String>,
        field_name: impl Into<String>,
    ) -> Self {
        Self {
            table,
            query: query.into(),
            results: 10,
            semantic_weight: 0.7,
            keyword_weight: 0.3,
            field_name: field_name.into(),
        }
    }

    /// Set the number of results to return (default: 10)
    pub fn with_results(mut self, results: usize) -> Self {
        self.results = results;
        self
    }

    /// Set the weight of the semantic search (default: 0.7)
    pub fn with_semantic_weight(mut self, weight: f32) -> Self {
        self.semantic_weight = weight;
        self
    }

    /// Set the weight of the keyword search (default: 0.3)
    pub fn with_keyword_weight(mut self, weight: f32) -> Self {
        self.keyword_weight = weight;
        self
    }

    /// Execute the hybrid search using weighted combination
    pub async fn run_weighted(self) -> Result<Vec<HybridSearchResult<R>>, HybridSearchError>
    where
        R: Serialize + DeserializeOwned + Clone + AsRef<Document> + Send + Sync + Hash,
        <M as Embedder>::Error: std::fmt::Debug + std::fmt::Display + 'static,
    {
        // Perform semantic search
        let semantic_results = self
            .table
            .search(self.query.clone())
            .with_results(self.results * 2)
            .run()
            .await?;

        // Perform keyword search using SurrealDB full-text search
        let keyword_query = format!(
            "SELECT *, search::score(1) AS keyword_score FROM {} WHERE {} @@ $query ORDER BY keyword_score DESC LIMIT $limit",
            self.table.table().table(),
            self.field_name
        );

        let keyword_results: Vec<KeywordResult<R>> = self
            .table
            .table()
            .db()
            .query(keyword_query)
            .bind(("query", self.query.clone()))
            .bind(("limit", self.results * 2))
            .await?
            .take(0)
            .map_err(|e| HybridSearchError::KeywordSearchError(e.to_string()))?;

        // Normalize scores
        let max_semantic = semantic_results.first().map(|r| r.distance).unwrap_or(1.0);

        let max_keyword = keyword_results
            .iter()
            .map(|r| r.keyword_score)
            .fold(0.0f32, f32::max);

        // Build score maps
        let mut semantic_map: HashMap<u64, (RecordIdKey, R, f32)> = HashMap::new();
        for result in semantic_results {
            let normalized = distance_to_similarity(result.distance, max_semantic).unwrap_or(0.0); // If max_semantic is 0, treat as no similarity
            let key = create_content_hash(&result.record);
            semantic_map.insert(key, (result.record_id, result.record, normalized));
        }

        let mut keyword_map: HashMap<u64, (R, f32)> = HashMap::new();
        for result in keyword_results {
            let normalized = if max_keyword > 0.0 {
                result.keyword_score / max_keyword
            } else {
                0.0 // All keyword scores are 0
            };
            let key = create_content_hash(&result.record);
            keyword_map.insert(key, (result.record, normalized));
        }

        // Merge using helper function
        let combined = merge_results(semantic_map, keyword_map, |sem, key| {
            calculate_weighted_score(sem, key, self.semantic_weight, self.keyword_weight)
        });

        // Convert to result type and sort
        let mut results: Vec<HybridSearchResult<R>> = combined
            .into_iter()
            .map(
                |(_, id, record, score, semantic_score, keyword_score)| HybridSearchResult {
                    record,
                    id,
                    score,
                    semantic_score,
                    keyword_score,
                },
            )
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(self.results);

        Ok(results)
    }

    /// Execute the hybrid search using Reciprocal Rank Fusion (RRF)
    ///
    /// # Arguments
    /// * `k` - RRF constant, typically 60
    pub async fn run_with_rrf(self, k: f32) -> Result<Vec<HybridSearchResult<R>>, HybridSearchError>
    where
        R: Serialize + DeserializeOwned + Clone + AsRef<Document> + Send + Sync + Hash,
        <M as Embedder>::Error: std::fmt::Debug + std::fmt::Display + 'static,
    {
        // Perform semantic search
        let semantic_results = self
            .table
            .search(self.query.clone())
            .with_results(self.results * 2)
            .run()
            .await?;

        // Perform keyword search
        let keyword_query = format!(
            "SELECT *, search::score(1) AS keyword_score FROM {} WHERE {} @@ $query ORDER BY keyword_score DESC LIMIT $limit",
            self.table.table().table(),
            self.field_name
        );

        let keyword_results: Vec<KeywordResult<R>> = self
            .table
            .table()
            .db()
            .query(keyword_query)
            .bind(("query", self.query.clone()))
            .bind(("limit", self.results * 2))
            .await?
            .take(0)
            .map_err(|e| HybridSearchError::KeywordSearchError(e.to_string()))?;

        // Build rank maps
        let mut semantic_ranks: HashMap<u64, usize> = HashMap::new();
        for (rank, result) in semantic_results.iter().enumerate() {
            let key = create_content_hash(&result.record);
            semantic_ranks.insert(key, rank);
        }

        let mut keyword_ranks: HashMap<u64, usize> = HashMap::new();
        for (rank, result) in keyword_results.iter().enumerate() {
            let key = create_content_hash(&result.record);
            keyword_ranks.insert(key, rank);
        }

        // Collect all unique keys
        let all_keys: HashSet<u64> = semantic_ranks
            .keys()
            .chain(keyword_ranks.keys())
            .copied()
            .collect();

        // Build result map with RRF scores
        let mut rrf_results: HashMap<u64, (R, f32, f32, f32, RecordIdKey)> = HashMap::new();

        for key in all_keys {
            let semantic_rank = semantic_ranks.get(&key).copied();
            let keyword_rank = keyword_ranks.get(&key).copied();

            let rrf_score = combine_rrf_scores(semantic_rank, keyword_rank, k);

            // Get normalized scores for display
            let semantic_score = semantic_rank
                .map(|r| {
                    let result = &semantic_results[r];
                    1.0 - result.distance
                })
                .unwrap_or(0.0);

            let keyword_score = keyword_rank
                .map(|r| keyword_results[r].keyword_score)
                .unwrap_or(0.0);

            // Get record and ID (prefer semantic)
            if let Some(rank) = semantic_rank {
                let result = &semantic_results[rank];
                rrf_results.insert(
                    key,
                    (
                        result.record.clone(),
                        rrf_score,
                        semantic_score,
                        keyword_score,
                        result.record_id.clone(),
                    ),
                );
            } else if let Some(rank) = keyword_rank {
                let result = &keyword_results[rank];
                rrf_results.insert(
                    key,
                    (
                        result.record.clone(),
                        rrf_score,
                        semantic_score,
                        keyword_score,
                        RecordIdKey::from("keyword_only"),
                    ),
                );
            }
        }

        // Convert to result type and sort
        let mut results: Vec<HybridSearchResult<R>> = rrf_results
            .into_iter()
            .map(
                |(_, (record, score, semantic, keyword, id))| HybridSearchResult {
                    record,
                    id,
                    score,
                    semantic_score: semantic,
                    keyword_score: keyword,
                },
            )
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(self.results);

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use surrealdb::RecordIdKey;

    #[test]
    fn test_normalize_scores() {
        // Test Empty
        let scores = vec![];
        let normalized = normalize_scores(&scores);
        assert_eq!(normalized, Some(vec![]));

        // Test Single Score
        let scores = vec![5.0];
        let normalized = normalize_scores(&scores);
        assert_eq!(normalized, Some(vec![1.0]));

        // Test zeros
        let scores = vec![0.0, 0.0, 0.0];
        let normalized = normalize_scores(&scores);

        // Should return None - cannot normalize all zeros
        assert_eq!(normalized, None);

        // Test negatives
        let scores = vec![-1.0, -2.0, -3.0];
        let normalized = normalize_scores(&scores);

        // All negative scores should return None
        assert_eq!(normalized, None);
    }

    #[test]
    fn test_normalize_scores_basic() {
        let scores = vec![10.0, 5.0, 2.5];
        let normalized = normalize_scores(&scores).unwrap();

        // Max is 10.0, so: [10/10, 5/10, 2.5/10] = [1.0, 0.5, 0.25]
        assert!((normalized[0] - 1.0).abs() < 0.001);
        assert!((normalized[1] - 0.5).abs() < 0.001);
        assert!((normalized[2] - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_normalize_scores_range() {
        let scores = vec![100.0, 50.0, 25.0, 10.0];
        let normalized = normalize_scores(&scores).unwrap();

        // All should be in [0, 1]
        assert!(normalized.iter().all(|&n| n >= 0.0 && n <= 1.0));
        // Max should be 1.0
        assert!((normalized[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_distance_to_similarity() {
        // Test zero distance - should give max similarity
        match distance_to_similarity(0.0, 10.0) {
            Some(similarity) => assert!((similarity - 1.0).abs() < 0.001),
            None => panic!("Expected Some for valid inputs (0.0, 10.0)"),
        }

        // Test max distance - should give zero similarity
        let max = 5.0;
        match distance_to_similarity(max, max) {
            Some(similarity) => {
                println!("Similarity for ({}, {}): {}", max, max, similarity);
                assert!((similarity - 0.0).abs() < 0.001);
            }
            None => panic!("Expected Some for valid inputs ({}, {})", max, max),
        }

        // Test half distance
        match distance_to_similarity(5.0, 10.0) {
            Some(similarity) => assert!((similarity - 0.5).abs() < 0.001),
            None => panic!("Expected Some for valid inputs (5.0, 10.0)"),
        }

        // Test zero max distance - should return None
        assert!(
            distance_to_similarity(0.0, 0.0).is_none(),
            "Expected None for zero max_distance"
        );

        // Test negative max distance - should return None
        assert!(
            distance_to_similarity(1.0, -5.0).is_none(),
            "Expected None for negative max_distance"
        );
    }

    #[test]
    fn test_weighted_score() {
        // Test pure semantic
        let score = calculate_weighted_score(0.9, 0.3, 1.0, 0.0);
        assert!((score - 0.9).abs() < 0.001);

        // Test pure keyword
        let score = calculate_weighted_score(0.9, 0.3, 0.0, 1.0);
        assert!((score - 0.3).abs() < 0.001);

        // Test balanced score
        let score = calculate_weighted_score(0.8, 0.4, 0.5, 0.5);
        // (0.8 + 0.4) / 2 = 0.6
        assert!((score - 0.6).abs() < 0.001);

        // Test zero weights - should return zero
        let score = calculate_weighted_score(0.0, 0.0, 0.7, 0.3);
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_rrf_score() {
        let k = 60.0;

        // Test rank zero
        let score = calculate_rrf_score(0, k);
        // 1 / (60 + 0 + 1) = 1/61 ≈ 0.0164
        assert!((score - 0.0164).abs() < 0.001);

        // Test increasing rank
        let score0 = calculate_rrf_score(0, k);
        let score1 = calculate_rrf_score(1, k);
        let score2 = calculate_rrf_score(2, k);

        // Scores should decrease with rank
        assert!(score0 > score1);
        assert!(score1 > score2);

        // Test with different k params
        let rank = 0;
        let score_k10 = calculate_rrf_score(rank, 10.0);
        let score_k60 = calculate_rrf_score(rank, 60.0);
        let score_k100 = calculate_rrf_score(rank, 100.0);

        // Smaller k gives higher scores
        assert!(score_k10 > score_k60);
        assert!(score_k60 > score_k100);

        // Test high rank
        let score = calculate_rrf_score(1000, k);

        // Even at high ranks, score should be positive
        assert!(score > 0.0);
        // But very small
        assert!(score < 0.01);
    }

    #[test]
    fn test_combine_rrf() {
        let k = 60.0;
        // Test for both scores present
        let score = combine_rrf_scores(Some(0), Some(5), k);

        let expected = calculate_rrf_score(0, k) + calculate_rrf_score(5, k);
        assert!((score - expected).abs() < 0.0001);

        // Test for only semantic
        let score = combine_rrf_scores(Some(3), None, k);

        let expected = calculate_rrf_score(3, k);
        assert!((score - expected).abs() < 0.0001);

        // Test for only keyword
        let score = combine_rrf_scores(None, Some(7), k);

        let expected = calculate_rrf_score(7, k);
        assert!((score - expected).abs() < 0.0001);

        // Test with neither present
        let score = combine_rrf_scores(None, None, k);
        assert_eq!(score, 0.0);
    }
}
