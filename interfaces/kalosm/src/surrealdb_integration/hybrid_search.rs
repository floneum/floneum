use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use surrealdb::{Connection, RecordIdKey, Surreal};

use kalosm_language::prelude::*;

use crate::language::{
    DocumentTable, DocumentTableAddContextError, DocumentTableBuilder, DocumentTableCreationError,
    DocumentTableModifyError, DocumentTableSearchBuilder, DocumentTableSearchError,
};
use crate::{EmbeddedIndexedTableError, EmbeddingIndexedTable};

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

/// Extension trait to add `with_hybrid_search()` to DocumentTableBuilder
pub trait DocumentTableBuilderHybridExt<C: Connection, E, K: Chunker>: Sized {
    /// Enable hybrid search capability on this table
    ///
    /// # Arguments
    /// * `field_name` - The field to index for full-text search (e.g., "body", "content", "text")
    ///
    /// # Example
    /// ```rust,no_run
    /// use kalosm::language::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let db = /* your database connection */;
    /// let document_table = db
    ///     .document_table_builder("documents")
    ///     .with_hybrid_search("body")  // 👈 Enable hybrid search
    ///     .with_chunker(SemanticChunker::new())
    ///     .at("./db/embeddings.db")
    ///     .build::<Document>()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    fn with_hybrid_search(
        self,
        field_name: impl Into<String>,
    ) -> HybridDocumentTableBuilder<C, E, K>;
}

impl<C: Connection, E, K: Chunker> DocumentTableBuilderHybridExt<C, E, K>
    for DocumentTableBuilder<C, E, K>
{
    fn with_hybrid_search(
        self,
        field_name: impl Into<String>,
    ) -> HybridDocumentTableBuilder<C, E, K> {
        HybridDocumentTableBuilder::new(self, field_name)
    }
}

/// Builder for creating document tables with hybrid search enabled
///
/// This builder is created by calling `.with_hybrid_search()` on a `DocumentTableBuilder`.
/// It will automatically create the necessary full-text search index when `.build()` is called.
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

    /// Change which field to use for full-text search (default is set in `with_hybrid_search()`)
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
    ///
    /// This will:
    /// 1. Create the underlying document table
    /// 2. Create a full-text search index on the specified field
    /// 3. Return a `HybridDocumentTable` that supports both semantic and keyword search
    pub async fn build<R>(self) -> Result<HybridDocumentTable<C, R, E, K>, HybridSearchError>
    where
        E: Embedder,
        R: Serialize + DeserializeOwned + AsRef<Document>,
    {
        let table = self.inner.build().await?;
        let links_table = table.table().table_links();

        // Define the extraction function
        let func_query = format!(
            r#"
                DEFINE FUNCTION fn::extract_chunk($doc_id: record, $start: int, $end: int) {{
                    LET $doc = (SELECT object FROM {} WHERE meta::id(id) = $doc_id)[0];
                    RETURN string::slice($doc.object.{}, $start, $end);
                }};
            "#,
            table.table().table(),
            self.field_name
        );
        table.table().db().query(func_query).await?;

        // Add chunk_text field to existing links by updating them
        // This migrates any existing data
        let update_links_query = format!(
            r#"
                UPDATE `{}`
                SET chunk_text = fn::extract_chunk(document_id, byte_range.start, byte_range.end)
                WHERE chunk_text IS NONE;
            "#,
            links_table
        );
        table.table().db().query(update_links_query).await?;

        // Create the search index on the materialized field
        let index_query = format!(
            r#"
                DEFINE ANALYZER simple TOKENIZERS class,blank FILTERS lowercase, ascii;
                DEFINE INDEX chunk_text_idx ON TABLE `{}`
                FIELDS chunk_text
                SEARCH ANALYZER simple BM25 HIGHLIGHTS;
            "#,
            links_table
        );
        table.table().db().query(index_query).await?;

        Ok(HybridDocumentTable {
            inner: table,
            field_name: self.field_name,
        })
    }
}

/// A document table with hybrid search capabilities enabled
///
/// This type is returned when building a table with `.with_hybrid_search()`.
/// It provides both semantic search (via the inner `DocumentTable`) and
/// hybrid search that combines semantic and keyword matching.
pub struct HybridDocumentTable<C: Connection, R, E: Embedder, K: Chunker> {
    inner: DocumentTable<C, R, E, K>,
    field_name: String,
}

impl<C: Connection, R, E: Embedder, K: Chunker> HybridDocumentTable<C, R, E, K> {
    /// Get a reference to the underlying document table
    pub fn inner(&self) -> &DocumentTable<C, R, E, K> {
        &self.inner
    }

    /// Get a mutable reference to the underlying document table
    pub fn inner_mut(&mut self) -> &mut DocumentTable<C, R, E, K> {
        &mut self.inner
    }

    /// Consume this hybrid table and return the underlying document table
    pub fn into_inner(self) -> DocumentTable<C, R, E, K> {
        self.inner
    }

    /// Get the field name used for keyword search
    pub fn field_name(&self) -> &str {
        &self.field_name
    }
}

// For convenience, also support regular search operations
impl<C: Connection, R, E: Embedder, K: Chunker> HybridDocumentTable<C, R, E, K> {
    /// Perform hybrid search combining semantic and keyword search
    ///
    /// The field name for keyword search was already configured when building the table.
    pub fn hybrid_search(&self, query: impl Into<String>) -> HybridSearchBuilder<'_, C, R, E, K> {
        HybridSearchBuilder::new(&self.inner, query, &self.field_name)
    }

    /// Perform regular semantic search (without keyword matching)
    pub fn search<Em>(&self, embedding: Em) -> DocumentTableSearchBuilder<'_, C, R, E, K, Em>
    where
        Em: IntoEmbedding,
        R: DeserializeOwned,
    {
        self.inner.search(embedding)
    }

    /// Get a reference to the underlying embedding indexed table
    pub fn table(&self) -> &EmbeddingIndexedTable<C, R> {
        self.inner.table()
    }

    /// Get a reference to the embedding model
    pub fn embedding_model(&self) -> &E {
        self.inner.embedding_model()
    }

    /// Get a reference to the chunker
    pub fn chunker(&self) -> &K {
        self.inner.chunker()
    }

    /// Insert a new record into the table
    pub async fn insert(
        &self,
        value: R,
    ) -> Result<RecordIdKey, DocumentTableModifyError<K::Error<E::Error>>>
    where
        R: AsRef<Document> + Serialize + DeserializeOwned + 'static,
    {
        // Chunk the document
        let chunks = self
            .inner
            .chunker()
            .chunk(value.as_ref(), self.inner.embedding_model())
            .await
            .map_err(DocumentTableModifyError::EmbedItem)?;

        // Insert with materialized chunk text
        let record_key = self
            .inner
            .table()
            .insert_with_chunk_text(chunks, value, |v, range| {
                v.as_ref().body()[range.clone()].to_string()
            })
            .await?;

        Ok(record_key)
    }

    /// Extend the table with an iterator of new records
    pub async fn extend<T: IntoIterator<Item = R> + Send>(
        &self,
        iter: T,
    ) -> Result<Vec<RecordIdKey>, DocumentTableModifyError<K::Error<E::Error>>>
    where
        R: AsRef<Document> + Serialize + DeserializeOwned + 'static,
        K: Sync,
    {
        let entries = iter.into_iter().collect::<Vec<_>>();
        let documents = entries.iter().map(|v| v.as_ref()).collect::<Vec<_>>();

        // Batch chunk all documents
        let embeddings = self
            .inner
            .chunker()
            .chunk_batch(documents, self.inner.embedding_model())
            .await
            .map_err(DocumentTableModifyError::EmbedItem)?;

        let mut ids = Vec::new();
        for (value, chunks) in entries.into_iter().zip(embeddings) {
            // Insert each with materialized chunk text
            let id = self
                .inner
                .table()
                .insert_with_chunk_text(chunks, value, |v, range| {
                    v.as_ref().body()[range.clone()].to_string()
                })
                .await?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Extend the table with context documents
    pub async fn add_context<D>(
        &self,
        context: D,
    ) -> Result<Vec<RecordIdKey>, DocumentTableAddContextError<D::Error, K::Error<E::Error>>>
    where
        D: IntoDocuments,
        R: From<Document> + AsRef<Document> + Serialize + DeserializeOwned + 'static,
        K: Sync,
    {
        let documents = context
            .into_documents()
            .await
            .map_err(DocumentTableAddContextError::ConvertItem)?;
        let iter = documents.into_iter().map(|v| v.into());
        self.extend(iter)
            .await
            .map_err(DocumentTableAddContextError::ModifyTable)
    }

    /// Select a record from the table
    pub async fn select(&self, id: impl Into<RecordIdKey>) -> Result<R, EmbeddedIndexedTableError>
    where
        R: Serialize + DeserializeOwned + 'static,
    {
        self.inner.select(id).await
    }

    /// Delete a record from the table
    pub async fn delete(
        &self,
        id: impl Into<RecordIdKey>,
    ) -> Result<Option<R>, EmbeddedIndexedTableError>
    where
        R: Serialize + DeserializeOwned + 'static,
    {
        self.inner.delete(id).await
    }

    /// Select all records from the table
    pub async fn select_all(&self) -> Result<Vec<R>, EmbeddedIndexedTableError>
    where
        R: Serialize + DeserializeOwned + 'static,
    {
        self.inner.select_all().await
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
    user_query: String,
    results: usize,
    semantic_weight: f32,
    keyword_weight: f32,
    field_name: String,
}

#[derive(Deserialize)]
struct KeywordChunkResult {
    document_id: RecordIdKey,
    byte_range: std::ops::Range<usize>,
    chunk_text: String,
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
            user_query: query.into(),
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
        R: Serialize + DeserializeOwned + Clone + AsRef<Document> + Send + Sync,
        <M as Embedder>::Error: std::fmt::Debug + std::fmt::Display + 'static,
    {
        let search_limit = self.results * 3;

        // Perform semantic search
        let semantic_results = self
            .table
            .search(self.user_query.clone())
            .with_results(search_limit)
            .run()
            .await?;

        // Perform keyword search on the links table
        let links_table = self.table.table().table_links();
        let keyword_query = r#"
            SELECT
                document_id,
                byte_range,
                chunk_text,
                search::score(1) as keyword_score
            FROM type::table($table)
            WHERE
                chunk_text @1@ $query
            ORDER BY
                keyword_score DESC
            LIMIT $limit;
        "#;

        let keyword_results: Vec<KeywordChunkResult> = self
            .table
            .table()
            .db()
            .query(keyword_query)
            .bind(("query", self.user_query.clone()))
            .bind(("table", links_table))
            .bind(("limit", search_limit))
            .await?
            .take(0)
            .map_err(|e| HybridSearchError::KeywordSearchError(e.to_string()))?;

        // Process semantic results - keep order with keys
        let semantic_inverse_scores: Vec<f32> = semantic_results
            .iter()
            .map(|r| {
                if r.distance > 0.0 {
                    1.0 / r.distance
                } else {
                    f32::MAX
                }
            })
            .collect();

        let normalized_semantic_vec = normalize_scores(&semantic_inverse_scores)
            .unwrap_or_else(|| vec![0.0; semantic_inverse_scores.len()]);

        let semantic_map: HashMap<String, (RecordIdKey, R, f32)> = semantic_results
            .into_iter()
            .zip(normalized_semantic_vec.iter())
            .map(|(result, &norm_score)| {
                (
                    result.record_id.to_string(),
                    (result.record_id, result.record, norm_score),
                )
            })
            .collect();

        // Aggregate keyword scores by document (keep MAX score per document)
        let mut keyword_aggregated: HashMap<String, f32> = HashMap::new();
        for result in keyword_results {
            let key = result.document_id.to_string();
            keyword_aggregated
                .entry(key)
                .and_modify(|score| *score = score.max(result.keyword_score))
                .or_insert(result.keyword_score);
        }

        // Normalize keyword scores - maintain order
        let keyword_keys: Vec<String> = keyword_aggregated.keys().cloned().collect();
        let keyword_scores: Vec<f32> = keyword_keys.iter().map(|k| keyword_aggregated[k]).collect();

        let normalized_keyword_vec =
            normalize_scores(&keyword_scores).unwrap_or_else(|| vec![0.0; keyword_scores.len()]);

        let keyword_map: HashMap<String, f32> = keyword_keys
            .into_iter()
            .zip(normalized_keyword_vec.iter())
            .map(|(k, &score)| (k, score))
            .collect();

        // Combine results
        let mut combined_results = Vec::new();
        let all_keys: HashSet<String> = semantic_map
            .keys()
            .chain(keyword_map.keys())
            .cloned()
            .collect();

        for key in all_keys {
            let semantic_score = semantic_map.get(&key).map(|(_, _, s)| *s).unwrap_or(0.0);
            let keyword_score = keyword_map.get(&key).copied().unwrap_or(0.0);

            let combined_score = calculate_weighted_score(
                semantic_score,
                keyword_score,
                self.semantic_weight,
                self.keyword_weight,
            );

            // Get the record
            let (record_id, record) = if let Some((id, rec, _)) = semantic_map.get(&key) {
                (id.clone(), rec.clone())
            } else {
                let id = RecordIdKey::from(key.as_str());
                match self.table.table().select(id.clone()).await {
                    Ok(rec) => (id, rec),
                    Err(_) => continue,
                }
            };

            combined_results.push(HybridSearchResult {
                record,
                id: record_id,
                score: combined_score,
                semantic_score,
                keyword_score,
            });
        }

        // Sort and truncate
        combined_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        combined_results.truncate(self.results);

        Ok(combined_results)
    }

    /// Execute the hybrid search using Reciprocal Rank Fusion (RRF)
    ///
    /// # Arguments
    /// * `k` - RRF constant, typically 60
    pub async fn run_rrf(self, k: f32) -> Result<Vec<HybridSearchResult<R>>, HybridSearchError>
    where
        R: Serialize + DeserializeOwned + Clone + AsRef<Document> + Send + Sync,
        <M as Embedder>::Error: std::fmt::Debug + std::fmt::Display + 'static,
    {
        let search_limit = self.results * 3;

        // Perform semantic search
        let semantic_results = self
            .table
            .search(self.user_query.clone())
            .with_results(search_limit)
            .run()
            .await?;

        // Perform keyword search on the links table
        let links_table = self.table.table().table_links();
        let keyword_query = r#"
            SELECT
                document_id,
                byte_range,
                chunk_text,
                search::score(1) as keyword_score
            FROM type::table($table)
            WHERE
                chunk_text @1@ $query
            ORDER BY
                keyword_score DESC
            LIMIT $limit;
        "#;

        let keyword_results: Vec<KeywordChunkResult> = self
            .table
            .table()
            .db()
            .query(keyword_query)
            .bind(("query", self.user_query.clone()))
            .bind(("table", links_table))
            .bind(("limit", search_limit))
            .await?
            .take(0)
            .map_err(|e| HybridSearchError::KeywordSearchError(e.to_string()))?;

        // Build semantic rank map
        let mut semantic_ranks: HashMap<String, (usize, RecordIdKey, R)> = HashMap::new();
        for (rank, result) in semantic_results.iter().enumerate() {
            let key = result.record_id.to_string();
            semantic_ranks.insert(key, (rank, result.record_id.clone(), result.record.clone()));
        }

        // Build keyword rank map (aggregate by document, keep best rank)
        let mut keyword_ranks: HashMap<String, (usize, RecordIdKey)> = HashMap::new();
        for (rank, result) in keyword_results.iter().enumerate() {
            let key = result.document_id.to_string();
            keyword_ranks
                .entry(key)
                .and_modify(|(best_rank, _)| {
                    if rank < *best_rank {
                        *best_rank = rank;
                    }
                })
                .or_insert((rank, result.document_id.clone()));
        }

        // Calculate RRF scores
        let mut rrf_results = Vec::new();
        let all_keys: HashSet<String> = semantic_ranks
            .keys()
            .chain(keyword_ranks.keys())
            .cloned()
            .collect();

        for key in all_keys {
            let semantic_entry = semantic_ranks.get(&key);
            let keyword_entry = keyword_ranks.get(&key);

            let semantic_rrf = semantic_entry
                .map(|(rank, _, _)| calculate_rrf_score(*rank, k))
                .unwrap_or(0.0);

            let keyword_rrf = keyword_entry
                .map(|(rank, _)| calculate_rrf_score(*rank, k))
                .unwrap_or(0.0);

            let rrf_score = semantic_rrf + keyword_rrf;

            // Get the record
            if let Some((_, record_id, record)) = semantic_entry {
                rrf_results.push(HybridSearchResult {
                    record: record.clone(),
                    id: record_id.clone(),
                    score: rrf_score,
                    semantic_score: semantic_rrf,
                    keyword_score: keyword_rrf,
                });
            } else if let Some((_, record_id)) = keyword_entry {
                if let Ok(record) = self.table.table().select(record_id.clone()).await {
                    rrf_results.push(HybridSearchResult {
                        record,
                        id: record_id.clone(),
                        score: rrf_score,
                        semantic_score: semantic_rrf,
                        keyword_score: keyword_rrf,
                    });
                }
            }
        }

        // Sort and truncate
        rrf_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rrf_results.truncate(self.results);

        Ok(rrf_results)
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

    #[test]
    fn test_full_weighted_pipeline() {
        // Simulate a full weighted hybrid search calculation
        let raw_semantic_distances = vec![0.1, 0.3, 0.5];
        let raw_keyword_scores = vec![10.0, 7.0, 3.0];

        // Normalize
        let max_semantic = 0.5;
        let semantic_scores: Vec<f32> = raw_semantic_distances
            .iter()
            .filter_map(|&d| distance_to_similarity(d, max_semantic))
            .collect();

        let keyword_scores = normalize_scores(&raw_keyword_scores).unwrap();

        // Combine
        let combined: Vec<f32> = semantic_scores
            .iter()
            .zip(keyword_scores.iter())
            .map(|(&sem, &key)| calculate_weighted_score(sem, key, 0.7, 0.3))
            .collect();

        // Verify all scores are valid
        assert!(combined.iter().all(|&s| s >= 0.0 && s <= 1.0));

        // First result should have highest score (smallest distance, highest keyword)
        assert!(combined[0] > combined[1]);
        assert!(combined[1] > combined[2]);
    }

    #[test]
    fn test_full_rrf_pipeline() {
        // Simulate RRF calculation
        let k = 60.0;

        // Document rankings: [semantic_rank, keyword_rank]
        let rankings = vec![
            (Some(0), Some(2)), // High in semantic, medium in keyword
            (Some(5), Some(0)), // Medium in semantic, high in keyword
            (Some(1), None),    // High in semantic, not in keyword
        ];

        let rrf_scores: Vec<f32> = rankings
            .iter()
            .map(|&(sem, key)| combine_rrf_scores(sem, key, k))
            .collect();

        // All scores should be positive
        assert!(rrf_scores.iter().all(|&s| s > 0.0));

        // Doc 0 (rank 0 semantic + rank 2 keyword) should score highest
        assert!(rrf_scores[0] > rrf_scores[1]);
        assert!(rrf_scores[0] > rrf_scores[2]);
    }
}
