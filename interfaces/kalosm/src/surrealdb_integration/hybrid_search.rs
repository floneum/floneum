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
    // The inner error (E) MUST implement these traits for formatting and safe unwinding:
    E: std::fmt::Debug + std::fmt::Display + 'static,
{
    fn from(error: DocumentTableSearchError<E>) -> Self {
        match error {
            // DocumentTableSearchError::SearchTable converts directly to SemanticSearchError
            DocumentTableSearchError::SearchTable(err) => Self::SemanticSearchError(err),

            // DocumentTableSearchError::EmbedQuery converts the error to a String
            DocumentTableSearchError::EmbedQuery(err) => {
                Self::SemanticQueryError(format!("Embedding failed: {}", err))
            }
        }
    }
}

/// Extension trait to help set up hybrid search capability.
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

/// Create a hash-based key from record content for deduplication
fn create_content_hash<T>(record: &T) -> u64
where
    T: Hash,
{
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    record.hash(&mut hasher);
    hasher.finish()
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
        // The problematic line:
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

        // Normalize scores to 0-1 range
        let max_semantic = semantic_results
            .first()
            .map(|r| r.distance)
            .unwrap_or(1.0)
            .max(0.001);

        let max_keyword = keyword_results
            .iter()
            .map(|r| r.keyword_score)
            .fold(0.0f32, f32::max)
            .max(0.001);

        // Create score maps using record content as key
        let mut semantic_map: HashMap<u64, (RecordIdKey, R, f32)> = HashMap::new();
        for result in semantic_results {
            let normalized = 1.0 - (result.distance / max_semantic);
            let key = create_content_hash(&result.record);
            semantic_map.insert(key, (result.record_id, result.record, normalized));
        }

        let mut keyword_map: HashMap<u64, (R, f32)> = HashMap::new();
        for result in keyword_results {
            let normalized = result.keyword_score / max_keyword;
            let key = create_content_hash(&result.record);
            keyword_map.insert(key, (result.record, normalized));
        }

        // Combine scores using weighted sum
        let mut combined_results = Vec::new();
        let mut seen_keys = HashSet::new();

        // Collect all unique keys
        let all_keys: Vec<u64> = semantic_map
            .keys()
            .chain(keyword_map.keys())
            .copied()
            .collect();

        for key in all_keys {
            if seen_keys.insert(key) {
                let semantic_score = semantic_map.get(&key).map(|(_, _, s)| *s).unwrap_or(0.0);
                let keyword_score = keyword_map.get(&key).map(|(_, s)| *s).unwrap_or(0.0);

                // Weighted combination
                let combined_score =
                    semantic_score * self.semantic_weight + keyword_score * self.keyword_weight;

                // Get the record (prefer semantic result if available)
                if let Some((record_id, record, _)) = semantic_map.get(&key) {
                    combined_results.push(HybridSearchResult {
                        record: record.clone(),
                        id: record_id.clone(),
                        score: combined_score,
                        semantic_score,
                        keyword_score,
                    });
                } else if let Some((record, _)) = keyword_map.get(&key) {
                    // For keyword-only results
                    combined_results.push(HybridSearchResult {
                        record: record.clone(),
                        id: RecordIdKey::from("keyword_only"),
                        score: combined_score,
                        semantic_score,
                        keyword_score,
                    });
                }
            }
        }

        // Sort by combined score and limit results
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

        // RRF formula: score = sum(1 / (k + rank))
        let mut rrf_scores: HashMap<u64, (R, f32, f32, f32, RecordIdKey)> = HashMap::new();

        // Add semantic ranks
        for (rank, result) in semantic_results.iter().enumerate() {
            let rrf_score = 1.0 / (k + rank as f32 + 1.0);
            let key = create_content_hash(&result.record);
            let normalized_semantic = 1.0 - result.distance;

            rrf_scores
                .entry(key)
                .and_modify(|entry| {
                    entry.1 += rrf_score;
                    entry.2 = normalized_semantic;
                })
                .or_insert((
                    result.record.clone(),
                    rrf_score,
                    normalized_semantic,
                    0.0,
                    result.record_id.clone(),
                ));
        }

        // Add keyword ranks
        for (rank, result) in keyword_results.iter().enumerate() {
            let rrf_score = 1.0 / (k + rank as f32 + 1.0);
            let key = create_content_hash(&result.record);

            rrf_scores
                .entry(key.clone())
                .and_modify(|entry| {
                    entry.1 += rrf_score;
                    entry.3 = result.keyword_score;
                })
                .or_insert((
                    result.record.clone(),
                    rrf_score,
                    0.0,
                    result.keyword_score,
                    RecordIdKey::from("unknown"),
                ));
        }

        // Convert to results and sort
        let mut results: Vec<_> = rrf_scores
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
