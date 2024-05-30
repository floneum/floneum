use kalosm_language::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use surrealdb::sql::{Id, Thing};
use surrealdb::{Connection, Surreal};

#[cfg(feature = "language")]
mod document_table;
#[cfg(feature = "language")]
pub use document_table::*;

#[derive(Serialize, Deserialize)]
struct DocumentLink {
    document_id: Id,
    byte_range: std::ops::Range<usize>,
}

#[derive(Serialize, Deserialize)]
struct ObjectWithEmbeddingIds<T> {
    #[serde(flatten)]
    object: T,
    embedding_ids: Vec<EmbeddingId>,
}

/// A table in a surreal database with a primary key tied to an embedding in a vector database.
pub struct EmbeddingIndexedTable<C: Connection, R, S = UnknownVectorSpace> {
    table: String,
    db: Surreal<C>,
    vector_db: VectorDB<S>,
    phantom: std::marker::PhantomData<R>,
}

impl<C: Connection, R, S: VectorSpace> EmbeddingIndexedTable<C, R, S> {
    /// Get the name of the table.
    pub fn table(&self) -> &str {
        &self.table
    }

    /// Get the name of the table. The table has a id with the same number as an embedding id and the value of the id of the object in the table
    fn table_links(&self) -> String {
        format!("{}-links", &self.table)
    }

    /// Get the raw vector database.
    pub fn vector_db(&self) -> &VectorDB<S> {
        &self.vector_db
    }

    /// Get the raw surreal database.
    pub fn db(&self) -> &Surreal<C> {
        &self.db
    }

    /// Insert a new record into the table with the given embedding.
    pub async fn insert(
        &self,
        chunks: impl IntoIterator<Item = Chunk<S>>,
        value: R,
    ) -> anyhow::Result<Id>
    where
        R: Serialize + DeserializeOwned,
    {
        let id = Id::uuid();

        let mut embedding_ids = Vec::new();
        let thing = Thing {
            tb: self.table.clone(),
            id: id.clone(),
        };

        for chunk in chunks {
            let chunk_embedding_ids = self.vector_db.add_embeddings(chunk.embeddings)?;
            for embedding_id in chunk_embedding_ids {
                let byte_range = chunk.byte_range.clone();
                embedding_ids.push(embedding_id);

                let link = Thing {
                    tb: self.table_links(),
                    id: Id::Number(embedding_id.0 as i64),
                };
                self.db
                    .create::<Option<DocumentLink>>(link)
                    .content(DocumentLink {
                        document_id: id.clone(),
                        byte_range,
                    })
                    .await?;
            }
        }

        self.db
            .create::<Option<ObjectWithEmbeddingIds<R>>>(thing)
            .content(ObjectWithEmbeddingIds {
                object: value,
                embedding_ids,
            })
            .await?;

        Ok(id)
    }

    /// Update a record in the table with the given embedding id.
    pub async fn update(&self, id: Id, value: R) -> anyhow::Result<Option<R>>
    where
        R: Serialize + DeserializeOwned,
    {
        let thing = Thing {
            tb: self.table.clone(),
            id,
        };
        let old = self.db.update::<Option<R>>(thing).merge(value).await?;

        Ok(old)
    }

    /// Select a record from the table with the given embedding id.
    pub async fn select(&self, id: Id) -> anyhow::Result<R>
    where
        R: DeserializeOwned,
    {
        let thing = Thing {
            tb: self.table.clone(),
            id,
        };
        let record = self.db.select::<Option<R>>(thing).await?;
        match record {
            Some(record) => Ok(record),
            None => anyhow::bail!("Record not found"),
        }
    }

    /// Delete a record from the table with the given embedding id.
    pub async fn delete(&self, id: Id) -> anyhow::Result<Option<R>>
    where
        R: Serialize + DeserializeOwned,
    {
        // First delete the record from the main table
        let thing = Thing {
            tb: self.table.clone(),
            id,
        };
        let old = self
            .db
            .delete::<Option<ObjectWithEmbeddingIds<R>>>(thing)
            .await?;

        if let Some(old) = old {
            let ObjectWithEmbeddingIds {
                object,
                embedding_ids,
            } = old;
            // Then delete the links from the links table
            for id in embedding_ids {
                let link = Thing {
                    tb: self.table_links(),
                    id: Id::Number(id.0 as i64),
                };
                self.db.delete::<Option<DocumentLink>>(link).await?;
                // Then delete the embedding from the vector db
                self.vector_db.remove_embedding(id)?;
            }

            Ok(Some(object))
        } else {
            Ok(None)
        }
    }

    /// Select all records from the table.
    pub async fn select_all(&self) -> anyhow::Result<Vec<R>>
    where
        R: DeserializeOwned,
    {
        let records = self.db.select::<Vec<R>>(self.table.clone()).await?;
        Ok(records)
    }

    /// Select the top k records nearest records to the given embedding.
    pub async fn select_nearest(
        &self,
        embedding: Embedding<S>,
        k: usize,
    ) -> anyhow::Result<Vec<EmbeddingIndexedTableSearchResult<R>>>
    where
        R: DeserializeOwned,
    {
        let ids = self.vector_db.get_closest(embedding, k)?;
        let mut records = Vec::new();
        for id in ids {
            let main_table_id = self
                .db
                .select::<Option<DocumentLink>>(Thing {
                    tb: self.table_links(),
                    id: Id::Number(id.value.0 as i64),
                })
                .await?
                .ok_or_else(|| anyhow::anyhow!("Record not found"))?;
            let record = self.select(main_table_id.document_id.clone()).await?;
            records.push(EmbeddingIndexedTableSearchResult {
                distance: id.distance,
                id: id.value,
                record_id: main_table_id.document_id,
                byte_range: main_table_id.byte_range,
                record,
            });
        }
        Ok(records)
    }
}

/// The result of a search in an embedding indexed table.
#[derive(Debug, Clone)]
pub struct EmbeddingIndexedTableSearchResult<R> {
    /// The distance from the searched point.
    pub distance: f32,
    /// The embedding id of the record.
    pub id: EmbeddingId,
    /// The record id.
    pub record_id: Id,
    /// The byte range of the record.
    pub byte_range: std::ops::Range<usize>,
    /// The record.
    pub record: R,
}

impl<R> EmbeddingIndexedTableSearchResult<R>
where
    R: DeserializeOwned,
{
    /// Get the text of the search result.
    pub fn text(&self) -> String
    where
        R: AsRef<Document>,
    {
        self.record.as_ref().body()[self.byte_range.clone()].to_string()
    }
}

/// A builder for creating a new document table.
pub struct EmbeddingIndexedTableBuilder<C: Connection> {
    table: String,
    db: Surreal<C>,
    location: Option<std::path::PathBuf>,
}

impl<C: Connection> EmbeddingIndexedTableBuilder<C> {
    /// Create a new document table builder.
    pub fn new(table: &str, db: Surreal<C>) -> Self {
        Self {
            table: table.to_string(),
            db,
            location: None,
        }
    }

    /// Set the location of the vector database.
    pub fn at(mut self, location: impl AsRef<std::path::Path>) -> Self {
        self.location = Some(location.as_ref().to_path_buf());
        self
    }

    /// Build the document table.
    pub fn build<S: VectorSpace, R: Serialize + DeserializeOwned>(
        self,
    ) -> anyhow::Result<EmbeddingIndexedTable<C, R, S>> {
        let vector_db = if let Some(location) = self.location {
            VectorDB::new_at(location)?
        } else {
            VectorDB::new()?
        };
        Ok(EmbeddingIndexedTable {
            table: self.table.to_string(),
            db: self.db,
            vector_db,
            phantom: std::marker::PhantomData,
        })
    }
}

/// An extension trait for the surreal database to interact with vector indexed tables.
pub trait VectorDbSurrealExt<C: Connection> {
    /// Create a new vector indexed table builder.
    fn vector_indexed_table_builder(&self, table: &str) -> EmbeddingIndexedTableBuilder<C>;

    /// Create a new document table builder.    
    fn document_table_builder(&self, table: &str) -> DocumentTableBuilder<C, Bert, ChunkStrategy>;
}

impl<C: Connection> VectorDbSurrealExt<C> for Surreal<C> {
    fn vector_indexed_table_builder(&self, table: &str) -> EmbeddingIndexedTableBuilder<C> {
        EmbeddingIndexedTableBuilder::new(table, self.clone())
    }

    fn document_table_builder(&self, table: &str) -> DocumentTableBuilder<C, Bert, ChunkStrategy> {
        DocumentTableBuilder::new(table, self.clone())
    }
}
