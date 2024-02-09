use kalosm_language::kalosm_language_model::{UnknownVectorSpace, VectorSpace};
use kalosm_language::prelude::*;
use kalosm_language::vector_db::VectorDB;
use serde::de::DeserializeOwned;
use serde::Serialize;
use surrealdb::sql::{Id, Thing};
use surrealdb::{Connection, Surreal};

#[cfg(feature = "language")]
mod document_table;
#[cfg(feature = "language")]
pub use document_table::*;

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

    /// Get the raw vector database.
    pub fn vector_db(&self) -> &VectorDB<S> {
        &self.vector_db
    }

    /// Get the raw surreal database.
    pub fn db(&self) -> &Surreal<C> {
        &self.db
    }

    /// Insert a new record into the table with the given embedding.
    pub async fn insert(&self, embedding: Embedding<S>, value: R) -> anyhow::Result<EmbeddingId>
    where
        R: Serialize + DeserializeOwned,
    {
        let record_id = self.vector_db.add_embedding(embedding)?;
        let thing = Thing {
            tb: self.table.clone(),
            id: Id::Number(record_id.0 as i64),
        };
        let old = self.db.create::<Option<R>>(thing).content(value).await?;
        debug_assert!(old.is_none());

        Ok(record_id)
    }

    /// Update a record in the table with the given embedding id.
    pub async fn update(&self, id: EmbeddingId, value: R) -> anyhow::Result<Option<R>>
    where
        R: Serialize + DeserializeOwned,
    {
        let thing = Thing {
            tb: self.table.clone(),
            id: Id::Number(id.0 as i64),
        };
        let old = self.db.update::<Option<R>>(thing).merge(value).await?;

        Ok(old)
    }

    /// Select a record from the table with the given embedding id.
    pub async fn select(&self, id: EmbeddingId) -> anyhow::Result<R>
    where
        R: DeserializeOwned,
    {
        let thing = Thing {
            tb: self.table.clone(),
            id: Id::Number(id.0 as i64),
        };
        let record = self.db.select::<Option<R>>(thing).await?;
        match record {
            Some(record) => Ok(record),
            None => anyhow::bail!("Record not found"),
        }
    }

    /// Delete a record from the table with the given embedding id.
    pub async fn delete(&self, id: EmbeddingId) -> anyhow::Result<Option<R>>
    where
        R: Serialize + DeserializeOwned,
    {
        // First delete the record from the vector db
        self.vector_db.remove_embedding(id)?;

        let thing = Thing {
            tb: self.table.clone(),
            id: Id::Number(id.0 as i64),
        };
        let old = self.db.delete::<Option<R>>(thing).await?;

        Ok(old)
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
            let record = self.select(id.value).await?;
            records.push(EmbeddingIndexedTableSearchResult {
                distance: id.distance,
                id: id.value,
                record,
            });
        }
        Ok(records)
    }
}

/// The result of a search in an embedding indexed table.
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingIndexedTableSearchResult<R> {
    /// The distance from the searched point.
    pub distance: f32,
    /// The embedding id of the record.
    pub id: EmbeddingId,
    /// The record.
    pub record: R,
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
