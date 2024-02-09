use kalosm_language::kalosm_language_model::{UnknownVectorSpace, VectorSpace};
use kalosm_language::prelude::*;
use kalosm_language::vector_db::VectorDB;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use surrealdb::sql::{Id, Thing};
use surrealdb::{Connection, Surreal};

#[derive(Debug, Serialize)]
struct Name<'a> {
    first: &'a str,
    last: &'a str,
}

#[derive(Debug, Serialize)]
struct Person<'a> {
    title: &'a str,
    name: Name<'a>,
    marketing: bool,
}

#[derive(Debug, Serialize)]
struct Responsibility {
    marketing: bool,
}

#[derive(Debug, Deserialize)]
struct Record {
    #[allow(dead_code)]
    id: Thing,
}

/// A table in a surreal database with a primary key tied to an embedding in a vector database.
pub struct IndexedTable<C: Connection, R, S: VectorSpace = UnknownVectorSpace> {
    table: String,
    db: Surreal<C>,
    vector_db: VectorDB<S>,
    phantom: std::marker::PhantomData<R>,
}

impl<C: Connection, R, S: VectorSpace> IndexedTable<C, R, S> {
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
        R: Serialize + DeserializeOwned,
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
        R: Serialize + DeserializeOwned,
    {
        let records = self.db.select::<Vec<R>>(self.table.clone()).await?;
        Ok(records)
    }
}

trait VectorDbSurrealExt<C: Connection> {
    fn create_embedded_indexed_table_with_vector_db<S: VectorSpace, R: Serialize + DeserializeOwned>(
        &self,
        table: &str,
        vector_db: VectorDB<S>,
    ) -> IndexedTable<C, R, S>;

	fn create_embedded_indexed_table<R: Serialize + DeserializeOwned>(
		&self,
		table: &str,
	) -> anyhow::Result<IndexedTable<C, R, UnknownVectorSpace>> {
		Ok(self.create_embedded_indexed_table_with_vector_db(table, VectorDB::new()?))
	}

	fn create_embedded_indexed_table_at<R: Serialize + DeserializeOwned>(
		&self,
		table: &str,
		path: impl AsRef<std::path::Path>,
	) -> anyhow::Result<IndexedTable<C, R, UnknownVectorSpace>> {
		Ok(self.create_embedded_indexed_table_with_vector_db(table, VectorDB::new_at(path)?))
	}
}

impl<C: Connection> VectorDbSurrealExt<C> for Surreal<C> {
    fn create_embedded_indexed_table_with_vector_db<S: VectorSpace, R: Serialize + DeserializeOwned>(
        &self,
        table: &str,
        vector_db: VectorDB<S>,
    ) -> IndexedTable<C, R, S> {
        IndexedTable {
            table: table.to_string(),
            db: self.clone(),
            vector_db,
            phantom: std::marker::PhantomData,
        }
    }
}
