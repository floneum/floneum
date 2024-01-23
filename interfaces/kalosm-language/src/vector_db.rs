//! A vector database that can be used to store embeddings and search for similar embeddings.

use std::cell::Cell;
use std::fmt::Debug;
use std::sync::Mutex;

use arroy::distances::Euclidean;
use arroy::{Database as ArroyDatabase, Reader, Writer};
use heed::EnvOpenOptions;
use kalosm_language_model::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// A vector database that can be used to store embeddings and search for similar embeddings.
///
/// It uses an in memory database with fast lookups for nearest neighbors and points within a certain distance.
///
/// # Example
///
/// ```rust, no_run
/// use kalosm_language::VectorDB;
/// use kalosm_language_model::Embedder;
/// use rbert::*;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let mut bert = Bert::builder().build()?;
///     let sentences = vec![
///         "Cats are cool",
///         "The geopolitical situation is dire",
///         "Pets are great",
///         "Napoleon was a tyrant",
///         "Napoleon was a great general",
///     ];
///     let embeddings = bert.embed_batch(&sentences).await?;
///     println!("embeddings {:?}", embeddings);
///
///     // Create a vector database from the embeddings
///     let mut db = VectorDB::new(embeddings, sentences);
///     // Find the closest sentence to "Cats are good"
///     let embedding = bert.embed("Cats are good").await?;
///     let closest = db.get_closest(embedding, 1);
///     println!("closest: {:?}", closest);
///
///     Ok(())
/// }
/// ```
pub struct VectorDB<S: VectorSpace = UnknownVectorSpace> {
    database: ArroyDatabase<Euclidean>,
    env: heed::Env,
    max_id: Cell<EmbeddingId>,
    recycled_ids: Mutex<Vec<EmbeddingId>>,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: VectorSpace + Sync> VectorDB<S>
where
    Self: Sync + Send,
{
    /// Create a new temporary vector database.
    #[tracing::instrument]
    pub fn new() -> anyhow::Result<Self> {
        let dir = tempfile::tempdir()?;

        Self::new_at(dir.path())
    }

    /// Create a new vector database at the given path.
    pub fn new_at(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        const TWENTY_HUNDRED_MIB: usize = 2 * 1024 * 1024 * 1024;

        let env = EnvOpenOptions::new()
            .map_size(TWENTY_HUNDRED_MIB)
            .open(path)?;

        let mut wtxn = env.write_txn()?;
        let db: ArroyDatabase<Euclidean> = env.create_database(&mut wtxn, None)?;
        wtxn.commit()?;

        Ok(Self {
            database: db,
            env,
            max_id: Cell::new(EmbeddingId(0)),
            recycled_ids: Mutex::new(Vec::new()),
            _phantom: std::marker::PhantomData,
        })
    }

    fn take_id(&self) -> EmbeddingId {
        self.recycled_ids.lock().unwrap().pop().unwrap_or_else(|| {
            let id = self.max_id.get();
            self.max_id.set(EmbeddingId(id.0 + 1));
            id
        })
    }

    #[allow(dead_code)]
    fn recycle_id(&self, id: EmbeddingId) {
        self.recycled_ids.lock().unwrap().push(id);
    }

    /// Get the underlying database.
    pub fn raw(&self) -> (&ArroyDatabase<Euclidean>, &heed::Env) {
        (&self.database, &self.env)
    }

    /// Add a new embedding to the vector database.
    ///
    /// Note: Adding embeddings in a batch with [`add_embeddings`] will be faster.
    pub fn add_embedding(&self, embedding: Embedding<S>) -> anyhow::Result<EmbeddingId> {
       
        let embedding = embedding.vector().to_vec1()?;

        let mut wtxn = self.env.write_txn()?;

        let writer = Writer::<Euclidean>::new(self.database, 0, embedding.len())?;

       

        let id = self.take_id();

        writer.add_item(&mut wtxn, id.0, &embedding)?;

        let mut rng = StdRng::from_entropy();

        writer.build(&mut wtxn, &mut rng, None)?;

        wtxn.commit()?;

        Ok(id)
    }

    /// Add a new batch of embeddings to the vector database.
    pub fn add_embeddings(
        &self,
        embedding: impl IntoIterator<Item = Embedding<S>>,
    ) -> anyhow::Result<Vec<EmbeddingId>> {
        let mut embeddings = embedding.into_iter().map(|e| e.vector().to_vec1());
        let first_embedding = match embeddings.next() {
            Some(e) => e?,
            None => return Ok(Vec::new()),
        };

        let mut wtxn = self.env.write_txn()?;
        let writer = Writer::<Euclidean>::new(self.database, 0, first_embedding.len())?;

        let mut ids: Vec<_> = Vec::with_capacity(embeddings.size_hint().0 + 1);

        {
            let first_id = self.take_id();
            writer.add_item(&mut wtxn, first_id.0, &first_embedding)?;
            ids.push(first_id);
        }

        for embedding in embeddings {
            let id = self.take_id();
            writer.add_item(&mut wtxn, id.0, &embedding?)?;
            ids.push(id);
        }

        let mut rng = StdRng::from_entropy();

        writer.build(&mut wtxn, &mut rng, None)?;

        wtxn.commit()?;

        Ok(ids)
    }

    /// Get the closest N embeddings to the given embedding.
    pub fn get_closest(
        &self,
        embedding: Embedding<S>,
        n: usize,
    ) -> anyhow::Result<Vec<VectorDBSearchResult>> {
        let rtxn = self.env.read_txn()?;
        let reader = Reader::<Euclidean>::open(&rtxn, 0, self.database)?;

        let vector = embedding.vector().to_vec1()?;
        let arroy_results = reader.nns_by_vector(&rtxn, &vector, n, None, None)?;

        Ok(arroy_results
            .into_iter()
            .map(|(id, distance)| {
                let value = EmbeddingId(id);
                VectorDBSearchResult { distance, value }
            })
            .collect::<Vec<_>>())
    }
}

/// A resulting point from a search.
pub struct VectorDBSearchResult {
    /// The distance from the searched point.
    pub distance: f32,
    /// The value of the point.
    pub value: EmbeddingId,
}

/// A unique identifier for an embedding. If you delete an embedding, the id will be recycled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EmbeddingId(u32);
