//! A vector database that can be used to store embeddings and search for similar embeddings.

use heed::{types::*, RwTxn};
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;

use arroy::distances::Angular;
use arroy::{Database as ArroyDatabase, Reader, Writer};
use candle_core::Tensor;
use heed::types::SerdeJson;
use heed::{Database, EnvOpenOptions};
use kalosm_language_model::*;
use kalosm_llama::accelerated_device_if_available;
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
/// # use kalosm_language::prelude::*;
/// # use kalosm_language_model::*;
/// # use rbert::*;
/// # use std::collections::HashMap;
/// # #[tokio::main]
/// # async fn main() {
/// // Create a good default Bert model for search
/// let bert = Bert::new_for_search().await.unwrap();
/// let sentences = [
///     "Kalosm can be used to build local AI applications",
///     "With private LLMs data never leaves your computer",
///     "The quick brown fox jumps over the lazy dog",
/// ];
/// // Embed sentences into the vector space
/// let embeddings = bert.embed_batch(sentences).await.unwrap();
/// println!("embeddings {:?}", embeddings);
///
/// // Create a vector database from the embeddings along with a map between the embedding ids and the sentences
/// let db = VectorDB::new().unwrap();
/// let embeddings = db.add_embeddings(embeddings).unwrap();
/// let embedding_id_to_sentence: HashMap<EmbeddingId, &str> =
///     HashMap::from_iter(embeddings.into_iter().zip(sentences));
///
/// // Find the closest sentence to "What is Kalosm?"
/// let query = "What is Kalosm?";
/// // Embed the query into the vector space. We use `embed_query` instead of `embed` because some models embed queries differently than normal text.
/// let embedding = bert.embed_query(query).await.unwrap();
/// let closest = db.get_closest(embedding, 1).unwrap();
/// if let [closest] = closest.as_slice() {
///     let distance = closest.distance;
///     let text = embedding_id_to_sentence.get(&closest.value).unwrap();
///     println!("distance: {distance}");
///     println!("closest:  {text}");
/// }
/// # }
/// ```
#[doc(alias = "VectorDatabase")]
#[doc(alias = "Vector Database")]
pub struct VectorDB<S = UnknownVectorSpace> {
    database: ArroyDatabase<Angular>,
    metadata: Database<Str, SerdeJson<Vec<u32>>>,
    env: heed::Env,
    dim: AtomicUsize,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: VectorSpace + Sync> Default for VectorDB<S> {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl<S: VectorSpace + Sync> VectorDB<S> {
    fn set_dim(&self, dim: usize) {
        if dim == 0 {
            panic!("Dimension cannot be 0");
        }
        self.dim.store(dim, std::sync::atomic::Ordering::Relaxed);
    }

    fn get_dim(&self) -> anyhow::Result<usize> {
        let mut dims = self.dim.load(std::sync::atomic::Ordering::Relaxed);
        if dims == 0 {
            let rtxn = self.env.read_txn()?;
            let reader = Reader::<Angular>::open(&rtxn, 0, self.database)?;
            dims = reader.dimensions();
            self.set_dim(dims);
        }
        Ok(dims)
    }

    /// Create a new temporary vector database.
    #[tracing::instrument]
    pub fn new() -> heed::Result<Self> {
        let dir = tempfile::tempdir()?;

        Self::new_at(dir.path())
    }

    /// Create a new vector database at the given path.
    pub fn new_at(path: impl AsRef<std::path::Path>) -> heed::Result<Self> {
        const TWENTY_HUNDRED_MIB: usize = 2 * 1024 * 1024 * 1024;

        std::fs::create_dir_all(&path)?;

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(TWENTY_HUNDRED_MIB)
                .open(path)
        }?;

        let mut wtxn = env.write_txn()?;
        let db: ArroyDatabase<Angular> = env.create_database(&mut wtxn, None)?;
        let metadata: Database<Str, SerdeJson<Vec<u32>>> = env.create_database(&mut wtxn, None)?;
        wtxn.commit()?;

        Ok(Self {
            database: db,
            metadata,
            env,
            dim: AtomicUsize::new(0),
            _phantom: std::marker::PhantomData,
        })
    }

    fn take_id(&self, wtxn: &mut RwTxn) -> anyhow::Result<EmbeddingId> {
        if let Some(mut free) = self.metadata.get(wtxn, "free")? {
            if let Some(id) = free.pop() {
                self.metadata.put(wtxn, "free", &free)?;
                return Ok(EmbeddingId(id));
            }
        }
        match self.metadata.get(wtxn, "max")? {
            Some(max) => {
                let id = max[0];
                self.metadata.put(wtxn, "max", &vec![id + 1])?;
                Ok(EmbeddingId(id))
            }
            None => {
                self.metadata.put(wtxn, "max", &vec![1])?;
                Ok(EmbeddingId(0))
            }
        }
    }

    fn recycle_id(&self, id: EmbeddingId, wtxn: &mut RwTxn) -> anyhow::Result<()> {
        let mut free = self.metadata.get(wtxn, "free").unwrap().unwrap_or_default();
        free.push(id.0);
        self.metadata.put(wtxn, "free", &free)?;

        Ok(())
    }

    /// Get the underlying database.
    pub fn raw(&self) -> (&ArroyDatabase<Angular>, &heed::Env) {
        (&self.database, &self.env)
    }

    /// Clear the vector database.
    pub async fn clear(&self) -> anyhow::Result<()> {
        let mut wtxn = self.env.write_txn()?;
        let dims = self.get_dim()?;
        let writer = Writer::<Angular>::new(self.database, 0, dims);
        writer.clear(&mut wtxn)?;

        // Reset the ids
        self.metadata.put(&mut wtxn, "max", &vec![0])?;
        self.metadata.put(&mut wtxn, "free", &vec![])?;
        wtxn.commit()?;

        Ok(())
    }

    /// Remove an embedding from the vector database.
    pub fn remove_embedding(&self, embedding_id: EmbeddingId) -> anyhow::Result<()> {
        let dims = self.get_dim()?;

        let mut wtxn = self.env.write_txn()?;

        let writer = Writer::<Angular>::new(self.database, 0, dims);

        writer.del_item(&mut wtxn, embedding_id.0)?;
        self.recycle_id(embedding_id, &mut wtxn)?;

        let mut rng = StdRng::from_entropy();

        writer.build(&mut wtxn, &mut rng, None)?;

        wtxn.commit()?;

        Ok(())
    }

    /// Add a new embedding to the vector database.
    ///
    /// Note: Adding embeddings in a batch with [`VectorDB::add_embeddings`] will be faster.
    pub fn add_embedding(&self, embedding: Embedding<S>) -> anyhow::Result<EmbeddingId> {
        let embedding = embedding.vector().to_vec1()?;

        self.set_dim(embedding.len());

        let mut wtxn = self.env.write_txn()?;

        let writer = Writer::<Angular>::new(self.database, 0, embedding.len());

        let id = self.take_id(&mut wtxn)?;

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
        self.set_dim(first_embedding.len());

        let mut wtxn = self.env.write_txn()?;
        let writer = Writer::<Angular>::new(self.database, 0, first_embedding.len());

        let mut ids: Vec<_> = Vec::with_capacity(embeddings.size_hint().0 + 1);

        {
            let first_id = self.take_id(&mut wtxn)?;
            writer.add_item(&mut wtxn, first_id.0, &first_embedding)?;
            ids.push(first_id);
        }

        for embedding in embeddings {
            let id = self.take_id(&mut wtxn)?;
            writer.add_item(&mut wtxn, id.0, &embedding?)?;
            ids.push(id);
        }

        let mut rng = StdRng::from_entropy();

        writer.build(&mut wtxn, &mut rng, None)?;

        wtxn.commit()?;

        Ok(ids)
    }

    /// Get the embedding for an embedding id.
    pub fn get_embedding(&self, embedding_id: EmbeddingId) -> anyhow::Result<Embedding<S>> {
        let rtxn = self.env.read_txn()?;
        let reader = Reader::<Angular>::open(&rtxn, 0, self.database)?;

        let embedding = reader
            .item_vector(&rtxn, embedding_id.0)?
            .ok_or_else(|| anyhow::anyhow!("Embedding not found"))?;

        let shape = (embedding.len(),);
        Ok(Embedding::new(Tensor::from_vec(
            embedding,
            shape,
            &accelerated_device_if_available()?,
        )?))
    }

    /// Get the closest N embeddings to the given embedding.
    pub fn get_closest(
        &self,
        embedding: Embedding<S>,
        n: usize,
    ) -> anyhow::Result<Vec<VectorDBSearchResult>> {
        let rtxn = self.env.read_txn()?;
        let reader = Reader::<Angular>::open(&rtxn, 0, self.database)?;

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
#[derive(Debug, Clone)]
pub struct VectorDBSearchResult {
    /// The distance from the searched point.
    pub distance: f32,
    /// The value of the point.
    pub value: EmbeddingId,
}

/// A unique identifier for an embedding. If you delete an embedding, the id will be recycled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EmbeddingId(pub u32);
