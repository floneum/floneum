use futures_util::future::BoxFuture;
use std::{hash::BuildHasher, io::BufWriter, num::NonZeroUsize, path::Path, sync::Mutex};

use postcard::{from_bytes, to_io};
use serde::{de::DeserializeOwned, Serialize};

use crate::{Embedder, Embedding, EmbeddingInput};

/// Embedding models can be expensive to run. This struct wraps an embedding model with a cache that stores embeddings that have been computed before.
///
/// # Example
/// ```rust, no_run
/// use kalosm::language::*;
/// use std::num::NonZeroUsize;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let bert = Bert::builder()
///         .build()
///         .await?
///         // You can call the `.cached` method to cache the results of the Bert embedding in a LRU cache with the given capacity.
///         .cached(NonZeroUsize::new(1000).unwrap());
///
///     // Try to load the cache from the filesystem
///     let _ = bert.load_cache("cache.bin");
///
///     let start_time = std::time::Instant::now();
///     let sentences = [
///         "Cats are cool",
///         "The geopolitical situation is dire",
///         "Pets are great",
///         "Napoleon was a tyrant",
///         "Napoleon was a great general",
///     ];
///     // When you embed a new sentence, the cache will store the embedding for that sentence.
///     let embeddings = bert.embed_batch(sentences).await?;
///     println!("{:?}", embeddings);
///     println!("embedding uncached took {:?}", start_time.elapsed());
///
///     let start_time = std::time::Instant::now();
///     // If you embed the same sentences again, the cache will be used.
///     let embeddings = bert.embed_batch(sentences).await?;
///     println!("{:?}", embeddings);
///     println!("embedding cached took {:?}", start_time.elapsed());
///
///     let sentences = [
///         "Cats are cool",
///         "The geopolitical situation is dire",
///         "Pets are great",
///         "Napoleon was a king",
///         "Napoleon was kind of not a great guy",
///     ];
///     // When you embed a new sentence, the cache will store the embedding for that sentence.
///     let embeddings = bert.embed_batch(sentences).await?;
///     println!("{:?}", embeddings);
///     println!("embedding partially cached took {:?}", start_time.elapsed());
///
///     // Save the cache to the filesystem for future use
///     bert.save_cache("cache.bin").unwrap();
///
///     Ok(())
/// }
/// ```
pub struct CachedEmbeddingModel<M: Embedder, S = lru::DefaultHasher> {
    model: M,
    cache: Mutex<lru::LruCache<EmbeddingInput, Embedding<M::VectorSpace>, S>>,
}

impl<M: Embedder> CachedEmbeddingModel<M> {
    /// Create a new cached embedding model.
    pub fn new(model: M, cache_size: NonZeroUsize) -> Self {
        Self {
            model,
            cache: Mutex::new(lru::LruCache::new(cache_size)),
        }
    }
}

impl<M: Embedder, S> CachedEmbeddingModel<M, S> {
    /// Get a reference to the underlying embedder.
    pub fn get_embedder(&self) -> &M {
        &self.model
    }

    /// Get a mutable reference to the underlying embedder.
    pub fn get_embedder_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

impl<M: Embedder, S: BuildHasher> CachedEmbeddingModel<M, S> {
    /// Create a new cached embedding model with a custom hasher.
    pub fn new_with_hasher(model: M, cache_size: NonZeroUsize, hasher: S) -> Self {
        Self {
            model,
            cache: Mutex::new(lru::LruCache::with_hasher(cache_size, hasher)),
        }
    }

    /// Save the cache to a file for future use. You can load the cache from the file with [`Self::load_cache`].
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # use std::num::NonZeroUsize;
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// let bert = Bert::builder()
    ///     .build()
    ///     .await?
    ///     // You can call the `.cached` method to cache the results of the Bert embedding in a LRU cache with the given capacity.
    ///     .cached(NonZeroUsize::new(1000).unwrap());
    /// let sentences = [
    ///     "Cats are cool",
    ///     "The geopolitical situation is dire",
    ///     "Pets are great",
    ///     "Napoleon was a tyrant",
    ///     "Napoleon was a great general",
    /// ];
    /// // When you embed a new sentence, the cache will store the embedding for that sentence.
    /// let embeddings = bert.embed_batch(sentences).await?;
    /// println!("{:?}", embeddings);
    /// // Save the cache to the filesystem for future use
    /// bert.save_cache("cache.bin").unwrap();
    /// # Ok(())
    /// # }
    pub fn save_cache(&self, path: impl AsRef<Path>) -> Result<(), CacheIOError>
    where
        M::VectorSpace: Serialize,
    {
        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        let cache = self.cache.lock().unwrap();
        let items = cache
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>();
        to_io(&items, &mut writer)?;

        Ok(())
    }

    /// Load the cache from a file.
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # use std::num::NonZeroUsize;
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// let bert = Bert::builder()
    ///     .build()
    ///     .await?
    ///     // You can call the `.cached` method to cache the results of the Bert embedding in a LRU cache with the given capacity.
    ///     .cached(NonZeroUsize::new(1000).unwrap());
    ///
    /// // Try to load the cache from the filesystem
    /// let _ = bert.load_cache("cache.bin");
    ///
    /// let sentences = [
    ///     "Cats are cool",
    ///     "The geopolitical situation is dire",
    ///     "Pets are great",
    ///     "Napoleon was a tyrant",
    ///     "Napoleon was a great general",
    /// ];
    /// // If the sentence is already in the cache, it will be returned from the cache instead of running the model
    /// let embeddings = bert.embed_batch(sentences).await?;
    /// println!("{:?}", embeddings);
    /// # Ok(())
    /// # }
    pub fn load_cache(&self, path: impl AsRef<Path>) -> Result<(), CacheIOError>
    where
        M::VectorSpace: DeserializeOwned,
    {
        let contents = std::fs::read(path)?;
        let items: Vec<_> = from_bytes(&contents)?;
        let mut cache = self.cache.lock().unwrap();
        for (k, v) in items {
            cache.put(k, v);
        }

        Ok(())
    }
}

impl<M: Embedder> Embedder for CachedEmbeddingModel<M> {
    /// The vector space that this embedder uses.
    type VectorSpace = M::VectorSpace;

    /// Embed a single string.
    fn embed_for(
        &self,
        input: EmbeddingInput,
    ) -> BoxFuture<'_, anyhow::Result<Embedding<Self::VectorSpace>>> {
        Box::pin(async move {
            {
                // first check if the embedding is in the cache
                let mut write = self.cache.lock().unwrap();
                if let Some(embedding) = write.get(&input) {
                    return Ok(embedding.clone());
                }
            }
            // if not, embed the string and add it to the cache
            let embedding = self.model.embed_for(input.clone()).await?;
            let mut cache = self.cache.lock().unwrap();
            cache.put(input, embedding.clone());
            Ok(embedding)
        })
    }

    /// Embed a batch of strings.
    fn embed_vec_for(
        &self,
        inputs: Vec<EmbeddingInput>,
    ) -> BoxFuture<'_, anyhow::Result<Vec<Embedding<Self::VectorSpace>>>> {
        Box::pin(async move {
            let mut embeddings = vec![Embedding::from([]); inputs.len()];
            // Find any text with embeddings that are already in the cache and fill in first
            let mut text_not_in_cache = Vec::with_capacity(inputs.len());
            let mut indices_not_in_cache = Vec::with_capacity(inputs.len());
            {
                let mut cache = self.cache.lock().unwrap();
                for (i, input) in inputs.into_iter().enumerate() {
                    if let Some(embedding) = cache.get(&input) {
                        embeddings[i] = embedding.clone();
                    } else {
                        text_not_in_cache.push(input);
                        indices_not_in_cache.push(i);
                    }
                }
            }

            // If everything is in the cache, we can just return the embeddings
            if text_not_in_cache.is_empty() {
                return Ok(embeddings);
            }

            // Otherwise embed any text that was not in the cache
            let embeddings_not_in_cache =
                self.model.embed_vec_for(text_not_in_cache.clone()).await?;
            // And add the embeddings to the cache
            for ((i, input), text) in indices_not_in_cache
                .into_iter()
                .zip(embeddings_not_in_cache)
                .zip(text_not_in_cache)
            {
                let mut cache = self.cache.lock().unwrap();
                cache.put(text, input.clone());
                embeddings[i] = input;
            }
            Ok(embeddings)
        })
    }
}

/// An extension trait for [`Embedder`] that allows for caching embeddings.
pub trait EmbedderCacheExt: Embedder {
    /// Wrap the embedder with a cache for previously computed embeddings.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # use std::num::NonZeroUsize;
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// let bert = Bert::builder()
    ///     .build()
    ///     .await?
    ///     // You can call the `.cached` method on any embedder to cache the results of the embedding in a LRU cache with the given capacity.
    ///     .cached(NonZeroUsize::new(1000).unwrap());
    /// # Ok(())
    /// # }
    fn cached(self, cache_size: NonZeroUsize) -> CachedEmbeddingModel<Self>
    where
        Self: Sized,
    {
        CachedEmbeddingModel::new(self, cache_size)
    }
}

impl<M: Embedder> EmbedderCacheExt for M {}

/// An error that can occur when loading or saving a cache.
#[derive(thiserror::Error, Debug)]
pub enum CacheIOError {
    /// Failed to serialize or deserialize cache.
    #[error("Failed to serialize or deserialize cache: {0}")]
    Postcard(#[from] postcard::Error),
    /// Failed to read or write cache file.
    #[error("Failed to read or write cache file: {0}")]
    Io(#[from] std::io::Error),
}
