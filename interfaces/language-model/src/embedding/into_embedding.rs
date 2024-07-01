use futures_util::Future;

use crate::{Embedder, EmbedderExt, Embedding, VectorSpace};

/// Convert a type into an embedding with an embedding model.
pub trait IntoEmbedding<S: VectorSpace> {
    /// Convert the type into an embedding with the given embedding model.
    fn into_embedding<E: Embedder<VectorSpace = S>>(
        self,
        embedder: &E,
    ) -> impl Future<Output = anyhow::Result<Embedding<S>>>;

    /// Convert the type into a query embedding with the given embedding model.
    fn into_query_embedding<E: Embedder<VectorSpace = S>>(
        self,
        embedder: &E,
    ) -> impl Future<Output = anyhow::Result<Embedding<S>>>;
}

/// Convert any type that implements [`ToString`] into an embedding with an embedding model.
impl<S: ToString, V: VectorSpace> IntoEmbedding<V> for S {
    async fn into_embedding<E: Embedder<VectorSpace = V>>(
        self,
        embedder: &E,
    ) -> anyhow::Result<Embedding<V>> {
        embedder.embed(self).await
    }

    async fn into_query_embedding<E: Embedder<VectorSpace = V>>(
        self,
        embedder: &E,
    ) -> anyhow::Result<Embedding<V>> {
        embedder.embed_query(self).await
    }
}

/// Convert an embedding of the same vector space into an embedding with an embedding model.
impl<S: VectorSpace> IntoEmbedding<S> for Embedding<S> {
    async fn into_embedding<E: Embedder<VectorSpace = S>>(
        self,
        _: &E,
    ) -> anyhow::Result<Embedding<S>> {
        Ok(self)
    }

    async fn into_query_embedding<E: Embedder<VectorSpace = S>>(
        self,
        _: &E,
    ) -> anyhow::Result<Embedding<S>> {
        Ok(self)
    }
}
