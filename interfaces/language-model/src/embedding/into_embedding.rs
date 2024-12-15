use futures_util::Future;

use crate::{Embedder, EmbedderExt, Embedding, VectorSpace};

/// Convert a type into an embedding with an embedding model.
pub trait IntoEmbedding<S: VectorSpace> {
    /// Convert the type into an embedding with the given embedding model.
    fn into_embedding<E: Embedder<VectorSpace = S>>(
        self,
        embedder: &E,
    ) -> impl Future<Output = Result<Embedding<S>, E::Error>> + Send;

    /// Convert the type into a query embedding with the given embedding model.
    fn into_query_embedding<E: Embedder<VectorSpace = S>>(
        self,
        embedder: &E,
    ) -> impl Future<Output = Result<Embedding<S>, E::Error>> + Send;
}

/// Convert any type that implements [`ToString`] into an embedding with an embedding model.
impl<S: ToString + Send, V: VectorSpace> IntoEmbedding<V> for S {
    async fn into_embedding<E: Embedder<VectorSpace = V>>(
        self,
        embedder: &E,
    ) -> Result<Embedding<V>, E::Error> {
        embedder.embed(self).await
    }

    async fn into_query_embedding<E: Embedder<VectorSpace = V>>(
        self,
        embedder: &E,
    ) -> Result<Embedding<V>, E::Error> {
        embedder.embed_query(self).await
    }
}

/// Convert an embedding of the same vector space into an embedding with an embedding model.
impl<S: VectorSpace> IntoEmbedding<S> for Embedding<S> {
    async fn into_embedding<E: Embedder<VectorSpace = S>>(
        self,
        _: &E,
    ) -> Result<Embedding<S>, E::Error> {
        Ok(self)
    }

    async fn into_query_embedding<E: Embedder<VectorSpace = S>>(
        self,
        _: &E,
    ) -> Result<Embedding<S>, E::Error> {
        Ok(self)
    }
}
