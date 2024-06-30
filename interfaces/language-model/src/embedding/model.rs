use kalosm_common::BoxedFuture;

use crate::embedding::{Embedding, VectorSpace};
use crate::UnknownVectorSpace;

/// A model that can be used to embed text. This trait is generic over the vector space that the model uses to help keep track of what embeddings came from which model.
///
/// # Example
///
/// ```rust, no_run
/// use kalosm_language_model::Embedder;
/// use rbert::*;
///
/// #[tokio::main]
/// async fn main() {
///     // Bert implements Embedder
///     let mut bert = Bert::builder().build().unwrap();
///     let sentences = [
///         "Cats are cool",
///         "The geopolitical situation is dire",
///         "Pets are great",
///         "Napoleon was a tyrant",
///         "Napoleon was a great general",
///     ];
///     // Embed a batch of documents into the bert vector space
///     let embeddings = bert.embed_batch(sentences).await.unwrap();
///     println!("embeddings {:?}", embeddings);
/// }
/// ```
pub trait Embedder: Send + Sync + 'static {
    /// The vector space that this embedder uses.
    type VectorSpace: VectorSpace + Send + Sync + 'static;

    /// Embed some text into a vector space.
    fn embed_string(
        &self,
        input: String,
    ) -> BoxedFuture<'_, anyhow::Result<Embedding<Self::VectorSpace>>>;

    /// Embed a [`Vec<String>`] into a vector space. Returns a list of embeddings in the same order as the inputs.
    fn embed_vec(
        &self,
        inputs: Vec<String>,
    ) -> BoxedFuture<'_, anyhow::Result<Vec<Embedding<Self::VectorSpace>>>> {
        let inputs = inputs.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        Box::pin(async move {
            let mut embeddings = Vec::with_capacity(inputs.len());
            for input in inputs {
                embeddings.push(self.embed_string(input).await?);
            }
            Ok(embeddings)
        })
    }
}

/// An extension trait for [`Embedder`] with helper methods for iterators, and types that can be converted into a string.
///
/// This trait is automatically implemented for any item that implements [`Embedder`].
pub trait EmbedderExt: Embedder {
    /// Convert this embedder into an embedder trait object.
    fn into_any_embedder(self) -> DynEmbedder
    where
        Self: Sized,
    {
        Box::new(AnyEmbedder::<Self>(self))
    }

    /// Embed some text into a vector space
    fn embed(
        &self,
        input: impl ToString,
    ) -> BoxedFuture<'_, anyhow::Result<Embedding<Self::VectorSpace>>> {
        self.embed_string(input.to_string())
    }

    /// Embed a batch of text into a vector space. Returns a list of embeddings in the same order as the inputs.
    fn embed_batch(
        &self,
        inputs: impl IntoIterator<Item = impl ToString>,
    ) -> BoxedFuture<'_, anyhow::Result<Vec<Embedding<Self::VectorSpace>>>> {
        let inputs = inputs
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        self.embed_vec(inputs)
    }
}

impl<E: Embedder> EmbedderExt for E {}

/// A trait object for an embedder.
pub type DynEmbedder = Box<dyn Embedder<VectorSpace = UnknownVectorSpace>>;

struct AnyEmbedder<E: Embedder + Send + Sync + 'static>(E);

impl<E: Embedder + Send + Sync + 'static> Embedder for AnyEmbedder<E> {
    type VectorSpace = UnknownVectorSpace;

    fn embed_string(
        &self,
        input: String,
    ) -> BoxedFuture<'_, anyhow::Result<Embedding<UnknownVectorSpace>>> {
        let future = self.0.embed_string(input);
        Box::pin(async move { future.await.map(|e| e.cast()) })
    }

    fn embed_vec(
        &self,
        inputs: Vec<String>,
    ) -> BoxedFuture<'_, anyhow::Result<Vec<Embedding<UnknownVectorSpace>>>> {
        let future = self.0.embed_vec(inputs);
        Box::pin(async move {
            future
                .await
                .map(|e| e.into_iter().map(|e| e.cast()).collect())
        })
    }
}
