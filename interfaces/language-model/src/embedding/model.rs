use std::sync::Arc;

use kalosm_common::BoxedFuture;

use crate::embedding::{Embedding, VectorSpace};
use crate::UnknownVectorSpace;

/// A model that can be used to embed text. This trait is generic over the vector space that the model uses to help keep track of what embeddings came from which model.
///
/// # Example
///
/// ```rust, no_run
/// use kalosm_language_model::Embedder;
/// use kalosm::language::*;
///
/// #[tokio::main]
/// async fn main() {
///     // Bert implements Embedder
///     let mut bert = Bert::new().await.unwrap();
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
    ) -> BoxedFuture<'_, anyhow::Result<Embedding<Self::VectorSpace>>> {
        self.embed_for(EmbeddingInput {
            text: input,
            variant: EmbeddingVariant::Document,
        })
    }

    /// Embed a [`Vec<String>`] into a vector space. Returns a list of embeddings in the same order as the inputs.
    fn embed_vec(
        &self,
        inputs: Vec<String>,
    ) -> BoxedFuture<'_, anyhow::Result<Vec<Embedding<Self::VectorSpace>>>> {
        Box::pin(async move {
            let mut embeddings = Vec::with_capacity(inputs.len());
            for input in inputs {
                embeddings.push(self.embed_string(input).await?);
            }
            Ok(embeddings)
        })
    }

    /// Embed a [`EmbeddingInput`] into a vector space
    fn embed_for(
        &self,
        input: EmbeddingInput,
    ) -> BoxedFuture<'_, anyhow::Result<Embedding<Self::VectorSpace>>>;

    /// Embed a [`Vec<String>`] into a vector space. Returns a list of embeddings in the same order as the inputs.
    fn embed_vec_for(
        &self,
        inputs: Vec<EmbeddingInput>,
    ) -> BoxedFuture<'_, anyhow::Result<Vec<Embedding<Self::VectorSpace>>>> {
        Box::pin(async move {
            let mut embeddings = Vec::with_capacity(inputs.len());
            for input in inputs {
                embeddings.push(self.embed_for(input).await?);
            }
            Ok(embeddings)
        })
    }
}

impl<E: Embedder> Embedder for Arc<E> {
    type VectorSpace = E::VectorSpace;

    fn embed_for(
            &self,
            input: EmbeddingInput,
        ) -> BoxedFuture<'_, anyhow::Result<Embedding<Self::VectorSpace>>> {
        E::embed_for(self, input)
    }

    fn embed_string(
            &self,
            input: String,
        ) -> BoxedFuture<'_, anyhow::Result<Embedding<Self::VectorSpace>>> {
        E::embed_string(&self, input)
    }

    fn embed_vec(
            &self,
            inputs: Vec<String>,
        ) -> BoxedFuture<'_, anyhow::Result<Vec<Embedding<Self::VectorSpace>>>> {
        E::embed_vec(&self, inputs)
    }

    fn embed_vec_for(
            &self,
            inputs: Vec<EmbeddingInput>,
        ) -> BoxedFuture<'_, anyhow::Result<Vec<Embedding<Self::VectorSpace>>>> {
        E::embed_vec_for(&self, inputs)
    }
}

/// The input to an embedding model. This includes the text to be embedded and the type of embedding to output.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmbeddingInput {
    /// The text to embed.
    pub text: String,
    /// The type of embedding to embed the text into.
    pub variant: EmbeddingVariant,
}

impl EmbeddingInput {
    /// Create a new embedding input.
    pub fn new(text: impl ToString, variant: EmbeddingVariant) -> Self {
        Self {
            text: text.to_string(),
            variant,
        }
    }
}

/// The type of embedding the model should output. For models that output different embeddings for queries and documents, this
///
/// For most models, the type will not effect the output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EmbeddingVariant {
    /// The model should output an embedding for a query.
    Query,
    /// The model should output an embedding for documents.
    #[default]
    Document,
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

    /// Embed a query into a vector space
    fn embed_query(
        &self,
        input: impl ToString,
    ) -> BoxedFuture<'_, anyhow::Result<Embedding<Self::VectorSpace>>> {
        self.embed_for(EmbeddingInput {
            text: input.to_string(),
            variant: EmbeddingVariant::Query,
        })
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

    /// Embed a batch of [`EmbeddingInput`] into a vector space. Returns a list of embeddings in the same order as the inputs.
    fn embed_batch_for(
        &self,
        inputs: impl IntoIterator<Item = EmbeddingInput>,
    ) -> BoxedFuture<'_, anyhow::Result<Vec<Embedding<Self::VectorSpace>>>> {
        self.embed_vec_for(inputs.into_iter().collect())
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

    fn embed_for(
        &self,
        input: EmbeddingInput,
    ) -> BoxedFuture<'_, anyhow::Result<Embedding<UnknownVectorSpace>>> {
        let future = self.0.embed_for(input);
        Box::pin(async move { future.await.map(|e| e.cast()) })
    }

    fn embed_vec_for(
        &self,
        inputs: Vec<EmbeddingInput>,
    ) -> BoxedFuture<'_, anyhow::Result<Vec<Embedding<Self::VectorSpace>>>> {
        let future = self.0.embed_vec_for(inputs);
        Box::pin(async move {
            future
                .await
                .map(|e| e.into_iter().map(|e| e.cast()).collect())
        })
    }
}
