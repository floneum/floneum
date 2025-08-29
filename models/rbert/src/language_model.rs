use std::future::Future;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::pin::Pin;

pub use crate::Bert;
use crate::BertBuilder;
use crate::BertError;
use crate::BertLoadingError;
use crate::Pooling;
pub use kalosm_language_model::{
    Embedder, EmbedderCacheExt, EmbedderExt, Embedding, EmbeddingInput, EmbeddingVariant,
    ModelBuilder,
};
use kalosm_model_types::ModelLoadingProgress;

impl ModelBuilder for BertBuilder {
    type Model = Bert;
    type Error = BertLoadingError;

    /// Start the model with a loading handler.
    async fn start_with_loading_handler(
        self,
        handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Self::Model, Self::Error> {
        self.build_with_loading_handler(handler).await
    }

    fn requires_download(&self) -> bool {
        true
    }
}

impl Bert {
    /// Embed a sentence with a specific pooling strategy.
    pub async fn embed_with_pooling(
        &self,
        input: &str,
        pooling: Pooling,
    ) -> Result<Embedding, BertError> {
        let mut tensors = self.embed_batch_raw(vec![input], pooling)?;

        let last = tensors.pop().unwrap();
        let last_slice = last
            .as_slice()
            .await
            .map_err(|err| BertError::Fusor(fusor_core::Error::BufferAsyncError(err)))?;
        Ok(Embedding::from(last_slice.into_iter().next().unwrap()))
    }

    /// Embed a batch of sentences with a specific pooling strategy.
    pub async fn embed_batch_with_pooling(
        &self,
        inputs: Vec<&str>,
        pooling: Pooling,
    ) -> Result<Vec<Embedding>, BertError> {
        let tensors = self.embed_batch_raw(inputs, pooling)?;

        let mut embeddings = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            embeddings.push(Embedding::from(
                tensor.to_vec2()?.into_iter().next().unwrap(),
            ));
        }

        Ok(embeddings)
    }
}

impl Embedder for Bert {
    type Error = BertError;

    fn embed_for(
        &self,
        input: EmbeddingInput,
    ) -> impl Future<Output = Result<Embedding, Self::Error>> + Send {
        match (&*self.embedding_search_prefix, input.variant) {
            (Some(prefix), EmbeddingVariant::Query) => {
                let mut new_input = prefix.clone();
                new_input.push_str(&input.text);
                self.embed_string(new_input)
            }
            _ => self.embed_string(input.text),
        }
    }

    fn embed_vec_for(
        &self,
        inputs: Vec<EmbeddingInput>,
    ) -> impl Future<Output = Result<Vec<Embedding>, Self::Error>> + Send {
        let inputs = inputs
            .into_iter()
            .map(
                |input| match (&*self.embedding_search_prefix, input.variant) {
                    (Some(prefix), EmbeddingVariant::Query) => {
                        let mut new_input = prefix.clone();
                        new_input.push_str(&input.text);
                        new_input
                    }
                    _ => input.text,
                },
            )
            .collect::<Vec<_>>();
        self.embed_vec(inputs)
    }

    async fn embed_string(&self, input: String) -> Result<Embedding, Self::Error> {
        let self_clone = self.clone();
        tokio::task::spawn_blocking(move || self_clone.embed_with_pooling(&input, Pooling::CLS))
            .await?
    }

    async fn embed_vec(&self, inputs: Vec<String>) -> Result<Vec<Embedding>, Self::Error> {
        let self_clone = self.clone();
        tokio::task::spawn_blocking(move || {
            let inputs_borrowed = inputs.iter().map(|s| s.as_str()).collect::<Vec<_>>();
            self_clone.embed_batch_with_pooling(inputs_borrowed, Pooling::CLS)
        })
        .await?
    }
}

impl Deref for Bert {
    type Target = dyn Fn(
        &str,
    ) -> Pin<
        Box<dyn Future<Output = Result<Embedding, BertError>> + Send + 'static>,
    >;

    fn deref(&self) -> &Self::Target {
        // https://github.com/dtolnay/case-studies/tree/master/callable-types

        // Create an empty allocation for Self.
        let uninit_callable = MaybeUninit::<Self>::uninit();
        // Move a closure that captures just self into the uninitialized memory. Closures create an anonymous type that implement
        // FnOnce. In this case, the layout of the type should just be Self because self is the only field in the closure type.
        let uninit_closure = move |text: &str| {
            let myself = unsafe { &*uninit_callable.as_ptr() };
            let self_clone = myself.clone();
            let input = text.to_string();

            Box::pin(async move {
                tokio::task::spawn_blocking(move || {
                    self_clone.embed_with_pooling(&input, Pooling::CLS)
                })
                .await?
            })
                as Pin<Box<dyn Future<Output = Result<Embedding, BertError>> + Send + 'static>>
        };

        // Make sure the layout of the closure and Self is the same.
        let size_of_closure = std::alloc::Layout::for_value(&uninit_closure);
        assert_eq!(size_of_closure, std::alloc::Layout::new::<Self>());

        // Then cast the lifetime of the closure to the lifetime of &self.
        fn cast_lifetime<'a, T>(_a: &T, b: &'a T) -> &'a T {
            b
        }
        let reference_to_closure = cast_lifetime(
            {
                // The real closure that we will never use.
                &uninit_closure
            },
            #[allow(clippy::missing_transmute_annotations)]
            // We transmute self into a reference to the closure. This is safe because we know that the closure has the same memory layout as Self so &Closure == &Self.
            unsafe {
                std::mem::transmute(self)
            },
        );

        // Cast the closure to a trait object.
        reference_to_closure as &_
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_bert() {
    use crate::BertSource;

    let bert = Bert::builder()
        .with_source(BertSource::snowflake_arctic_embed_extra_small())
        .build()
        .await
        .unwrap();
    let result = bert("The quick brown fox jumps over the lazy dog.")
        .await
        .unwrap();
    println!("{result:?}");
}
