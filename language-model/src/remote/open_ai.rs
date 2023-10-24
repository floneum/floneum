use async_openai::types::CreateEmbeddingRequestArgs;
use async_openai::{
    types::{CompletionResponseStream, CreateCompletionRequestArgs},
    Client,
};
use floneumin_sample::Tokenizer;
use futures_util::Stream;
use std::sync::Arc;

use crate::{CreateModel, Embedder, Embedding, GenerationParameters, VectorSpace};

macro_rules! openai_model {
    ($ty: ident, $model: literal) => {
        /// A model that uses OpenAI's API.
        pub struct $ty {
            client: Client<async_openai::config::OpenAIConfig>,
        }

        impl $ty {
            /// Creates a new OpenAI model.
            pub fn new() -> Self {
                Self {
                    client: Client::new(),
                }
            }
        }

        #[async_trait::async_trait]
        impl CreateModel for $ty {
            async fn start() -> Self {
                let client = Client::new();
                $ty { client }
            }

            fn requires_download() -> bool {
                false
            }
        }

        #[async_trait::async_trait]
        impl crate::model::Model for $ty {
            type TextStream = MappedResponseStream;
            type SyncModel = crate::SyncModelNotSupported;

            fn tokenizer(&self) -> Arc<dyn Tokenizer + Send + Sync> {
                panic!("OpenAI does not expose tokenization")
            }

            async fn stream_text_inner(
                &mut self,
                prompt: &str,
                generation_parameters: GenerationParameters,
            ) -> anyhow::Result<Self::TextStream> {
                let request = CreateCompletionRequestArgs::default()
                    .model($model)
                    .n(1)
                    .prompt(prompt)
                    .stream(true)
                    .frequency_penalty(generation_parameters.repetition_penalty)
                    .temperature(generation_parameters.temperature)
                    .stop(
                        generation_parameters
                            .stop_on
                            .iter()
                            .cloned()
                            .collect::<Vec<String>>(),
                    )
                    .max_tokens(generation_parameters.max_length as u16)
                    .build()?;

                Ok(MappedResponseStream {
                    inner: self.client.completions().create_stream(request).await?,
                })
            }
        }
    };
}

openai_model!(Gpt3_5, "gpt-3.5-turbo");
openai_model!(Gpt4, "text-davinci-003");

/// A stream of text from OpenAI's API.
#[pin_project::pin_project]
pub struct MappedResponseStream {
    #[pin]
    inner: CompletionResponseStream,
}

impl Stream for MappedResponseStream {
    type Item = String;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.project();
        this.inner.poll_next(cx).map(|opt| {
            opt.and_then(|res| match res {
                Ok(res) => Some(
                    res.choices
                        .iter()
                        .map(|c| c.text.clone())
                        .collect::<Vec<_>>()
                        .join(""),
                ),
                Err(e) => {
                    tracing::error!("Error from OpenAI: {}", e);
                    None
                }
            })
        })
    }
}

/// An embedder that uses OpenAI's API for the Ada embedding model.
#[derive(Debug)]
pub struct AdaEmbedder {
    client: Client<async_openai::config::OpenAIConfig>,
}

impl Default for AdaEmbedder {
    fn default() -> Self {
        Self {
            client: Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl CreateModel for AdaEmbedder {
    async fn start() -> Self {
        let client = Client::new();
        AdaEmbedder { client }
    }

    fn requires_download() -> bool {
        false
    }
}

/// The embedding space for the Ada embedding model.
pub struct AdaEmbedding;

impl VectorSpace for AdaEmbedding {}

#[async_trait::async_trait]
impl Embedder<AdaEmbedding> for AdaEmbedder {
    /// Embed a single string.
    async fn embed(&mut self, input: &str) -> anyhow::Result<Embedding<AdaEmbedding>> {
        let request = CreateEmbeddingRequestArgs::default()
            .model("text-embedding-ada-002")
            .input([input])
            .build()?;

        let response = self.client.embeddings().create(request).await?;

        let embedding = Embedding::from(response.data[0].embedding.iter().copied());

        Ok(embedding)
    }
}
