use async_openai::types::CreateEmbeddingRequestArgs;
use async_openai::{
    types::{CompletionResponseStream, CreateCompletionRequestArgs},
    Client,
};
use futures_util::Stream;
use kalosm_sample::Tokenizer;
use std::sync::Arc;

use crate::{CreateModel, Embedder, Embedding, GenerationParameters, VectorSpace};

macro_rules! openai_model {
    ($ty: ident, $tybuilder: ident, $model: literal) => {
        /// A model that uses OpenAI's API.
        pub struct $ty {
            client: Client<async_openai::config::OpenAIConfig>,
        }

        /// A builder for
        #[doc = $model]
        #[derive(Debug, Default)]
        pub struct $tybuilder {
            config: async_openai::config::OpenAIConfig,
        }

        impl $tybuilder {
            /// Creates a new builder
            pub fn new() -> Self {
                Self {
                    config: Default::default(),
                }
            }

            /// Sets the API key for the builder.
            pub fn with_api_key(mut self, api_key: &str) -> Self {
                self.config = self.config.with_api_key(api_key);
                self
            }

            /// Set the base URL of the API.
            pub fn with_base_url(mut self, base_url: &str) -> Self {
                self.config = self.config.with_api_base(base_url);
                self
            }

            /// Set the organization ID for the builder.
            pub fn with_organization_id(mut self, organization_id: &str) -> Self {
                self.config = self.config.with_org_id(organization_id);
                self
            }

            /// Build the model.
            pub fn build(self) -> $ty {
                $ty {
                    client: Client::with_config(self.config),
                }
            }
        }

        impl $ty {
            /// Creates a new builder
            pub fn builder() -> $tybuilder {
                $tybuilder::new()
            }
        }

        impl $ty {
            /// Creates a new builder
            pub fn builder() -> $tybuilder {
                $tybuilder::new()
            }
        }


        impl Default for $ty {
            fn default() -> Self {
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

openai_model!(Gpt3_5, Gpt3_5Builder, "gpt-3.5-turbo");
openai_model!(Gpt4, Gpt4Builder, "text-davinci-003");

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

/// A builder for the Ada embedder.
#[derive(Debug, Default)]
pub struct AdaEmbedderBuilder {
    config: async_openai::config::OpenAIConfig,
}

impl AdaEmbedderBuilder {
    /// Creates a new builder
    pub fn new() -> Self {
        Self {
            config: Default::default(),
        }
    }

    /// Sets the API key for the builder.
    pub fn with_api_key(mut self, api_key: &str) -> Self {
        self.config = self.config.with_api_key(api_key);
        self
    }

    /// Set the base URL of the API.
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.config = self.config.with_api_base(base_url);
        self
    }

    /// Set the organization ID for the builder.
    pub fn with_organization_id(mut self, organization_id: &str) -> Self {
        self.config = self.config.with_org_id(organization_id);
        self
    }

    /// Build the model.
    pub fn build(self) -> AdaEmbedder {
        AdaEmbedder {
            client: Client::with_config(self.config),
        }
    }
}

impl AdaEmbedder {
    /// Creates a new builder
    pub fn builder() -> AdaEmbedderBuilder {
        AdaEmbedderBuilder::new()
    }
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
