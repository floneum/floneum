use async_openai::types::CreateEmbeddingRequestArgs;
use async_openai::{types::CreateCompletionRequestArgs, Client};
use futures_util::{Future, StreamExt};
use kalosm_common::*;
use kalosm_sample::Tokenizer;
use kalosm_streams::text_stream::ChannelTextStream;
use std::pin::Pin;
use std::sync::Arc;

use crate::{Embedder, Embedding, GenerationParameters, ModelBuilder, VectorSpace};

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

        impl Default for $ty {
            fn default() -> Self {
                Self::builder().build()
            }
        }

        #[async_trait::async_trait]
        impl ModelBuilder for $tybuilder {
            type Model = $ty;

            async fn start_with_loading_handler(
                self,
                _: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
            ) -> anyhow::Result<$ty> {
                let client = Client::new();
                Ok($ty { client })
            }

            fn requires_download(&self) -> bool {
                false
            }
        }

        #[async_trait::async_trait]
        impl crate::model::Model for $ty {
            type TextStream = ChannelTextStream<String>;
            type SyncModel = crate::SyncModelNotSupported;

            fn tokenizer(&self) -> Arc<dyn Tokenizer + Send + Sync> {
                panic!("OpenAI does not expose tokenization")
            }

            async fn stream_text_inner(
                &self,
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

                let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

                let mut stream = self.client.completions().create_stream(request).await?;

                tokio::spawn(async move {
                    while let Some(response) = stream.next().await {
                        match response {
                            Ok(response) => {
                                let text = response.choices[0].text.clone();
                                if tx.send(text).is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                log::error!("Error in OpenAI stream: {}", e);
                                break;
                            }
                        }
                    }

                    Ok::<(), anyhow::Error>(())
                });

                Ok(rx.into())
            }
        }
    };
}

openai_model!(Gpt3_5, Gpt3_5Builder, "gpt-3.5-turbo");
openai_model!(Gpt4, Gpt4Builder, "text-davinci-003");

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
impl ModelBuilder for AdaEmbedderBuilder {
    type Model = AdaEmbedder;

    async fn start_with_loading_handler(
        self,
        _: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> anyhow::Result<AdaEmbedder> {
        Ok(self.build())
    }

    fn requires_download(&self) -> bool {
        false
    }
}

/// The embedding space for the Ada embedding model.
pub struct AdaEmbedding;

impl AdaEmbedding {
    /// The model ID for the Ada embedding model.
    pub const MODEL_ID: &'static str = "text-embedding-ada-002";
}

impl VectorSpace for AdaEmbedding {}

impl Embedder for AdaEmbedder {
    type VectorSpace = AdaEmbedding;

    /// Embed a single string.
    fn embed_string(
        &self,
        input: String,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<Embedding<AdaEmbedding>>> + Send + '_>> {
        Box::pin(async move {
            let request = CreateEmbeddingRequestArgs::default()
                .model(Self::MODEL_ID)
                .input([input])
                .build()?;
            let response = self.client.embeddings().create(request).await?;

            let embedding = Embedding::from(response.data[0].embedding.iter().copied());

            Ok(embedding)
        })
    }

    /// Embed a single string.
    fn embed_vec(
        &self,
        input: Vec<String>,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<Vec<Embedding<AdaEmbedding>>>> + Send + '_>>
    {
        Box::pin(async move {
            let request = CreateEmbeddingRequestArgs::default()
                .model(Self::MODEL_ID)
                .input(input)
                .build()?;
            let response = self.client.embeddings().create(request).await?;

            Ok(response
                .data
                .into_iter()
                .map(|data| Embedding::from(data.embedding.into_iter()))
                .collect())
        })
    }
}
