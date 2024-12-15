use async_openai::error::OpenAIError;
use async_openai::types::CreateEmbeddingRequestArgs;
use async_openai::{types::CreateCompletionRequestArgs, Client};
use futures_util::{Future, StreamExt};
use kalosm_common::*;
use kalosm_streams::text_stream::ChannelTextStream;
use llm_samplers::types::Sampler;
use std::sync::{Arc, Mutex};
use tokenizers::tokenizer::Tokenizer;

use crate::{Embedder, Embedding, GenerationParameters, ModelBuilder, SyncModel, VectorSpace};

/// A model that uses OpenAI's API.
pub struct RemoteOpenAICompatibleModel {
    model: String,
    client: Client<async_openai::config::OpenAIConfig>,
}

/// A builder for any remote OpenAI compatible model.
#[derive(Debug, Default)]
pub struct RemoteOpenAICompatibleModelBuilder<const WITH_NAME: bool> {
    model: Option<String>,
    config: async_openai::config::OpenAIConfig,
}

impl RemoteOpenAICompatibleModelBuilder<false> {
    /// Creates a new builder
    pub fn new() -> Self {
        Self {
            model: None,
            config: Default::default(),
        }
    }

    /// Set the name of the model to use.
    pub fn with_model(self, model: impl ToString) -> RemoteOpenAICompatibleModelBuilder<true> {
        RemoteOpenAICompatibleModelBuilder {
            model: Some(model.to_string()),
            config: self.config,
        }
    }
}

impl<const WITH_NAME: bool> RemoteOpenAICompatibleModelBuilder<WITH_NAME> {
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
}

impl RemoteOpenAICompatibleModelBuilder<true> {
    /// Build the model.
    pub fn build(self) -> RemoteOpenAICompatibleModel {
        RemoteOpenAICompatibleModel {
            model: self.model.unwrap(),
            client: Client::with_config(self.config),
        }
    }
}

impl RemoteOpenAICompatibleModel {
    /// Creates a new builder
    pub fn builder() -> RemoteOpenAICompatibleModelBuilder<false> {
        RemoteOpenAICompatibleModelBuilder::new()
    }
}

/// A mock sync model for OpenAI which **does not** support sync models.
pub struct RemoteOpenAINotSyncModel;

impl SyncModel for RemoteOpenAINotSyncModel {
    type Session = ();
    type Error = RemoteOpenAICompatibleModelError;

    fn new_session(&self) -> Result<Self::Session, Self::Error> {
        Err(RemoteOpenAICompatibleModelError::OpenAIDoesNotSupportSyncModel)
    }

    fn feed_text(
        &self,
        _session: &mut Self::Session,
        _prompt: &str,
        _out: &mut Vec<f32>,
    ) -> Result<(), Self::Error> {
        Err(RemoteOpenAICompatibleModelError::OpenAIDoesNotSupportSyncModel)
    }

    fn feed_tokens(
        &self,
        _session: &mut Self::Session,
        _tokens: &[u32],
        _out: &mut Vec<f32>,
    ) -> Result<(), Self::Error> {
        Err(RemoteOpenAICompatibleModelError::OpenAIDoesNotSupportSyncModel)
    }

    fn stop_token(&self) -> Result<u32, Self::Error> {
        Err(RemoteOpenAICompatibleModelError::OpenAIDoesNotSupportSyncModel)
    }

    fn tokenizer(&self) -> Arc<Tokenizer> {
        panic!("OpenAI does not expose tokenization")
    }
}

#[async_trait::async_trait]
impl crate::model::Model for RemoteOpenAICompatibleModel {
    type TextStream = ChannelTextStream;
    type SyncModel = RemoteOpenAINotSyncModel;
    type Error = RemoteOpenAICompatibleModelError;

    fn tokenizer(&self) -> Arc<Tokenizer> {
        panic!("OpenAI does not expose tokenization")
    }

    async fn stream_text_inner(
        &self,
        prompt: &str,
        generation_parameters: GenerationParameters,
    ) -> Result<Self::TextStream, Self::Error> {
        let mut builder = CreateCompletionRequestArgs::default();
        builder
            .model(&self.model)
            .n(1)
            .prompt(prompt)
            .stream(true)
            .frequency_penalty(generation_parameters.repetition_penalty)
            .temperature(generation_parameters.temperature)
            .max_tokens(generation_parameters.max_length as u16);
        if let Some(stop_on) = generation_parameters.stop_on {
            builder.stop(stop_on);
        }
        let request = builder.build()?;

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

            Ok::<(), Self::Error>(())
        });

        Ok(rx.into())
    }

    fn run_sync_raw(
        &self,
        _f: Box<
            dyn for<'a> FnOnce(
                    &'a mut Self::SyncModel,
                )
                    -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>>
                + Send,
        >,
    ) -> Result<(), Self::Error> {
        Err(RemoteOpenAICompatibleModelError::OpenAIDoesNotSupportSyncModel)
    }

    async fn stream_text_with_sampler(
        &self,
        _prompt: &str,
        _max_tokens: Option<u32>,
        _stop_on: Option<&str>,
        _sampler: Arc<Mutex<dyn Sampler>>,
    ) -> Result<Self::TextStream, Self::Error> {
        Err(RemoteOpenAICompatibleModelError::OpenAIDoesNotSupportCustomSamplers)
    }
}

macro_rules! openai_completion_model {
    ($ty: ident, $tybuilder: ident, $model: literal) => {
        /// A model that uses OpenAI's API.
        pub struct $ty {
            inner: RemoteOpenAICompatibleModel,
        }

        /// A builder for
        #[doc = $model]
        #[derive(Debug, Default)]
        pub struct $tybuilder {
            inner: RemoteOpenAICompatibleModelBuilder<true>,
        }

        impl $tybuilder {
            /// Creates a new builder
            pub fn new() -> Self {
                Self {
                    inner: RemoteOpenAICompatibleModelBuilder::new().with_model($model),
                }
            }

            /// Sets the API key for the builder.
            pub fn with_api_key(mut self, api_key: &str) -> Self {
                self.inner = self.inner.with_api_key(api_key);
                self
            }

            /// Set the base URL of the API.
            pub fn with_base_url(mut self, base_url: &str) -> Self {
                self.inner = self.inner.with_base_url(base_url);
                self
            }

            /// Set the organization ID for the builder.
            pub fn with_organization_id(mut self, organization_id: &str) -> Self {
                self.inner = self.inner.with_organization_id(organization_id);
                self
            }

            /// Build the model.
            pub fn build(self) -> $ty {
                $ty {
                    inner: self.inner.build(),
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
            type Error = OpenAIError;

            async fn start_with_loading_handler(
                self,
                _: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
            ) -> Result<$ty, Self::Error> {
                Ok($ty {
                    inner: self.inner.build(),
                })
            }

            fn requires_download(&self) -> bool {
                false
            }
        }

        #[async_trait::async_trait]
        impl crate::model::Model for $ty {
            type TextStream = ChannelTextStream;
            type SyncModel = RemoteOpenAINotSyncModel;
            type Error = RemoteOpenAICompatibleModelError;

            fn tokenizer(&self) -> Arc<Tokenizer> {
                panic!("OpenAI does not expose tokenization")
            }

            async fn stream_text_inner(
                &self,
                prompt: &str,
                generation_parameters: GenerationParameters,
            ) -> Result<Self::TextStream, Self::Error> {
                self.inner
                    .stream_text_inner(prompt, generation_parameters)
                    .await
            }

            fn run_sync_raw(
                &self,
                _f: Box<
                    dyn for<'a> FnOnce(
                            &'a mut Self::SyncModel,
                        ) -> std::pin::Pin<
                            Box<dyn std::future::Future<Output = ()> + 'a>,
                        > + Send,
                >,
            ) -> Result<(), Self::Error> {
                Err(RemoteOpenAICompatibleModelError::OpenAIDoesNotSupportSyncModel)
            }

            async fn stream_text_with_sampler(
                &self,
                _prompt: &str,
                _max_tokens: Option<u32>,
                _stop_on: Option<&str>,
                _sampler: Arc<Mutex<dyn Sampler>>,
            ) -> Result<Self::TextStream, Self::Error> {
                Err(RemoteOpenAICompatibleModelError::OpenAIDoesNotSupportCustomSamplers)
            }
        }
    };
}

/// An error that can occur when running a [`RemoteOpenAICompatibleModel`].
#[derive(Debug, thiserror::Error)]
pub enum RemoteOpenAICompatibleModelError {
    /// OpenAI does not expose tokenization
    #[error("OpenAI does not expose tokenization")]
    OpenAIDoesNotExposeTokenization,
    /// OpenAI does not support sync models
    #[error("OpenAI does not support sync models")]
    OpenAIDoesNotSupportSyncModel,
    /// OpenAI does not support custom samplers
    #[error("OpenAI does not support custom samplers")]
    OpenAIDoesNotSupportCustomSamplers,
    /// An error from the OpenAI API
    #[error("OpenAI API error: {0}")]
    OpenAIAPIError(#[from] OpenAIError),
}

openai_completion_model!(Gpt3_5, Gpt3_5Builder, "gpt-3.5-turbo-instruct");
// The rest of the openai models only support the chat API which currently isn't supported for remote models in kalosm
// openai_chat_model!(Gpt4, Gpt4Builder, "gpt-4");
// openai_chat_model!(Gpt4Turbo, Gpt4TurboBuilder, "gpt-4-turbo");
// openai_chat_model!(Gpt4O, Gpt4OBuilder, "gpt-4o");
// openai_chat_model!(Gpt4Mini, Gpt4MiniBuilder, "gpt-4o-mini");

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
    type Error = OpenAIError;

    async fn start_with_loading_handler(
        self,
        _: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<AdaEmbedder, Self::Error> {
        Ok(self.build())
    }

    fn requires_download(&self) -> bool {
        false
    }
}

/// The embedding space for the Ada embedding model.
pub struct AdaEmbedding;

impl VectorSpace for AdaEmbedding {}

impl AdaEmbedder {
    /// The model ID for the Ada embedding model.
    pub const MODEL_ID: &'static str = "text-embedding-ada-002";
}

impl Embedder for AdaEmbedder {
    type VectorSpace = AdaEmbedding;
    type Error = OpenAIError;

    fn embed_for(
        &self,
        input: crate::EmbeddingInput,
    ) -> impl Future<Output = Result<Embedding<Self::VectorSpace>, Self::Error>> + Send {
        self.embed_string(input.text)
    }

    fn embed_vec_for(
        &self,
        inputs: Vec<crate::EmbeddingInput>,
    ) -> impl Future<Output = Result<Vec<Embedding<Self::VectorSpace>>, Self::Error>> + Send {
        let inputs = inputs
            .into_iter()
            .map(|input| input.text)
            .collect::<Vec<_>>();
        self.embed_vec(inputs)
    }

    /// Embed a single string.
    fn embed_string(
        &self,
        input: String,
    ) -> impl Future<Output = Result<Embedding<AdaEmbedding>, Self::Error>> + Send {
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
    ) -> impl Future<Output = Result<Vec<Embedding<AdaEmbedding>>, Self::Error>> + Send {
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
