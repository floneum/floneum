use futures_util::{Future, StreamExt};
use kalosm_common::*;
use kalosm_streams::text_stream::ChannelTextStream;
use llm_samplers::types::Sampler;
use std::sync::{Arc, Mutex};
use tokenizers::tokenizer::Tokenizer;

use crate::{Embedder, Embedding, ModelBuilder, VectorSpace};

use super::OpenAICompatibleClient;

/// A model that uses OpenAI's API.
pub struct RemoteOpenAICompatibleModel {
    model: String,
    client: OpenAICompatibleClient,
}

/// A builder for any remote OpenAI compatible model.
#[derive(Debug, Default)]
pub struct RemoteOpenAICompatibleModelBuilder<const WITH_NAME: bool> {
    model: Option<String>,
    client: OpenAICompatibleClient,
}

impl RemoteOpenAICompatibleModelBuilder<false> {
    /// Creates a new builder
    pub fn new() -> Self {
        Self {
            model: None,
            client: Default::default(),
        }
    }
}

impl<const WITH_NAME: bool> RemoteOpenAICompatibleModelBuilder<WITH_NAME> {
    /// Set the name of the model to use.
    pub fn with_model(mut self, model: impl ToString) -> RemoteOpenAICompatibleModelBuilder<true> {
        RemoteOpenAICompatibleModelBuilder {
            model: Some(model.to_string()),
            client: self.client,
        }
    }

    /// Set the client used to make requests to the OpenAI API.
    pub fn with_client(mut self, client: OpenAICompatibleClient) -> Self {
        self.client = client;
        self
    }
}

/// An error that can occur when building a remote OpenAI model.
#[derive(Debug, thiserror::Error)]
pub enum OpenAIBuilderError {
    /// The OpenAI API key is not set and the environment variable `OPENAI_API_KEY` is not set.
    #[error(
        "The OpenAI API key is not set and the environment variable `OPENAI_API_KEY` is not set"
    )]
    OpenAIApiKeyNotSet,
}

impl RemoteOpenAICompatibleModelBuilder<true> {
    /// Build the model.
    pub fn build(self) -> RemoteOpenAICompatibleModel {
        RemoteOpenAICompatibleModel {
            model: self.config.model.unwrap(),
            client: self.client,
        }
    }
}

impl RemoteOpenAICompatibleModel {
    /// Creates a new builder
    pub fn builder() -> RemoteOpenAICompatibleModelBuilder<false> {
        RemoteOpenAICompatibleModelBuilder::new()
    }
}

// impl crate::model::Model for RemoteOpenAICompatibleModel {
//     type TextStream = ChannelTextStream;
//     type SyncModel = RemoteOpenAINotSyncModel;
//     type Error = RemoteOpenAICompatibleModelError;

//     fn tokenizer(&self) -> Arc<Tokenizer> {
//         panic!("OpenAI does not expose tokenization")
//     }

//     async fn stream_text_inner(
//         &self,
//         prompt: &str,
//         generation_parameters: GenerationParameters,
//     ) -> Result<Self::TextStream, Self::Error> {
//         let mut builder = CreateCompletionRequestArgs::default();
//         builder
//             .model(&self.model)
//             .n(1)
//             .prompt(prompt)
//             .stream(true)
//             .frequency_penalty(generation_parameters.repetition_penalty)
//             .temperature(generation_parameters.temperature)
//             .max_tokens(generation_parameters.max_length as u16);
//         if let Some(stop_on) = generation_parameters.stop_on {
//             builder.stop(stop_on);
//         }
//         let request = builder.build()?;

//         let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

//         let mut stream = self.client.completions().create_stream(request).await?;

//         tokio::spawn(async move {
//             while let Some(response) = stream.next().await {
//                 match response {
//                     Ok(response) => {
//                         let text = response.choices[0].text.clone();
//                         if tx.send(text).is_err() {
//                             break;
//                         }
//                     }
//                     Err(e) => {
//                         log::error!("Error in OpenAI stream: {}", e);
//                         break;
//                     }
//                 }
//             }

//             Ok::<(), Self::Error>(())
//         });

//         Ok(rx.into())
//     }

//     fn run_sync_raw(
//         &self,
//         _f: Box<
//             dyn for<'a> FnOnce(
//                     &'a mut Self::SyncModel,
//                 )
//                     -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>>
//                 + Send,
//         >,
//     ) -> Result<(), Self::Error> {
//         Err(RemoteOpenAICompatibleModelError::OpenAIDoesNotSupportSyncModel)
//     }

//     async fn stream_text_with_sampler(
//         &self,
//         _prompt: &str,
//         _max_tokens: Option<u32>,
//         _stop_on: Option<&str>,
//         _sampler: Arc<Mutex<dyn Sampler>>,
//     ) -> Result<Self::TextStream, Self::Error> {
//         Err(RemoteOpenAICompatibleModelError::OpenAIDoesNotSupportCustomSamplers)
//     }
// }

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

openai_completion_model!(Gpt3_5, Gpt3_5Builder, "gpt-3.5-turbo-instruct");
// The rest of the openai models only support the chat API which currently isn't supported for remote models in kalosm
// openai_chat_model!(Gpt4, Gpt4Builder, "gpt-4");
// openai_chat_model!(Gpt4Turbo, Gpt4TurboBuilder, "gpt-4-turbo");
// openai_chat_model!(Gpt4O, Gpt4OBuilder, "gpt-4o");
// openai_chat_model!(Gpt4Mini, Gpt4MiniBuilder, "gpt-4o-mini");
