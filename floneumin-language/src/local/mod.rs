pub use crate::local::bert::*;
pub use crate::local::phi::*;
pub use crate::local::session::*;
use crate::{download::download, embedding::Embedding, model::*};
use floneumin_streams::sender::ChannelTextStream;
use futures_util::StreamExt;
use llm::InferenceSessionConfig;

mod bert;
mod phi;
mod session;

pub(crate) trait LocalModelType {
    fn model_type() -> ModelType;
}

macro_rules! local_model {
    ($ty: expr, $space: ident) => {
        impl LocalModelType for LocalSession<$space> {
            fn model_type() -> ModelType {
                $ty
            }
        }

        #[async_trait::async_trait]
        impl crate::model::Model for LocalSession<$space> {
            type TextStream = ChannelTextStream<String>;

            async fn start() -> Self {
                let model = download(Self::model_type()).await;
                let session = model.start_session(InferenceSessionConfig {
                    n_batch: 64,
                    n_threads: num_cpus::get(),
                    ..Default::default()
                });

                LocalSession::new(model, session)
            }

            async fn generate_text(
                &mut self,
                prompt: &str,
                generation_parameters: crate::model::GenerationParameters,
            ) -> anyhow::Result<String> {
                let mut text = String::new();
                let mut stream = self.stream_text(prompt, generation_parameters).await?;
                while let Some(new) = stream.next().await {
                    text.push_str(&new);
                }
                Ok(text)
            }

            async fn stream_text(
                &mut self,
                prompt: &str,
                generation_parameters: crate::model::GenerationParameters,
            ) -> anyhow::Result<Self::TextStream> {
                let max_tokens = generation_parameters.max_length();
                Ok(self.infer(prompt.to_string(), Some(max_tokens), None).await)
            }
        }

        #[async_trait::async_trait]
        impl crate::model::Embedder<$space> for LocalSession<$space> {
            async fn embed(input: &str) -> anyhow::Result<Embedding<$space>> {
                Self::start().await.get_embedding(input).await
            }

            async fn embed_batch(inputs: &[&str]) -> anyhow::Result<Vec<Embedding<$space>>> {
                let session = Self::start().await;
                let mut result = Vec::new();
                for input in inputs {
                    result.push(session.get_embedding(input).await?);
                }
                Ok(result)
            }
        }
    };
}

local_model!(ModelType::Llama(LlamaType::Vicuna), VicunaSpace);
local_model!(ModelType::Llama(LlamaType::Guanaco), GuanacoSpace);
local_model!(ModelType::Llama(LlamaType::WizardLm), WizardLmSpace);
local_model!(ModelType::Llama(LlamaType::Orca), OrcaSpace);
local_model!(
    ModelType::Llama(LlamaType::LlamaSevenChat),
    LlamaSevenChatSpace
);
local_model!(
    ModelType::Llama(LlamaType::LlamaThirteenChat),
    LlamaThirteenChatSpace
);
local_model!(ModelType::Mpt(MptType::Base), BaseSpace);
local_model!(ModelType::Mpt(MptType::Story), StorySpace);
local_model!(ModelType::Mpt(MptType::Instruct), InstructSpace);
local_model!(ModelType::Mpt(MptType::Chat), ChatSpace);
local_model!(
    ModelType::GptNeoX(GptNeoXType::LargePythia),
    LargePythiaSpace
);
local_model!(ModelType::GptNeoX(GptNeoXType::TinyPythia), TinyPythiaSpace);
local_model!(
    ModelType::GptNeoX(GptNeoXType::DollySevenB),
    DollySevenBSpace
);
local_model!(ModelType::GptNeoX(GptNeoXType::StableLm), StableLmSpace);
