pub use crate::local::bert::*;
pub use crate::local::mistral::*;
pub use crate::local::phi::*;
pub use crate::local::session::*;
use crate::sample::Tokenizer;
use crate::{download::download, embedding::Embedding, model::*};
use floneumin_streams::sender::ChannelTextStream;
use futures_util::StreamExt;
use llm::InferenceSessionConfig;
use llm_samplers::configure::SamplerChainBuilder;
use llm_samplers::prelude::Sampler;
use llm_samplers::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;

mod bert;
mod mistral;
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

            fn tokenizer(&self) -> Arc<dyn Tokenizer + Send + Sync> {
                self.get_tokenizer() as Arc<dyn Tokenizer + Send + Sync>
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
                Ok(self.infer(prompt.to_string(), generation_parameters).await)
            }

            async fn stream_text_with_sampler(
                &mut self,
                prompt: &str,
                max_tokens: Option<u32>,
                sampler: Arc<Mutex<dyn Sampler<u32, f32>>>,
            ) -> anyhow::Result<Self::TextStream> {
                Ok(self
                    .infer_sampler(prompt.to_string(), max_tokens, sampler)
                    .await)
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

impl crate::model::GenerationParameters {
    pub fn sampler(self) -> SamplerChain {
        use llm_samplers::configure::SamplerSlot;
        let GenerationParameters {
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            repetition_penalty_range,
            max_length: _,
        } = self;
        SamplerChainBuilder::from([
            (
                "repetition",
                SamplerSlot::new_chain(
                    move || {
                        Box::new(
                            SampleRepetition::default()
                                .penalty(repetition_penalty as f32)
                                .last_n(repetition_penalty_range as usize),
                        )
                    },
                    [],
                ),
            ),
            (
                "freqpresence",
                SamplerSlot::new_chain(
                    move || Box::new(SampleFreqPresence::default().last_n(64)),
                    [],
                ),
            ),
            (
                "seqrepetition",
                SamplerSlot::new_chain(move || Box::<SampleSeqRepetition>::default(), []),
            ),
            (
                "topk",
                SamplerSlot::new_single(
                    move || Box::new(SampleTopK::default().k(top_k as usize)),
                    Option::<SampleTopK>::None,
                ),
            ),
            (
                "tailfree",
                SamplerSlot::new_single(
                    move || Box::<SampleTailFree>::default(),
                    Option::<SampleTailFree>::None,
                ),
            ),
            (
                "locallytypical",
                SamplerSlot::new_single(
                    move || Box::<SampleLocallyTypical>::default(),
                    Option::<SampleLocallyTypical>::None,
                ),
            ),
            (
                "topp",
                SamplerSlot::new_single(
                    move || Box::new(SampleTopP::default().p(top_p)),
                    Option::<SampleTopP>::None,
                ),
            ),
            (
                "temperature",
                SamplerSlot::new_single(
                    move || Box::new(SampleTemperature::default().temperature(temperature)),
                    Option::<SampleTemperature>::None,
                ),
            ),
            (
                "randdistrib",
                SamplerSlot::new_static(|| Box::<SampleRandDistrib>::default()),
            ),
        ])
        .into_chain()
    }
}
