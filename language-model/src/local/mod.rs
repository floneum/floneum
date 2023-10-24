pub use crate::local::session::*;
use crate::{embedding::Embedding, model::*};
use floneumin_sample::Tokenizer;
use floneumin_streams::ChannelTextStream;
use llm::InferenceSessionConfig;
use llm_samplers::configure::SamplerChainBuilder;
use llm_samplers::prelude::Sampler;
use llm_samplers::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;

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
        impl crate::model::CreateModel for LocalSession<$space> {
            async fn start() -> Self {
                let model = Self::model_type().download().await;
                let session = model.start_session(InferenceSessionConfig {
                    n_batch: 64,
                    n_threads: num_cpus::get(),
                    ..Default::default()
                });

                LocalSession::new(model, session)
            }

            fn requires_download() -> bool {
                Self::model_type().requires_download()
            }
        }

        #[async_trait::async_trait]
        impl crate::model::Model for LocalSession<$space> {
            type TextStream = ChannelTextStream<String>;
            type SyncModel = crate::SyncModelNotSupported;

            fn tokenizer(&self) -> Arc<dyn Tokenizer + Send + Sync> {
                self.get_tokenizer() as Arc<dyn Tokenizer + Send + Sync>
            }

            async fn stream_text_inner(
                &mut self,
                prompt: &str,
                generation_parameters: GenerationParameters,
            ) -> anyhow::Result<Self::TextStream> {
                Ok(self.infer(prompt.to_string(), generation_parameters).await)
            }

            async fn stream_text_with_sampler(
                &mut self,
                prompt: &str,
                max_tokens: Option<u32>,
                stop_on: Option<&str>,
                sampler: Arc<Mutex<dyn Sampler<u32, f32>>>,
            ) -> anyhow::Result<Self::TextStream> {
                Ok(self
                    .infer_sampler(prompt.to_string(), max_tokens, stop_on, sampler)
                    .await)
            }
        }

        #[async_trait::async_trait]
        impl crate::model::Embedder<$space> for LocalSession<$space> {
            async fn embed(&mut self, input: &str) -> anyhow::Result<Embedding<$space>> {
                self.get_embedding(input).await
            }

            async fn embed_batch(
                &mut self,
                inputs: &[&str],
            ) -> anyhow::Result<Vec<Embedding<$space>>> {
                let mut result = Vec::new();
                for input in inputs {
                    result.push(self.get_embedding(input).await?);
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
local_model!(ModelType::Mpt(MptType::Base), MptBaseSpace);
local_model!(ModelType::Mpt(MptType::Story), MptStorySpace);
local_model!(ModelType::Mpt(MptType::Instruct), MptInstructSpace);
local_model!(ModelType::Mpt(MptType::Chat), MptChatSpace);
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
    /// Create a sampler chain from the generation parameters.
    pub fn sampler(self) -> SamplerChain {
        use llm_samplers::configure::SamplerSlot;
        let GenerationParameters {
            temperature,
            tau,
            eta,
            mu,
            repetition_penalty,
            repetition_penalty_range,
            max_length: _,
            stop_on: _,
        } = self;
        SamplerChainBuilder::from([
            (
                "repetition",
                SamplerSlot::new_chain(
                    move || {
                        Box::new(
                            SampleRepetition::default()
                                .penalty(repetition_penalty)
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
                "mirostat2",
                SamplerSlot::new_single(
                    move || Box::new(SampleMirostat2::default().tau(tau).eta(eta).mu(mu)),
                    Option::<SampleTopK>::None,
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
