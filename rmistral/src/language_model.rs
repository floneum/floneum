use std::sync::Arc;

use crate::InferenceSettings;
pub use crate::Mistral;
use floneumin_language_model::{CreateModel, GenerationParameters, Model, VectorSpace};
use floneumin_sample::Tokenizer;
use floneumin_streams::ChannelTextStream;

#[async_trait::async_trait]
impl CreateModel for Mistral {
    async fn start() -> Self {
        Mistral::default()
    }

    fn requires_download() -> bool {
        !Mistral::downloaded()
    }
}

#[async_trait::async_trait]
impl Model for Mistral {
    type TextStream = ChannelTextStream<String>;

    fn tokenizer(&self) -> Arc<dyn Tokenizer + Send + Sync> {
        self.get_tokenizer() as Arc<dyn Tokenizer + Send + Sync>
    }

    async fn stream_text_inner(
        &mut self,
        prompt: &str,
        generation_parameters: GenerationParameters,
    ) -> anyhow::Result<Self::TextStream> {
        let temperature = generation_parameters.temperature();
        let top_p = generation_parameters.top_p();
        let repetition_penalty = generation_parameters.repetition_penalty();
        let repetition_penalty_range = generation_parameters.repetition_penalty_range();
        let max_length = generation_parameters.max_length();
        self.run(
            InferenceSettings::new(prompt)
                .with_sample_len(max_length as usize)
                .with_temperature(temperature.into())
                .with_top_p(top_p.into())
                .with_repeat_penalty(repetition_penalty)
                .with_repeat_last_n(repetition_penalty_range as usize),
        )
        .map(Into::into)
    }
}

pub struct MistralSpace;

impl VectorSpace for MistralSpace {}
