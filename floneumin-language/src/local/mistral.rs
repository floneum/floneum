use crate::embedding::VectorSpace;
use crate::local::Arc;
use floneumin_sample::Tokenizer;
use floneumin_streams::sender::ChannelTextStream;
use rmistral::InferenceSettings;
pub use rmistral::{self, Mistral};

#[async_trait::async_trait]
impl crate::model::Model for Mistral {
    type TextStream = ChannelTextStream<String>;

    async fn start() -> Self {
        Mistral::default()
    }

    fn tokenizer(&self) -> Arc<dyn Tokenizer + Send + Sync> {
        self.get_tokenizer() as Arc<dyn Tokenizer + Send + Sync>
    }

    async fn stream_text(
        &mut self,
        prompt: &str,
        generation_parameters: crate::model::GenerationParameters,
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
