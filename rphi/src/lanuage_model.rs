use crate::InferenceSettings;
pub use crate::Phi;
use floneumin_language_model::*;
use floneumin_streams::ChannelTextStream;
use std::sync::Arc;
use std::sync::Mutex;

#[async_trait::async_trait]
impl CreateModel for Phi {
    async fn start() -> Self {
        Phi::default()
    }

    fn requires_download() -> bool {
        !Phi::downloaded()
    }
}

#[async_trait::async_trait]
impl Model for Phi {
    type TextStream = ChannelTextStream<String>;

    fn tokenizer(&self) -> Arc<dyn floneumin_sample::Tokenizer + Send + Sync> {
        self.get_tokenizer() as Arc<dyn floneumin_sample::Tokenizer + Send + Sync>
    }

    fn stream_text<'a>(&'a mut self, prompt: &'a str) -> StreamTextBuilder<'a, Self> {
        StreamTextBuilder::new(prompt, self, |self_, prompt, generation_parameters| {
            Box::pin(async move {
                let max_length = generation_parameters.max_length();
                self_
                    .run(
                        InferenceSettings::new(prompt)
                            .with_sample_len(max_length as usize)
                            .with_stop_on(generation_parameters.stop_on().map(|s| s.to_string())),
                        Arc::new(Mutex::new(generation_parameters.sampler())),
                    )
                    .map(Into::into)
            })
        })
    }

    async fn stream_text_with_sampler(
        &mut self,
        prompt: &str,
        max_tokens: Option<u32>,
        stop_on: Option<&str>,
        sampler: Arc<Mutex<dyn llm_samplers::prelude::Sampler<u32, f32>>>,
    ) -> anyhow::Result<Self::TextStream> {
        let max_length = max_tokens.unwrap_or(64);
        self.run(
            InferenceSettings::new(prompt)
                .with_sample_len(max_length as usize)
                .with_stop_on(stop_on.map(|s| s.to_string())),
            sampler,
        )
        .map(Into::into)
    }
}

pub struct PhiSpace;

impl VectorSpace for PhiSpace {}
