use crate::embedding::VectorSpace;
use crate::structured_parser::{ParseStatus, ParseStream};
use crate::{
    download::download, embedding::get_embeddings, embedding::Embedding, model::*,
    structured::StructuredSampler, structured_parser::Validate,
};
use llm::{
    InferenceError, InferenceFeedback, InferenceParameters, InferenceRequest, InferenceResponse,
    InferenceSessionConfig, Model, OutputRequest, TokenUtf8Buffer,
};
use once_cell::sync::Lazy;
use std::fmt::Debug;
use std::sync::Mutex;
use std::{
    collections::HashMap,
    convert::Infallible,
    sync::{Arc, RwLock},
};

static MODELS: Lazy<RwLock<HashMap<ModelType, Box<dyn Model>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

pub struct LocalSession<S: VectorSpace> {
    model: Box<dyn Model>,
    session: llm::InferenceSession,
    embedding_cache: RwLock<HashMap<String, Embedding<S>>>,
}

trait LocalModelType {
    fn model_type() -> ModelType;
}

macro_rules! local_model {
    ($ty: expr, $space: ident) => {
        impl LocalModelType for LocalSession<$space> {
            fn model_type() -> ModelType {
                $ty
            }
        }

        impl Default for LocalSession<$space> {
            fn default() -> Self {
                use crate::model::Model;
                Self::start()
            }
        }

        impl crate::model::Model<$space> for LocalSession<$space> {
            fn start() -> Self {
                let model = download(Self::model_type());
                let session = model.start_session(InferenceSessionConfig {
                    n_batch: 64,
                    n_threads: num_cpus::get(),
                    ..Default::default()
                });

                LocalSession {
                    model,
                    session,
                    embedding_cache: Default::default(),
                }
            }

            fn embed(input: &str) -> anyhow::Result<Embedding<$space>> {
                Self::start()._get_embedding(input)
            }

            fn generate_text(
                &mut self,
                prompt: &str,
                _generation_parameters: crate::model::GenerationParameters,
            ) -> anyhow::Result<String> {
                self._infer(prompt.to_string(), None, None)
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

impl<S: VectorSpace> Debug for LocalSession<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalSession").finish()
    }
}

impl<S: VectorSpace> LocalSession<S> {
    pub fn _infer_validate<V: for<'a> Validate<'a> + Clone + Send + Sync + 'static>(
        &mut self,
        prompt: String,
        max_tokens: Option<u32>,
        validator: V,
    ) -> anyhow::Result<String> {
        let session = &mut self.session;
        let model = &mut *self.model;

        let tokens = model.tokenizer().tokenize(&prompt, false)?;

        let sampler = Arc::new(Mutex::new(StructuredSampler::new(
            match model.tokenizer() {
                llm::Tokenizer::Embedded(embedded) => llm::Tokenizer::Embedded(embedded.clone()),
                llm::Tokenizer::HuggingFace(hugging_face) => {
                    llm::Tokenizer::HuggingFace(hugging_face.clone())
                }
            },
            validator.clone(),
            tokens.len() + 1,
        )));

        let token_ids = tokens.iter().map(|(_, id)| *id).collect::<Vec<_>>();

        let mut rng = rand::thread_rng();
        let mut result_tokens = Vec::new();
        let request = InferenceRequest {
            prompt: llm::Prompt::Tokens(&token_ids),
            parameters: &Default::default(),
            play_back_previous_tokens: false,
            maximum_token_count: max_tokens.map(|x| x as usize),
        };

        let maximum_token_count = request.maximum_token_count.unwrap_or(usize::MAX);

        let mut output = OutputRequest::default();

        // Feed the initial prompt through the transformer, to update its
        // context window with new data.
        session.feed_prompt(&*model, request.prompt, &mut output, |_| {
            Result::<_, std::convert::Infallible>::Ok(InferenceFeedback::Continue)
        })?;

        // After the prompt is consumed, sample tokens by repeatedly calling
        // `infer_next_token`. We generate tokens until the model returns an
        // EndOfText token, or we run out of space in the context window,
        // or we reach the specified limit.
        let mut tokens_processed = 0;
        let mut token_utf8_buf = TokenUtf8Buffer::new();
        while tokens_processed < maximum_token_count {
            let parameters = &InferenceParameters {
                sampler: sampler.clone(),
            };

            let token = match session.infer_next_token(&*model, parameters, &mut output, &mut rng) {
                Ok(token) => token,
                Err(InferenceError::EndOfText) => break,
                Err(e) => panic!("Error: {:?}", e),
            };

            // Buffer the token until it's valid UTF-8, then call the callback.
            if let Some(token) = token_utf8_buf.push(&token) {
                result_tokens.push(token);
            }

            tokens_processed += 1;

            loop {
                let borrowed = result_tokens.iter().map(|x| x.as_str()).collect::<Vec<_>>();
                let status = sampler
                    .lock()
                    .unwrap()
                    .structure
                    .validate(ParseStream::new(&borrowed));
                match status {
                    ParseStatus::Incomplete {
                        required_next: Some(required_next),
                    } => {
                        // feed the required next text
                        let tokens = model.tokenizer().tokenize(&required_next, false)?;
                        let token_ids = tokens.iter().map(|(_, id)| *id).collect::<Vec<_>>();
                        session.feed_prompt(
                            &*model,
                            llm::Prompt::Tokens(&token_ids),
                            &mut output,
                            |_| {
                                Result::<_, std::convert::Infallible>::Ok(
                                    InferenceFeedback::Continue,
                                )
                            },
                        )?;
                        let token = tokens
                            .iter()
                            .flat_map(|(x, _)| x.iter().copied())
                            .collect::<Vec<u8>>();
                        if let Some(token) = token_utf8_buf.push(&token) {
                            result_tokens.push(token);
                        }

                        tokens_processed += token_ids.len();
                    }
                    ParseStatus::Complete(..) => return Ok(result_tokens.join("")),
                    _ => {
                        break;
                    }
                }
            }
        }

        Ok(result_tokens.join(""))
    }

    #[tracing::instrument]
    pub fn _infer(
        &mut self,
        prompt: String,
        max_tokens: Option<u32>,
        stop_on: Option<String>,
    ) -> anyhow::Result<String> {
        let session = &mut self.session;
        let model = &mut *self.model;

        let parmeters = Default::default();

        let mut rng = rand::thread_rng();
        let mut buf = String::new();
        let request = InferenceRequest {
            prompt: (&prompt).into(),
            parameters: &parmeters,
            play_back_previous_tokens: false,
            maximum_token_count: max_tokens.map(|x| x as usize),
        };

        if let Err(err) = session.infer(
            model,
            &mut rng,
            &request,
            &mut Default::default(),
            inference_callback(stop_on, &mut buf),
        ) {
            log::error!("{err}")
        }

        Ok(buf)
    }

    #[tracing::instrument]
    fn _get_embedding(&self, text: &str) -> anyhow::Result<Embedding<S>> {
        let mut write = self.embedding_cache.write().unwrap();
        let cache = &mut *write;
        Ok(if let Some(embedding) = cache.get(text) {
            embedding.clone()
        } else {
            let model = self.model.as_ref();
            let new_embedding = get_embeddings(model, text);
            cache.insert(text.to_string(), new_embedding.clone());
            new_embedding
        })
    }
}

/// buf is used here...
#[allow(clippy::needless_pass_by_ref_mut)]
fn inference_callback(
    stop_sequence: Option<String>,
    buf: &mut String,
) -> impl FnMut(InferenceResponse) -> Result<InferenceFeedback, Infallible> + '_ {
    move |resp| match resp {
        InferenceResponse::InferredToken(t) => {
            let mut reverse_buf = buf.clone();
            reverse_buf.push_str(t.as_str());
            if let Some(stop_sequence) = &stop_sequence {
                if stop_sequence.as_str().eq(reverse_buf.as_str()) {
                    return Ok(InferenceFeedback::Halt);
                }
            }
            buf.push_str(t.as_str());

            Ok(InferenceFeedback::Continue)
        }
        InferenceResponse::EotToken => Ok(InferenceFeedback::Halt),
        _ => Ok(InferenceFeedback::Continue),
    }
}
