use crate::structured_parser::{ParseStatus, ParseStream};
use crate::{
    download::download, embedding::get_embeddings, exports::plugins::main::definitions::Embedding,
    plugins::main::types::ModelId, structured::StructuredSampler, structured_parser::Validate,
    ModelType,
};
use llm::{
    InferenceError, InferenceFeedback, InferenceParameters, InferenceRequest, InferenceResponse,
    InferenceSession, InferenceSessionConfig, Model, OutputRequest, TokenUtf8Buffer,
};
use slab::Slab;
use std::fmt::Debug;
use std::{
    collections::HashMap,
    convert::Infallible,
    sync::{Arc, RwLock},
};

#[derive(Default)]
pub struct InferenceSessions {
    sessions: Slab<(Box<dyn Model>, llm::InferenceSession, ModelType)>,
    // We keep the last session running if there are no other active sessions.
    temp_retained_sessions: Option<ModelId>,
    embedding_cache: RwLock<Vec<HashMap<String, Embedding>>>,
}

impl Debug for InferenceSessions {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceSessions").finish()
    }
}

impl InferenceSessions {
    pub fn session_get(&self, id: ModelId) -> &(Box<dyn Model>, InferenceSession, ModelType) {
        self.sessions.get(id.id as usize).unwrap()
    }

    pub fn session_get_mut(
        &mut self,
        id: ModelId,
    ) -> &mut (Box<dyn Model>, InferenceSession, ModelType) {
        self.sessions.get_mut(id.id as usize).unwrap()
    }

    #[tracing::instrument]
    pub fn create(&mut self, ty: ModelType) -> ModelId {
        // Remove the temporary session if it exists.
        if let Some(id) = self.temp_retained_sessions.take() {
            let (_model, _session, _old_ty) = self.sessions.remove(id.id as usize);
            // unwind the model
            // todo: This currently causes an error in llm...
            // let token_count = session.tokens().len();
            // session.rewind(&*model, token_count).unwrap();

            // // If the model type is the same, we can reuse the session.
            // if cmp_model_types(old_ty, ty) {
            //     return ModelId {
            //         id: self.sessions.insert((model, session, ty)) as u32,
            //     };
            // }
        }
        let model = download(ty);
        let session = model.start_session(InferenceSessionConfig {
            use_gpu: true,
            ..Default::default()
        });
        ModelId {
            id: self.sessions.insert((model, session, ty)) as u32,
        }
    }

    #[tracing::instrument]
    pub fn remove(&mut self, id: ModelId) {
        if self.sessions.len() == 1 {
            self.temp_retained_sessions = Some(id);
        } else {
            self.sessions.remove(id.id as usize);
        }
    }

    pub fn infer_validate<V: for<'a> Validate<'a> + Clone + Send + Sync + 'static>(
        &mut self,
        id: ModelId,
        prompt: String,
        max_tokens: Option<u32>,
        validator: V,
    ) -> String {
        let (model, session, _) = self.session_get_mut(id);

        let tokens = model.tokenizer().tokenize(&prompt, false).unwrap();

        let sampler = Arc::new(StructuredSampler::new(
            match model.tokenizer() {
                llm::Tokenizer::Embedded(embedded) => llm::Tokenizer::Embedded(embedded.clone()),
                llm::Tokenizer::HuggingFace(hugging_face) => {
                    llm::Tokenizer::HuggingFace(hugging_face.clone())
                }
            },
            validator.clone(),
            tokens.len() + 1,
        ));

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

        let parameters = request.parameters;
        let mut output = OutputRequest::default();

        // Feed the initial prompt through the transformer, to update its
        // context window with new data.
        session
            .feed_prompt(&**model, parameters, request.prompt, &mut output, |_| {
                Result::<_, std::convert::Infallible>::Ok(InferenceFeedback::Continue)
            })
            .unwrap();

        // After the prompt is consumed, sample tokens by repeatedly calling
        // `infer_next_token`. We generate tokens until the model returns an
        // EndOfText token, or we run out of space in the context window,
        // or we reach the specified limit.
        let mut tokens_processed = 0;
        let mut token_utf8_buf = TokenUtf8Buffer::new();
        while tokens_processed < maximum_token_count {
            let parameters = &InferenceParameters {
                sampler: sampler.clone(),
                n_batch: 16,
                n_threads: 12,
            };

            let token = match session.infer_next_token(&**model, parameters, &mut output, &mut rng)
            {
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
                let status = sampler.structure.validate(ParseStream::new(&borrowed));
                match status {
                    ParseStatus::Incomplete {
                        required_next: Some(required_next),
                    } => {
                        // feed the required next text
                        let tokens = model.tokenizer().tokenize(&required_next, false).unwrap();
                        let token_ids = tokens.iter().map(|(_, id)| *id).collect::<Vec<_>>();
                        session
                            .feed_prompt(
                                &**model,
                                parameters,
                                llm::Prompt::Tokens(&token_ids),
                                &mut output,
                                |_| {
                                    Result::<_, std::convert::Infallible>::Ok(
                                        InferenceFeedback::Continue,
                                    )
                                },
                            )
                            .unwrap();
                        let token = tokens
                            .iter()
                            .map(|(x, _)| x.iter().copied())
                            .flatten()
                            .collect::<Vec<u8>>();
                        if let Some(token) = token_utf8_buf.push(&token) {
                            result_tokens.push(token);
                        }

                        tokens_processed += token_ids.len();
                    }
                    ParseStatus::Complete(..) => return result_tokens.join(""),
                    _ => {
                        break;
                    }
                }
            }
        }

        result_tokens.join("")
    }

    #[tracing::instrument]
    pub fn infer(
        &mut self,
        id: ModelId,
        prompt: String,
        max_tokens: Option<u32>,
        stop_on: Option<String>,
    ) -> String {
        let (model, session, _) = self.session_get_mut(id);

        let parmeters = Default::default();

        let mut rng = rand::thread_rng();
        let mut buf = String::new();
        let request = InferenceRequest {
            prompt: (&prompt).into(),
            parameters: &parmeters,
            play_back_previous_tokens: false,
            maximum_token_count: max_tokens.map(|x| x as usize),
        };

        session
            .infer(
                model.as_ref(),
                &mut rng,
                &request,
                &mut Default::default(),
                inference_callback(stop_on, &mut buf),
            )
            .unwrap_or_else(|e| panic!("{e}"));

        buf
    }

    #[tracing::instrument]
    pub fn get_embedding(&self, id: ModelId, text: &str) -> Embedding {
        let mut write = self.embedding_cache.write().unwrap();
        let cache = if let Some(cache) = write.get_mut(id.id as usize) {
            cache
        } else {
            if id.id as usize >= write.len() {
                write.resize_with(id.id as usize + 1, Default::default);
            }
            &mut write[id.id as usize]
        };
        if let Some(embedding) = cache.get(text) {
            embedding.clone()
        } else {
            let (model, _, _) = self.session_get(id);
            let inference_parameters = llm::InferenceParameters::default();
            let new_embedding = get_embeddings(model.as_ref(), &inference_parameters, text);
            cache.insert(text.to_string(), new_embedding.clone());
            new_embedding
        }
    }
}

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

#[allow(unused)]
fn cmp_model_types(first: ModelType, second: ModelType) -> bool {
    match (first, second) {
        (
            crate::plugins::main::types::ModelType::Mpt(mpt1),
            crate::plugins::main::types::ModelType::Mpt(mpt2),
        ) => mpt1 == mpt2,
        (
            crate::plugins::main::types::ModelType::GptNeoX(neo1),
            crate::plugins::main::types::ModelType::GptNeoX(neo2),
        ) => neo1 == neo2,
        (
            crate::plugins::main::types::ModelType::Llama(llama1),
            crate::plugins::main::types::ModelType::Llama(llama2),
        ) => llama1 == llama2,
        (_, _) => false,
    }
}
