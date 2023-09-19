use crate::embedding::VectorSpace;
use crate::structured_parser::{ParseStatus, ParseStream};
use crate::{
    embedding::get_embeddings, embedding::Embedding, structured::StructuredSampler,
    structured_parser::Validate,
};
use futures_util::Stream;

use llm::{
    InferenceError, InferenceFeedback, InferenceParameters, InferenceRequest, InferenceResponse,
    Model, OutputRequest, TokenUtf8Buffer,
};
use std::fmt::Debug;
use std::sync::Mutex;
use std::{
    collections::HashMap,
    convert::Infallible,
    sync::{Arc, RwLock},
};

pub struct LocalSession<S: VectorSpace> {
    task_sender: tokio::sync::mpsc::UnboundedSender<Task<S>>,
    thread_handle: Option<std::thread::JoinHandle<()>>,
}

impl<S: VectorSpace> Drop for LocalSession<S> {
    fn drop(&mut self) {
        self.task_sender.send(Task::Kill).unwrap();
        self.thread_handle.take().unwrap().join().unwrap();
    }
}

impl<S: VectorSpace> Debug for LocalSession<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalSession").finish()
    }
}

impl<S: VectorSpace + Send + Sync + 'static> LocalSession<S> {
    pub fn new(model: Box<dyn Model>, session: llm::InferenceSession) -> Self {
        let (task_sender, mut task_receiver) = tokio::sync::mpsc::unbounded_channel();

        let thread_handle = std::thread::spawn(move || {
            let mut inner = LocalSessionInner {
                model,
                session,
                embedding_cache: RwLock::new(HashMap::new()),
            };
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async move {
                    while let Some(task) = task_receiver.recv().await {
                        match task {
                            Task::Kill => break,
                            Task::Infer {
                                prompt,
                                max_tokens,
                                stop_on,
                                sender,
                            } => {
                                inner._infer(prompt, max_tokens, stop_on, sender);
                            }
                            Task::InferValidate {
                                prompt,
                                max_tokens,
                                validator,
                                sender,
                            } => {
                                let result = inner
                                    ._infer_validate(prompt, max_tokens, validator)
                                    .unwrap();
                                sender.send(Ok(result)).unwrap();
                            }
                            Task::GetEmbedding { text, sender } => {
                                let result = inner._get_embedding(&text).unwrap();
                                sender.send(Ok(result)).unwrap();
                            }
                        }
                    }
                })
        });
        Self {
            task_sender,
            thread_handle: Some(thread_handle),
        }
    }

    pub(crate) async fn infer_validate<V: for<'a> Validate<'a> + Clone + Send + Sync + 'static>(
        &mut self,
        prompt: String,
        max_tokens: Option<u32>,
        validator: V,
    ) -> anyhow::Result<String> {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        self.task_sender
            .send(Task::InferValidate {
                prompt,
                max_tokens,
                validator: ArcValidate(Arc::new(validator)),
                sender,
            })
            .unwrap();
        receiver
            .await
            .unwrap()
            .map_err(|_| anyhow::anyhow!("Failed to receive result"))
    }

    pub(crate) async fn infer(
        &mut self,
        prompt: String,
        max_tokens: Option<u32>,
        stop_on: Option<String>,
    ) -> LLMStream {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        self.task_sender
            .send(Task::Infer {
                prompt,
                max_tokens,
                stop_on,
                sender,
            })
            .unwrap();
        receiver.await.unwrap()
    }

    pub(crate) async fn get_embedding(&self, text: &str) -> anyhow::Result<Embedding<S>> {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        self.task_sender
            .send(Task::GetEmbedding {
                text: text.to_string(),
                sender,
            })
            .unwrap();
        receiver
            .await
            .unwrap()
            .map_err(|_| anyhow::anyhow!("Failed to receive result"))
    }
}

#[derive(Clone)]
pub struct ArcValidate(pub(crate) Arc<dyn for<'a> Validate<'a> + Send + Sync + 'static>);

impl<'a> Validate<'a> for ArcValidate {
    fn validate(&self, tokens: ParseStream<'a>) -> ParseStatus<'a> {
        self.0.validate(tokens)
    }
}

enum Task<S: VectorSpace> {
    Kill,
    Infer {
        prompt: String,
        max_tokens: Option<u32>,
        stop_on: Option<String>,
        sender: tokio::sync::oneshot::Sender<LLMStream>,
    },
    InferValidate {
        prompt: String,
        max_tokens: Option<u32>,
        validator: ArcValidate,
        sender: tokio::sync::oneshot::Sender<anyhow::Result<String>>,
    },
    GetEmbedding {
        text: String,
        sender: tokio::sync::oneshot::Sender<anyhow::Result<Embedding<S>>>,
    },
}

struct LocalSessionInner<S: VectorSpace> {
    model: Box<dyn Model>,
    session: llm::InferenceSession,
    embedding_cache: RwLock<HashMap<String, Embedding<S>>>,
}

impl<S: VectorSpace> Debug for LocalSessionInner<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalSessionInner").finish()
    }
}

impl<S: VectorSpace> LocalSessionInner<S> {
    fn _infer_validate<V: for<'a> Validate<'a> + Clone + Send + Sync + 'static>(
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

    #[tracing::instrument(skip(out))]
    fn _infer(
        &mut self,
        prompt: String,
        max_tokens: Option<u32>,
        stop_on: Option<String>,
        out: tokio::sync::oneshot::Sender<LLMStream>,
    ) {
        let session = &mut self.session;
        let model = &mut *self.model;

        let parameters = Default::default();

        let (callback, stream) = inference_callback();
        if let Err(_) = out.send(stream) {
            log::error!("Failed to send stream");
            return;
        }

        let mut rng = rand::thread_rng();

        let request = InferenceRequest {
            prompt: (&prompt).into(),
            parameters: &parameters,
            play_back_previous_tokens: false,
            maximum_token_count: max_tokens.map(|x| x as usize),
        };

        if let Err(err) =
            session.infer(model, &mut rng, &request, &mut Default::default(), callback)
        {
            log::error!("{err}")
        }
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

#[allow(clippy::needless_pass_by_ref_mut)]
fn inference_callback() -> (
    impl FnMut(InferenceResponse) -> Result<InferenceFeedback, Infallible>,
    LLMStream,
) {
    let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
    let stream = LLMStream { receiver };
    let callback = move |resp| match resp {
        InferenceResponse::InferredToken(t) => {
            sender.send(t).unwrap();

            Ok(InferenceFeedback::Continue)
        }
        InferenceResponse::EotToken => Ok(InferenceFeedback::Halt),
        _ => Ok(InferenceFeedback::Continue),
    };
    (callback, stream)
}

pub struct LLMStream {
    receiver: tokio::sync::mpsc::UnboundedReceiver<String>,
}

impl Stream for LLMStream {
    type Item = String;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> core::task::Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}
