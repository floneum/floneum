use std::{
    fmt::Display,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use kalosm_language::{ChatModel, ModelExt, SyncModel, SyncModelExt};
use kalosm_streams::ChannelTextStream;
use llm_samplers::types::Sampler;
use tokio::sync::mpsc::unbounded_channel;

/// A simple helper function for prompting the user for input.
pub fn prompt_input(prompt: impl Display) -> Result<String> {
    use std::io::Write;
    print!("{}", prompt);
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(input)
}

enum ChatState {
    SystemPrompt,
    UserMessage,
    ModelAnswer,
}

#[allow(unused)]
struct ChatHistoryItem {
    ty: ChatState,
    contents: String,
}

/// The history of a chat session.
struct ChatSession<Session> {
    user_marker: String,
    end_user_marker: String,
    assistant_marker: String,
    end_assistant_marker: String,
    history: Vec<ChatHistoryItem>,
    session: Session,
    unfed_text: String,
    eos: String,
    sampler: Arc<Mutex<dyn Sampler + Send + Sync>>,
}

impl<Session> ChatSession<Session> {
    #[allow(clippy::too_many_arguments)]
    /// Creates a new chat history.
    pub(crate) fn new<Model: SyncModel<Session = Session>>(
        model: &Model,
        system_prompt_marker: String,
        end_system_prompt_marker: String,
        user_marker: String,
        end_user_marker: String,
        assistant_marker: String,
        end_assistant_marker: String,
        system_prompt: String,
        sampler: Arc<Mutex<dyn Sampler + Send + Sync>>,
    ) -> Self {
        let unfed_text = system_prompt_marker + &system_prompt + &end_system_prompt_marker;
        let history = vec![ChatHistoryItem {
            ty: ChatState::SystemPrompt,
            contents: system_prompt,
        }];

        Self {
            user_marker,
            end_user_marker,
            assistant_marker,
            end_assistant_marker,
            eos: model
                .tokenizer()
                .decode(&[model.stop_token().unwrap()])
                .unwrap()
                .to_string(),
            session: model.new_session().unwrap(),
            unfed_text,
            history,
            sampler,
        }
    }

    /// Adds a message to the history.
    pub fn add_message<Model: SyncModel<Session = Session>>(
        &mut self,
        message: AddMessage<Model>,
        model: &mut Model,
        stream: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<()> {
        let AddMessage { message, filter } = message;
        let new_text = format!("{}{}{}", self.user_marker, message, self.end_user_marker);
        self.history.push(ChatHistoryItem {
            ty: ChatState::UserMessage,
            contents: message,
        });
        self.unfed_text += &new_text;
        let mut bot_response = String::new();
        self.unfed_text += &self.assistant_marker;
        let prompt = std::mem::take(&mut self.unfed_text);
        match filter {
            Some(filter) => {
                let mut filter = filter.lock().unwrap();
                loop {
                    bot_response.clear();
                    let on_token = |tok: String| {
                        bot_response += &tok;
                        Ok(kalosm_language::ModelFeedback::Continue)
                    };

                    model.stream_text_with_sampler(
                        &mut self.session,
                        &prompt,
                        None,
                        Some(&self.eos),
                        self.sampler.clone(),
                        on_token,
                    )?;
                    if filter(&bot_response, model) {
                        stream.send(bot_response.clone())?;
                        break;
                    } else {
                        tracing::trace!("Filtered out: {}", bot_response);
                    }
                }
            }
            None => {
                let on_token = |tok: String| {
                    bot_response += &tok;
                    stream.send(tok)?;
                    Ok(kalosm_language::ModelFeedback::Continue)
                };

                model.stream_text_with_sampler(
                    &mut self.session,
                    &prompt,
                    None,
                    Some(&self.eos),
                    self.sampler.clone(),
                    on_token,
                )?;
            }
        }

        self.unfed_text += &self.end_assistant_marker;
        self.history.push(ChatHistoryItem {
            ty: ChatState::ModelAnswer,
            contents: bot_response,
        });
        Ok(())
    }
}

type MessageFilter<M> = Option<Arc<Mutex<Box<dyn FnMut(&str, &mut M) -> bool + Send + Sync>>>>;

struct AddMessage<M> {
    message: String,
    filter: MessageFilter<M>,
}

/// A chat session.
pub struct Chat<M: ChatModel> {
    sender: tokio::sync::mpsc::UnboundedSender<AddMessage<M::SyncModel>>,
    channel: tokio::sync::mpsc::UnboundedReceiver<tokio::sync::mpsc::UnboundedReceiver<String>>,
    filter: MessageFilter<M::SyncModel>,
}

impl<M: ChatModel> Chat<M> {
    /// Creates a new chat session.
    pub async fn new(
        model: &mut M,
        system_prompt: impl Into<String>,
        sampler: impl Sampler + Send + Sync + 'static,
    ) -> Self {
        let system_prompt_marker = model.system_prompt_marker().to_string();
        let end_system_prompt_marker = model.end_system_prompt_marker().to_string();
        let user_marker = model.user_marker().to_string();
        let end_user_marker = model.end_user_marker().to_string();
        let assistant_marker = model.assistant_marker().to_string();
        let end_assistant_marker = model.end_assistant_marker().to_string();
        let system_prompt = system_prompt.into();
        let (sender_tx, mut sender_rx) = unbounded_channel();
        let (result_tx, result_rx) = unbounded_channel();
        model
            .run_sync(move |model| {
                Box::pin(async move {
                    let mut session = ChatSession::new(
                        model,
                        system_prompt_marker,
                        end_system_prompt_marker,
                        user_marker,
                        end_user_marker,
                        assistant_marker,
                        end_assistant_marker,
                        system_prompt,
                        Arc::new(Mutex::new(sampler)),
                    );

                    while let Some(message) = sender_rx.recv().await {
                        let (tx, rx) = unbounded_channel();
                        result_tx.send(rx).unwrap();
                        session.add_message(message, model, tx).unwrap();
                    }
                })
            })
            .await
            .unwrap();
        Self {
            sender: sender_tx,
            channel: result_rx,
            filter: None,
        }
    }

    /// Adds a message to the history.
    pub async fn add_message(
        &mut self,
        message: impl Into<String>,
    ) -> Result<ChannelTextStream<String>> {
        let message = message.into();
        let message = message.trim().to_string();
        self.sender
            .send(AddMessage {
                message,
                filter: self.filter.clone(),
            })
            .map_err(|_| anyhow::anyhow!("Model stopped"))?;
        self.channel
            .recv()
            .await
            .map(|c| c.into())
            .ok_or(anyhow::anyhow!("Model stopped"))
    }

    /// Filter all messages
    pub fn filter(
        self,
        filter: impl FnMut(&str, &mut M::SyncModel) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            filter: Some(Arc::new(Mutex::new(
                Box::new(filter) as Box<dyn FnMut(&str, &mut M::SyncModel) -> bool + Send + Sync>
            ))),
            ..self
        }
    }
}
