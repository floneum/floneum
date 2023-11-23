use std::{
    fmt::Display,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use kalosm_language::{ChatModel, ModelExt, SyncModel, SyncModelExt};
use kalosm_streams::ChannelTextStream;
use llm_samplers::types::Sampler;
use tokio::sync::mpsc::unbounded_channel;

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
    assistant_marker: String,
    history: Vec<ChatHistoryItem>,
    session: Session,
    unfed_text: String,
    eos: String,
    sampler: Arc<Mutex<dyn Sampler + Send + Sync>>,
}

impl<Session> ChatSession<Session> {
    /// Creates a new chat history.
    pub fn new<Model: SyncModel<Session = Session>>(
        model: &Model,
        system_prompt_marker: String,
        user_marker: String,
        assistant_marker: String,
        system_prompt: String,
        sampler: Arc<Mutex<dyn Sampler + Send + Sync>>,
    ) -> Self {
        let unfed_text = system_prompt_marker + &system_prompt;
        let history = vec![ChatHistoryItem {
            ty: ChatState::SystemPrompt,
            contents: system_prompt,
        }];

        Self {
            user_marker,
            assistant_marker,
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
        message: impl Display,
        model: &mut Model,
        stream: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<()> {
        let message = message.to_string();
        let new_text = format!("{}{}{}", self.user_marker, message, self.eos);
        self.history.push(ChatHistoryItem {
            ty: ChatState::UserMessage,
            contents: message,
        });
        self.unfed_text += &new_text;
        let mut bot_response = String::new();
        let on_token = |tok: String| {
            bot_response += &tok;
            stream.send(tok)?;
            Ok(kalosm_language::ModelFeedback::Continue)
        };
        self.unfed_text += &self.assistant_marker;
        model.stream_text_with_sampler(
            &mut self.session,
            &std::mem::take(&mut self.unfed_text),
            None,
            Some(&self.eos),
            self.sampler.clone(),
            on_token,
        )?;
        self.history.push(ChatHistoryItem {
            ty: ChatState::ModelAnswer,
            contents: bot_response,
        });
        Ok(())
    }
}

/// A chat session.
pub struct Chat {
    sender: tokio::sync::mpsc::UnboundedSender<String>,
    channel: tokio::sync::mpsc::UnboundedReceiver<tokio::sync::mpsc::UnboundedReceiver<String>>,
}

impl Chat {
    /// Creates a new chat session.
    pub async fn new<Model: ChatModel>(
        model: &mut Model,
        system_prompt: impl Into<String>,
        sampler: impl Sampler + Send + Sync + 'static,
    ) -> Self {
        let system_prompt_marker = model.system_prompt_marker().to_string();
        let user_marker = model.user_marker().to_string();
        let assistant_marker = model.assistant_marker().to_string();
        let system_prompt = system_prompt.into();
        let (sender_tx, mut sender_rx) = unbounded_channel();
        let (result_tx, result_rx) = unbounded_channel();
        model
            .run_sync(move |model| {
                Box::pin(async move {
                    let mut session = ChatSession::new(
                        model,
                        system_prompt_marker,
                        user_marker,
                        assistant_marker,
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
        }
    }

    /// Adds a message to the history.
    pub async fn add_message(
        &mut self,
        message: impl Into<String>,
    ) -> Result<ChannelTextStream<String>> {
        let message = message.into();
        let message = message.trim();
        self.sender
            .send(message.to_string())
            .map_err(|_| anyhow::anyhow!("Model stopped"))?;
        self.channel
            .recv()
            .await
            .map(|c| c.into())
            .ok_or(anyhow::anyhow!("Model stopped"))
    }
}
