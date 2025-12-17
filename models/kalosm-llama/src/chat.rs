use std::{
    future::Future,
    sync::{Arc, RwLock},
};

use crate::{model::LlamaModelError, session::LlamaSessionLoadingError, Llama, LlamaSession};
use kalosm_language_model::{
    ChatMessage, ChatModel, ChatSession, ContentChunk, CreateChatSession,
    CreateTextCompletionSession, MessageContent, MessageType, StructuredChatModel,
    StructuredTextCompletionModel, TextCompletionModel,
};
use kalosm_sample::{CreateParserState, Parser};
use llm_samplers::types::Sampler;
use minijinja::ErrorKind;

fn get_new_tokens(
    messages: &[ChatMessage],
    session: &mut LlamaChatSession,
    model: &Llama,
) -> Result<String, LlamaModelError> {
    let chat_template = model
        .config
        .chat_template
        .as_ref()
        .ok_or(LlamaModelError::NoChatTemplate)?;
    let bos_token = &model.config.start_token_string;
    let eos_token = &model.config.stop_token_string;
    let current_text = if session.history.is_empty() {
        String::new()
    } else {
        let old_formatted_text =
            chat_template.format(bos_token, eos_token, &session.history, true)?;
        // Some chat templates (like llama v3) always include the generation prompt even when we tell them not to. If they do, try to strip it off
        let (before_last_eos, _) = old_formatted_text
            .rsplit_once(eos_token)
            .unwrap_or((&old_formatted_text, ""));
        before_last_eos.to_string() + eos_token
    };
    session.history.extend_from_slice(messages);
    let updated_text = chat_template.format(bos_token, eos_token, &session.history, true)?;
    let new_text = updated_text.strip_prefix(&current_text).ok_or_else(|| {
        LlamaModelError::ChatTemplateError(minijinja::Error::new(
            ErrorKind::InvalidOperation,
            format!("Chat template should only add text to the end of the current text. Old text: {current_text}, new text: {updated_text}"),
        ))
    })?;

    Ok(new_text.to_string())
}

impl CreateChatSession for Llama {
    type Error = LlamaModelError;
    type ChatSession = LlamaChatSession;

    fn new_chat_session(&self) -> Result<Self::ChatSession, Self::Error> {
        Ok(LlamaChatSession::new(self.new_session()?))
    }
}

impl<S: Sampler + 'static> ChatModel<S> for Llama {
    fn add_messages_with_callback<'a>(
        &'a self,
        session: &'a mut Self::ChatSession,
        messages: &[ChatMessage],
        sampler: S,
        mut on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'a {
        let new_text = get_new_tokens(messages, session, self);
        let mut content = MessageContent::new();
        for message in messages {
            for chunk in message.content().chunks() {
                if matches!(chunk, ContentChunk::Media(_)) {
                    content.push(chunk.clone());
                }
            }
        }
        async move {
            let new_text = new_text?;
            let model_response = Arc::new(RwLock::new(String::new()));
            let on_token = {
                let model_response = model_response.clone();
                move |token: String| {
                    let mut model_response = model_response.write().unwrap();
                    *model_response += &token;
                    on_token(token)
                }
            };
            content.push(new_text);

            self.stream_text_with_callback(&mut session.session, content, sampler, on_token)
                .await?;
            session.history.push(ChatMessage::new(
                MessageType::ModelAnswer,
                model_response.read().unwrap().clone(),
            ));
            Ok(())
        }
    }
}

impl<S, Constraints> StructuredChatModel<Constraints, S> for Llama
where
    <Constraints as Parser>::Output: Send,
    Constraints: CreateParserState + Send + 'static,
    S: Sampler + 'static,
{
    fn add_message_with_callback_and_constraints<'a>(
        &'a self,
        session: &'a mut Self::ChatSession,
        messages: &[ChatMessage],
        sampler: S,
        constraints: Constraints,
        mut on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<
        Output = Result<
            <Constraints as kalosm_language_model::ModelConstraints>::Output,
            Self::Error,
        >,
    > + Send
           + 'a {
        let mut content = MessageContent::new();
        for message in messages {
            for chunk in message.content().chunks() {
                if matches!(chunk, ContentChunk::Media(_)) {
                    content.push(chunk.clone());
                }
            }
        }
        let new_text = get_new_tokens(messages, session, self);
        async move {
            let new_text = new_text?;
            let model_response = Arc::new(RwLock::new(String::new()));
            let on_token = {
                let model_response = model_response.clone();
                move |token: String| {
                    let mut model_response = model_response.write().unwrap();
                    *model_response += &token;
                    on_token(token)
                }
            };
            content.push(new_text);
            let result = self
                .stream_text_with_callback_and_parser(
                    &mut session.session,
                    content,
                    sampler,
                    constraints,
                    on_token,
                )
                .await?;
            session.history.push(ChatMessage::new(
                MessageType::ModelAnswer,
                model_response.read().unwrap().clone(),
            ));
            Ok(result)
        }
    }
}

/// A Llama chat session.
#[derive(Clone)]
pub struct LlamaChatSession {
    history: Vec<ChatMessage>,
    session: LlamaSession<half::f16>,
}

impl ChatSession for LlamaChatSession {
    type Error = LlamaSessionLoadingError;

    fn history(&self) -> Vec<ChatMessage> {
        self.history.clone()
    }

    fn try_clone(&self) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized,
    {
        Ok(self.clone())
    }
}

impl LlamaChatSession {
    #[allow(clippy::too_many_arguments)]
    /// Creates a new chat history.
    fn new(session: LlamaSession<half::f16>) -> Self {
        Self {
            history: Vec::new(),
            session,
        }
    }
}
