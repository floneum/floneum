use std::{
    future::Future,
    sync::{Arc, RwLock},
};

use crate::{model::LlamaModelError, session::LlamaSessionLoadingError, Llama, LlamaSession};
use kalosm_common::accelerated_device_if_available;
use kalosm_language_model::{
    ChatMessage, ChatModel, ChatSession, ContentChunk, CreateChatSession,
    CreateTextCompletionSession, MessageContent, MessageType, StructuredChatModel,
    StructuredTextCompletionModel, TextCompletionModel,
};
use kalosm_sample::{CreateParserState, Parser};
use llm_samplers::types::Sampler;
use minijinja::ErrorKind;

#[cfg(test)]
use pretty_assertions::assert_eq;

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
    session: LlamaSession,
}

impl ChatSession for LlamaChatSession {
    type Error = LlamaSessionLoadingError;

    fn write_to(&self, into: &mut Vec<u8>) -> Result<(), Self::Error> {
        let device = accelerated_device_if_available()?;

        let history_items = self.history.len() as u32;
        let mut all_bytes = Vec::new();
        all_bytes.extend_from_slice(&history_items.to_le_bytes());
        for item in &self.history {
            let ty = match item.role() {
                MessageType::UserMessage => 0u8,
                MessageType::ModelAnswer => 1,
                MessageType::SystemPrompt => 2,
            };
            all_bytes.extend_from_slice(&ty.to_le_bytes());
            let text_content = item
                .content()
                .chunks()
                .iter()
                .fold(String::new(), |acc, chunk| {
                    let content = match chunk {
                        ContentChunk::Text(text) => text,
                        ContentChunk::Media(_) => return acc,
                    };
                    acc + content
                });
            let content_bytes = text_content.as_bytes();
            let content_bytes_len = content_bytes.len() as u32;
            all_bytes.extend_from_slice(&content_bytes_len.to_le_bytes());
            all_bytes.extend_from_slice(content_bytes);
        }

        let tensors = self.session.get_tensor_map(&device);
        let bytes = safetensors::serialize(&tensors, &None)?;
        all_bytes.extend_from_slice(&bytes);

        into.extend_from_slice(&all_bytes);

        Ok(())
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized,
    {
        let mut history_items = Vec::new();
        let mut cursor_pos = 0;
        let history_item_count = u32::from_le_bytes(
            bytes
                .get(..4)
                .ok_or(LlamaSessionLoadingError::InvalidChatMessages)?
                .try_into()
                .map_err(|_| LlamaSessionLoadingError::InvalidChatMessages)?,
        );
        cursor_pos += 4;
        history_items.reserve(history_item_count as usize);
        for _ in 0..history_item_count {
            let ty = bytes[cursor_pos];
            let ty = match ty {
                0 => MessageType::UserMessage,
                1 => MessageType::ModelAnswer,
                2 => MessageType::SystemPrompt,
                _ => return Err(LlamaSessionLoadingError::InvalidChatMessages),
            };
            cursor_pos += 1;
            let content_bytes_len = u32::from_le_bytes(
                bytes[cursor_pos..cursor_pos + 4]
                    .try_into()
                    .map_err(|_| LlamaSessionLoadingError::InvalidChatMessages)?,
            );
            cursor_pos += 4;
            let content_bytes = &bytes[cursor_pos..cursor_pos + content_bytes_len as usize];
            cursor_pos += content_bytes_len as usize;
            let item = ChatMessage::new(
                ty,
                String::from_utf8(content_bytes.to_vec())
                    .map_err(|_| LlamaSessionLoadingError::InvalidChatMessages)?,
            );
            history_items.push(item);
        }

        let device = accelerated_device_if_available()?;
        let tensors = candle_core::safetensors::load_buffer(&bytes[cursor_pos..], &device)?;

        let session = LlamaSession::from_tensor_map(tensors)?;

        Ok(Self {
            history: history_items,
            session,
        })
    }

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

#[test]
fn test_serialize_deserialize_chat_session() {
    use crate::raw::LlamaConfig;

    let config = LlamaConfig::mock_test();
    let session = LlamaChatSession {
        history: vec![
            ChatMessage::new(MessageType::UserMessage, "Hello, world!".to_string()),
            ChatMessage::new(
                MessageType::ModelAnswer,
                "I'm doing great. How can I help you today?".to_string(),
            ),
            ChatMessage::new(
                MessageType::SystemPrompt,
                "The assistant will act like a pirate.".to_string(),
            ),
        ],
        session: LlamaSession::new(&config),
    };

    let bytes = session.to_bytes().unwrap();
    let session = LlamaChatSession::from_bytes(&bytes).unwrap();

    assert_eq!(session.history, session.history);
}

impl LlamaChatSession {
    #[allow(clippy::too_many_arguments)]
    /// Creates a new chat history.
    fn new(session: LlamaSession) -> Self {
        Self {
            history: Vec::new(),
            session,
        }
    }
}
