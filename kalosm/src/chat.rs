use kalosm_language::Model;

/// The history of a chat session.
pub struct ChatHistory {
    user_marker: String,
    assistant_marker: String,
    messages: Vec<String>,
}

impl ChatHistory {
    /// Creates a new chat history.
    pub fn new(user_marker: String, assistant_marker: String) -> Self {
        Self {
            user_marker,
            assistant_marker,
            messages: Vec::new(),
        }
    }

    /// Adds a message to the history.
    pub fn add_message(&mut self, message: String, is_user: bool) {
        let marker = if is_user {
            &self.user_marker
        } else {
            &self.assistant_marker
        };
        self.messages.push(format!("{} {}", marker, message));
    }
}

/// A chat session.
pub struct Chat<'a, M: Model> {
    model: &'a mut M,
    history: ChatHistory,
}

/// A model that has a chat format.
pub trait ChatModel {
    fn user_marker(&self) -> &str;
    fn assistant_marker(&self) -> &str;
    fn start_chat(&mut self);
}
