use kalosm_language_model::ChatModel;

/// Generates embeddings of questions 
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hypothetical<M: ChatModel> {
    model: M,
}

impl<M: ChatModel> Hypothetical<M> {
    /// Create a new hypothetical chunker.
    pub fn new(model: M) -> Self {
        Self { model }
    }
}
