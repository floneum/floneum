/// A source for the Mistral model.
pub struct MistralSource {
    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    pub(crate) model_id: String,
    pub(crate) revision: String,
    pub(crate) gguf_file: String,
    pub(crate) tokenizer_file: String,
}

impl MistralSource {
    /// Create a new source for the Mistral model.
    pub fn new(model_id: String, gguf_file: String, tokenizer_file: String) -> Self {
        Self {
            model_id,
            revision: "main".to_string(),
            gguf_file,
            tokenizer_file,
        }
    }

    /// Set the revision of the model to use.
    pub fn with_revision(mut self, revision: String) -> Self {
        self.revision = revision;
        self
    }

    /// Set the tokenizer file to use.
    pub fn with_tokenizer_file(mut self, tokenizer_file: String) -> Self {
        self.tokenizer_file = tokenizer_file;
        self
    }

    /// Set the model (gguf) file to use.
    pub fn with_model_file(mut self, gguf_file: String) -> Self {
        self.gguf_file = gguf_file;
        self
    }
}

impl Default for MistralSource {
    fn default() -> Self {
        Self {
            model_id: "lmz/candle-mistral".to_string(),
            revision: "main".to_string(),
            gguf_file: "model-q4k.gguf".into(),
            tokenizer_file: "tokenizer.json".to_string(),
        }
    }
}
