pub struct PhiSource {
    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    pub(crate) model_id: String,
    pub(crate) revision: String,
    pub(crate) weight_files: Vec<String>,
    pub(crate) tokenizer_file: String,
}

impl PhiSource {
    pub fn new(model_id: String, weight_files: Vec<String>, tokenizer_file: String) -> Self {
        Self {
            model_id,
            revision: "main".to_string(),
            weight_files,
            tokenizer_file,
        }
    }

    pub fn with_revision(mut self, revision: String) -> Self {
        self.revision = revision;
        self
    }

    pub fn with_tokenizer_file(mut self, tokenizer_file: String) -> Self {
        self.tokenizer_file = tokenizer_file;
        self
    }

    pub fn with_weight_files(mut self, weight_files: Vec<String>) -> Self {
        self.weight_files = weight_files;
        self
    }
}

impl Default for PhiSource {
    fn default() -> Self {
        Self {
            model_id: "lmz/candle-mistral".to_string(),
            revision: "main".to_string(),
            weight_files: vec![
                "pytorch_model-00001-of-00002.safetensors".to_string(),
                "pytorch_model-00002-of-00002.safetensors".to_string(),
            ],
            tokenizer_file: "tokenizer.json".to_string(),
        }
    }
}
