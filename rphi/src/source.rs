/// A PhiSource is the source to fetch a Phi-1.5 model from.
/// The model to use, check out available models: <https://huggingface.co/models?other=mixformer-sequential&sort=trending&search=phi>
/// The model must have a quantized version available with a safetensors file. (for example lmz/candle-quantized-phi)
pub struct PhiSource {
    pub(crate) model_id: String,
    pub(crate) revision: String,
    pub(crate) model_file: String,
}

impl PhiSource {
    /// Create a new PhiSource with the given model id.
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            revision: "main".to_string(),
            model_file: "model-q4k.gguf".to_string(),
        }
    }

    /// Set the revision to use for the model.
    pub fn with_revision(mut self, revision: String) -> Self {
        self.revision = revision;
        self
    }
}

impl Default for PhiSource {
    fn default() -> Self {
        let default_model = "lmz/candle-quantized-phi".to_string();
        let default_revision = "main".to_string();
        let default_model_file = "model-q4k.gguf".to_string();
        Self {
            model_id: default_model,
            revision: default_revision,
            model_file: default_model_file,
        }
    }
}
