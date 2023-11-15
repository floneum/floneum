/// A PhiSource is the source to fetch a Phi-1.5 model from.
/// The model to use, check out available models: <https://huggingface.co/models?other=mixformer-sequential&sort=trending&search=phi>
/// The model must have a quantized version available with a safetensors file. (for example lmz/candle-quantized-phi)
pub struct PhiSource {
    pub(crate) model_id: String,
    pub(crate) revision: String,
    pub(crate) model_file: String,
    pub(crate) tokenizer_file: String,
    pub(crate) phi_config: crate::Config,
}

impl PhiSource {
    /// Create a new PhiSource with the given model id.
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            revision: "main".to_string(),
            model_file: "model-q4k.gguf".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            phi_config: crate::Config::v1_5(),
        }
    }

    /// The phi-v1 model.
    pub fn v1() -> Self {
        Self::new("lmz/candle-quantized-phi".to_string())
            .with_model_file("model-v1-q4k.gguf".to_string())
            .with_tokenizer_file("tokenizer.json".to_string())
            .with_phi_config(crate::Config::v1())
    }

    /// The phi-1.5 model.
    pub fn v1_5() -> Self {
        Self::new("lmz/candle-quantized-phi".to_string())
            .with_model_file("model-q4k.gguf".to_string())
            .with_tokenizer_file("tokenizer.json".to_string())
            .with_phi_config(crate::Config::v1_5())
    }

    /// The puffin model based on phi-1.5.
    pub fn puffin_phi_v2() -> Self {
        Self::new("lmz/candle-quantized-phi".to_string())
            .with_model_file("model-puffin-phi-v2-q4k.gguf".to_string())
            .with_tokenizer_file("tokenizer-puffin-phi-v2.json".to_string())
            .with_phi_config(crate::Config::puffin_phi_v2())
    }

    /// Set the revision to use for the model.
    pub fn with_revision(mut self, revision: String) -> Self {
        self.revision = revision;
        self
    }

    /// Set the model file to use for the model.
    pub fn with_model_file(mut self, model_file: String) -> Self {
        self.model_file = model_file;
        self
    }

    /// Set the tokenizer file to use for the model.
    pub fn with_tokenizer_file(mut self, tokenizer_file: String) -> Self {
        self.tokenizer_file = tokenizer_file;
        self
    }

    /// Set the phi config to use for the model.
    pub fn with_phi_config(mut self, phi_config: crate::Config) -> Self {
        self.phi_config = phi_config;
        self
    }
}

impl Default for PhiSource {
    fn default() -> Self {
        Self::v1_5()
    }
}
