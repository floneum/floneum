/// The chat markers to use for the model.
#[derive(Default)]
pub struct ChatMarkers {
    /// The marker to use before user input.
    pub user_marker: Option<&'static str>,
    /// The marker to use after user input.
    pub end_user_marker: Option<&'static str>,
    /// The marker to use before assistant messages.
    pub assistant_marker: Option<&'static str>,
    /// The marker to use after assistant messages.
    pub end_assistant_marker: Option<&'static str>,
    /// The marker to use before system prompts.
    pub system_prompt_marker: Option<&'static str>,
    /// The marker to use after system prompts.
    pub end_system_marker: Option<&'static str>,
}

/// A PhiSource is the source to fetch a Phi-1.5 model from.
/// The model to use, check out available models: <https://huggingface.co/models?other=mixformer-sequential&sort=trending&search=phi>
/// The model must have a quantized version available with a safetensors file. (for example lmz/candle-quantized-phi)
pub struct PhiSource {
    pub(crate) model_id: String,
    pub(crate) revision: String,
    pub(crate) model_file: String,
    pub(crate) tokenizer_file: String,
    pub(crate) phi_config: crate::Config,
    pub(crate) phi2: bool,
    pub(crate) chat_markers: ChatMarkers,
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
            phi2: false,
            chat_markers: ChatMarkers::default(),
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

    /// The phi-2 model.
    pub fn v2() -> Self {
        let mut myself = Self::new("lmz/candle-quantized-phi".to_string())
            .with_model_file("model-v2-q4k.gguf".to_string())
            .with_tokenizer_file("tokenizer.json".to_string())
            .with_phi_config(crate::Config::v2());
        myself.phi2 = true;
        myself
    }

    /// The puffin model based on phi-1.5.
    pub fn puffin_phi_v2() -> Self {
        Self::new("lmz/candle-quantized-phi".to_string())
            .with_model_file("model-puffin-phi-v2-q4k.gguf".to_string())
            .with_tokenizer_file("tokenizer-puffin-phi-v2.json".to_string())
            .with_phi_config(crate::Config::puffin_phi_v2())
    }

    /// The dolphin model based on phi-2.
    pub fn dolphin_phi_v2() -> Self {
        let mut myself = Self::new("Demonthos/dolphin-2_6-phi-2-candle".to_string())
            .with_model_file("model-q4k.gguf".to_string())
            .with_tokenizer_file("tokenizer.json".to_string())
            .with_phi_config(crate::Config::v2())
            .with_chat_markers(ChatMarkers {
                user_marker: Some("<|im_start|>user"),
                end_user_marker: Some("<|im_end|>"),
                assistant_marker: Some("<|im_start|>assistant"),
                end_assistant_marker: Some("<|im_end|>"),
                system_prompt_marker: Some("<|im_start|>system"),
                end_system_marker: Some("<|im_end|>"),
            });
        myself.phi2 = true;
        myself
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

    /// Set the chat markers to use for the model.
    pub fn with_chat_markers(mut self, chat_markers: ChatMarkers) -> Self {
        self.chat_markers = chat_markers;
        self
    }
}

impl Default for PhiSource {
    fn default() -> Self {
        Self::v1_5()
    }
}
