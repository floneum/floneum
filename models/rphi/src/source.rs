use kalosm_common::FileSource;
use kalosm_language_model::ChatMarkers;

/// A PhiSource is the source to fetch a Phi-1.5 model from.
/// The model to use, check out available models: <https://huggingface.co/models?other=mixformer-sequential&sort=trending&search=phi>
/// The model must have a quantized version available with a safetensors file. (for example lmz/candle-quantized-phi)
pub struct PhiSource {
    pub(crate) model: FileSource,
    pub(crate) tokenizer: FileSource,
    pub(crate) phi_config: crate::Config,
    pub(crate) phi2: bool,
    pub(crate) chat_markers: Option<ChatMarkers>,
}

impl PhiSource {
    /// Create a new PhiSource with the given model id.
    pub fn new(model: FileSource, tokenizer: FileSource, phi_config: crate::Config) -> Self {
        Self {
            model,
            tokenizer,
            phi_config,
            phi2: false,
            chat_markers: Default::default(),
        }
    }

    /// The phi-v1 model.
    pub fn v1() -> Self {
        Self::new(
            FileSource::huggingface(
                "lmz/candle-quantized-phi".to_string(),
                "main".to_string(),
                "model-v1-q4k.gguf".to_string(),
            ),
            FileSource::huggingface(
                "lmz/candle-quantized-phi".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            crate::Config::v1(),
        )
    }

    /// The phi-1.5 model.
    pub fn v1_5() -> Self {
        Self::new(
            FileSource::huggingface(
                "lmz/candle-quantized-phi".to_string(),
                "main".to_string(),
                "model-q4k.gguf".to_string(),
            ),
            FileSource::huggingface(
                "lmz/candle-quantized-phi".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            crate::Config::v1_5(),
        )
    }

    /// The phi-2 model.
    pub fn v2() -> Self {
        let mut myself = Self::new(
            FileSource::huggingface(
                "lmz/candle-quantized-phi".to_string(),
                "main".to_string(),
                "model-v2-q4k.gguf".to_string(),
            ),
            FileSource::huggingface(
                "lmz/candle-quantized-phi".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            crate::Config::v2(),
        );
        myself.phi2 = true;
        myself
    }

    /// The puffin model based on phi-1.5.
    pub fn puffin_phi_v2() -> Self {
        Self::new(
            FileSource::huggingface(
                "lmz/candle-quantized-phi".to_string(),
                "main".to_string(),
                "model-puffin-phi-v2-q4k.gguf".to_string(),
            ),
            FileSource::huggingface(
                "lmz/candle-quantized-phi".to_string(),
                "main".to_string(),
                "tokenizer-puffin-phi-v2.json".to_string(),
            ),
            crate::Config::puffin_phi_v2(),
        )
    }

    /// The dolphin model based on phi-2.
    pub fn dolphin_phi_v2() -> Self {
        let mut myself = Self::new(
            FileSource::huggingface(
                "Demonthos/dolphin-2_6-phi-2-candle".to_string(),
                "main".to_string(),
                "model-q4k.gguf".to_string(),
            ),
            FileSource::huggingface(
                "Demonthos/dolphin-2_6-phi-2-candle".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            crate::Config::v2(),
        )
        .with_chat_markers(ChatMarkers {
            user_marker: "<|im_start|>user",
            end_user_marker: "<|im_end|>",
            assistant_marker: "<|im_start|>assistant",
            end_assistant_marker: "<|im_end|>",
            system_prompt_marker: "<|im_start|>system",
            end_system_prompt_marker: "<|im_end|>",
        });
        myself.phi2 = true;
        myself
    }

    /// Set the phi config to use for the model.
    pub fn with_phi_config(mut self, phi_config: crate::Config) -> Self {
        self.phi_config = phi_config;
        self
    }

    /// Set the chat markers to use for the model.
    pub fn with_chat_markers(mut self, chat_markers: ChatMarkers) -> Self {
        self.chat_markers = Some(chat_markers);
        self
    }
}

impl Default for PhiSource {
    fn default() -> Self {
        Self::v1_5()
    }
}
