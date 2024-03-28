use kalosm_common::FileSource;

/// The source whisper model to use.
#[derive(Clone, Copy, Debug)]
pub enum WhisperSource {
    /// The tiny model.
    Tiny,
    /// The tiny model with only English support.
    TinyEn,
    /// The base model.
    Base,
    /// The base model with only English support.
    BaseEn,
    /// The small model.
    Small,
    /// The small model with only English support.
    SmallEn,
    /// The medium model.
    Medium,
    /// The medium model with only English support.
    MediumEn,
    /// The large model.
    Large,
    /// The large model v2.
    LargeV2,
    /// The distil-medium english model.
    DistilMediumEn,
    /// The distil-large model.
    DistilLargeV2,
}

impl WhisperSource {
    /// Check if the model is multilingual.
    pub fn is_multilingual(&self) -> bool {
        match self {
            Self::Tiny
            | Self::Base
            | Self::Small
            | Self::Medium
            | Self::Large
            | Self::LargeV2
            | Self::DistilLargeV2 => true,
            Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn | Self::DistilMediumEn => {
                false
            }
        }
    }

    pub(crate) fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai/whisper-tiny", "main"),
            Self::TinyEn => ("openai/whisper-tiny.en", "main"),
            Self::Base => ("openai/whisper-base", "main"),
            Self::BaseEn => ("openai/whisper-base.en", "main"),
            Self::Small => ("openai/whisper-small", "main"),
            Self::SmallEn => ("openai/whisper-small.en", "main"),
            Self::Medium => ("openai/whisper-medium", "main"),
            Self::MediumEn => ("openai/whisper-medium.en", "main"),
            Self::Large => ("openai/whisper-large", "main"),
            Self::LargeV2 => ("openai/whisper-large-v2", "main"),
            Self::DistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
            Self::DistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
        }
    }
}

pub(crate) struct WhisperModelConfig {
    pub(crate) model: FileSource,
    pub(crate) tokenizer: FileSource,
    pub(crate) config: FileSource,
}
impl WhisperModelConfig {
    pub(crate) fn new(model: FileSource, tokenizer: FileSource, config: FileSource) -> Self {
        Self {
            model,
            tokenizer,
            config,
        }
    }
}
