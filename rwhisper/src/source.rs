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
    /// The large model with only English support.
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
            Self::TinyEn => ("openai/whisper-tiny.en", "refs/pr/15"),
            Self::Base => ("openai/whisper-base", "refs/pr/22"),
            Self::BaseEn => ("openai/whisper-base.en", "refs/pr/13"),
            Self::Small => ("openai/whisper-small", "main"),
            Self::SmallEn => ("openai/whisper-small.en", "refs/pr/10"),
            Self::Medium => ("openai/whisper-medium", "main"),
            Self::MediumEn => ("openai/whisper-medium.en", "main"),
            Self::Large => ("openai/whisper-large", "refs/pr/36"),
            Self::LargeV2 => ("openai/whisper-large-v2", "refs/pr/57"),
            Self::DistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
            Self::DistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
        }
    }
}
