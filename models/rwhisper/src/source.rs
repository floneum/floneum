use std::{fmt::Display, str::FromStr};

use kalosm_common::FileSource;

/// The source whisper model to use.
#[derive(Clone, Copy, Debug, Default)]
pub enum WhisperSource {
    /// The tiny model.
    Tiny,
    /// The tiny model quantized to run faster.
    QuantizedTiny,
    /// The tiny model with only English support.
    TinyEn,
    /// The tiny model with only English support quantized to run faster.
    QuantizedTinyEn,
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
    /// The distil-large-v3 model.
    DistilLargeV3,
    #[default]
    /// The quantized distil-large-v3 model.
    QuantizedDistilLargeV3,
}

impl WhisperSource {
    /// Check if the model is multilingual.
    pub fn is_multilingual(&self) -> bool {
        match self {
            Self::QuantizedTiny
            | Self::Tiny
            | Self::Base
            | Self::Small
            | Self::Medium
            | Self::Large
            | Self::LargeV2
            | Self::DistilLargeV2
            | Self::DistilLargeV3
            | Self::QuantizedDistilLargeV3 => true,
            Self::QuantizedTinyEn
            | Self::TinyEn
            | Self::BaseEn
            | Self::SmallEn
            | Self::MediumEn
            | Self::DistilMediumEn => false,
        }
    }

    /// Check if the model is quantized.
    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            Self::QuantizedTiny | Self::QuantizedTinyEn | Self::QuantizedDistilLargeV3
        )
    }

    pub(crate) fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai/whisper-tiny", "main"),
            Self::QuantizedTiny => ("lmz/candle-whisper", "main"),
            Self::TinyEn => ("openai/whisper-tiny.en", "main"),
            Self::QuantizedTinyEn => ("lmz/candle-whisper", "main"),
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
            Self::DistilLargeV3 => ("distil-whisper/distil-large-v3", "main"),
            Self::QuantizedDistilLargeV3 => {
                ("Demonthos/candle-quantized-whisper-distil-v3", "main")
            }
        }
    }
}

/// Error that reports the unsupported value
#[derive(Debug, PartialEq, Eq)]
pub struct ParseWhisperSourceError(String);

impl Display for ParseWhisperSourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Source {} not supported ", self.0)
    }
}

impl FromStr for WhisperSource {
    type Err = ParseWhisperSourceError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "tiny" => Ok(Self::Tiny),
            "quantized_tiny" => Ok(Self::QuantizedTiny),
            "tiny_en" => Ok(Self::TinyEn),
            "quantized_tiny_en" => Ok(Self::QuantizedTinyEn),
            "base" => Ok(Self::Base),
            "base_en" => Ok(Self::BaseEn),
            "small" => Ok(Self::Small),
            "small_en" => Ok(Self::SmallEn),
            "medium" => Ok(Self::Medium),
            "medium_en" => Ok(Self::MediumEn),
            "large" => Ok(Self::Large),
            "large_v2" => Ok(Self::LargeV2),
            "distil_medium_en" => Ok(Self::DistilMediumEn),
            "distil_large_v2" => Ok(Self::DistilLargeV2),
            "distil_large_v3" => Ok(Self::DistilLargeV3),
            "quantized_distil_large_v3" => Ok(Self::QuantizedDistilLargeV3),
            _ => Err(ParseWhisperSourceError(s.to_owned())),
        }
    }
}

impl Display for WhisperSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tiny => write!(f, "tiny"),
            Self::QuantizedTiny => write!(f, "quantized_tiny"),
            Self::TinyEn => write!(f, "tiny_en"),
            Self::QuantizedTinyEn => write!(f, "quantized_tiny_en"),
            Self::Base => write!(f, "base"),
            Self::BaseEn => write!(f, "base_en"),
            Self::Small => write!(f, "small"),
            Self::SmallEn => write!(f, "small_en"),
            Self::Medium => write!(f, "medium"),
            Self::MediumEn => write!(f, "medium_en"),
            Self::Large => write!(f, "large"),
            Self::LargeV2 => write!(f, "large_v2"),
            Self::DistilMediumEn => write!(f, "distil_medium_en"),
            Self::DistilLargeV2 => write!(f, "distil_large_v2"),
            Self::DistilLargeV3 => write!(f, "distil_large_v3"),
            Self::QuantizedDistilLargeV3 => write!(f, "quantized_distil_large_v3"),
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
