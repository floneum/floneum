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
    /// The medium model with only English support quantized to run faster.
    QuantizedDistilMediumEn,
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
    /// The quantized distil-large-v3 model.
    QuantizedDistilLargeV3,
    #[default]
    /// The quantized large-v3-turbo model.
    QuantizedLargeV3Turbo,
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
            | Self::QuantizedDistilLargeV3
            | Self::QuantizedLargeV3Turbo => true,
            Self::QuantizedTinyEn
            | Self::TinyEn
            | Self::BaseEn
            | Self::SmallEn
            | Self::MediumEn
            | Self::QuantizedDistilMediumEn
            | Self::DistilMediumEn => false,
        }
    }

    /// Check if the model is quantized.
    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            Self::QuantizedTiny
                | Self::QuantizedTinyEn
                | Self::QuantizedDistilMediumEn
                | Self::QuantizedDistilLargeV3
                | Self::QuantizedLargeV3Turbo
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
            Self::QuantizedDistilMediumEn => {
                ("Demonthos/candle-quantized-whisper-medium-distil", "main")
            }
            Self::QuantizedDistilLargeV3 => {
                ("Demonthos/candle-quantized-whisper-distil-v3", "main")
            }
            Self::QuantizedLargeV3Turbo => {
                ("Demonthos/candle-quantized-whisper-large-v3-turbo", "main")
            }
        }
    }

    pub(crate) fn timestamp_attention_heads(&self) -> Option<&'static [[usize; 2]]> {
        match self {
            Self::QuantizedDistilMediumEn | Self::DistilMediumEn | Self::DistilLargeV2 => None,
            Self::QuantizedTiny | Self::Tiny => {
                Some(&[[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]])
            }
            Self::QuantizedTinyEn | Self::TinyEn => Some(&[
                [1, 0],
                [2, 0],
                [2, 5],
                [3, 0],
                [3, 1],
                [3, 2],
                [3, 3],
                [3, 4],
            ]),
            Self::Base => Some(&[
                [3, 1],
                [4, 2],
                [4, 3],
                [4, 7],
                [5, 1],
                [5, 2],
                [5, 4],
                [5, 6],
            ]),
            Self::BaseEn => Some(&[[3, 3], [4, 7], [5, 1], [5, 5], [5, 7]]),
            Self::Small => Some(&[
                [5, 3],
                [5, 9],
                [8, 0],
                [8, 4],
                [8, 7],
                [8, 8],
                [9, 0],
                [9, 7],
                [9, 9],
                [10, 5],
            ]),
            Self::SmallEn => Some(&[
                [6, 6],
                [7, 0],
                [7, 3],
                [7, 8],
                [8, 2],
                [8, 5],
                [8, 7],
                [9, 0],
                [9, 4],
                [9, 8],
                [9, 10],
                [10, 0],
                [10, 1],
                [10, 2],
                [10, 3],
                [10, 6],
                [10, 11],
                [11, 2],
                [11, 4],
            ]),
            Self::Medium => Some(&[[13, 15], [15, 4], [15, 15], [16, 1], [20, 0], [23, 4]]),
            Self::MediumEn => Some(&[
                [11, 4],
                [14, 1],
                [14, 12],
                [14, 14],
                [15, 4],
                [16, 0],
                [16, 4],
                [16, 9],
                [17, 12],
                [17, 14],
                [18, 7],
                [18, 10],
                [18, 15],
                [20, 0],
                [20, 3],
                [20, 9],
                [20, 14],
                [21, 12],
            ]),
            Self::Large => Some(&[
                [9, 19],
                [11, 2],
                [11, 4],
                [11, 17],
                [22, 7],
                [22, 11],
                [22, 17],
                [23, 2],
                [23, 15],
            ]),
            Self::LargeV2 => Some(&[
                [10, 12],
                [13, 17],
                [16, 11],
                [16, 12],
                [16, 13],
                [17, 15],
                [17, 16],
                [18, 4],
                [18, 11],
                [18, 19],
                [19, 11],
                [21, 2],
                [21, 3],
                [22, 3],
                [22, 9],
                [22, 12],
                [23, 5],
                [23, 7],
                [23, 13],
                [25, 5],
                [26, 1],
                [26, 12],
                [27, 15],
            ]),
            Self::QuantizedLargeV3Turbo => {
                Some(&[[2, 4], [2, 11], [3, 3], [3, 6], [3, 11], [3, 14]])
            }
            Self::DistilLargeV3 | Self::QuantizedDistilLargeV3 => Some(&[
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [1, 6],
                [1, 7],
                [1, 8],
                [1, 9],
                [1, 10],
                [1, 11],
                [1, 12],
                [1, 13],
                [1, 14],
                [1, 15],
                [1, 16],
                [1, 17],
                [1, 18],
                [1, 19],
            ]),
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
            Self::QuantizedDistilMediumEn => write!(f, "quantized_distil_medium_en"),
            Self::QuantizedDistilLargeV3 => write!(f, "quantized_distil_large_v3"),
            Self::QuantizedLargeV3Turbo => write!(f, "quantized_large_v3_turbo"),
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
