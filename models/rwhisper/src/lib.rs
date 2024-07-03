//! # rwhisper
//! A Rust wrapper for [whisper](https://openai.com/research/whisper)
//!
//! ## Usage
//!
//! ```rust, no_run
//! use futures_util::StreamExt;
//! use rwhisper::*;
//! use tokio::time::{Duration, Instant};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), anyhow::Error> {
//!     // Create a new small whisper model.
//!     let model = WhisperBuilder::default()
//!         .with_source(WhisperSource::SmallEn)
//!         .build()
//!         .await?;
//!
//!     // Record audio from the microphone for 5 seconds.
//!     let audio = kalosm_sound::MicInput::default()
//!         .record_until(Instant::now() + Duration::from_secs(5))
//!         .await?;
//!
//!     // Transcribe the audio.
//!     let mut text = model.transcribe(audio)?;
//!
//!     // As the model transcribes the audio, print the text to the console.
//!     while let Some(text) = text.next().await {
//!         print!("{}", text.text());
//!     }
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]

use cpal::FromSample;
use kalosm_common::FileSource;
pub use kalosm_common::ModelLoadingProgress;
use kalosm_language_model::ModelBuilder;
use kalosm_streams::text_stream::ChannelTextStream;
use model::WhisperInner;
use rodio::{source::UniformSourceIterator, Source};
use std::{fmt::Display, str::FromStr, time::Duration};

use anyhow::Result;

use candle_transformers::models::whisper::{self as m};

use futures_util::{Stream, StreamExt};

mod model;
mod source;
pub use source::*;

#[derive(Debug, Clone)]
struct DecodingResult {
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    compression_ratio: f64,
}

/// A transcribed segment of audio.
#[derive(Debug, Clone)]
pub struct Segment {
    start: f64,
    duration: f64,
    elapsed_time: Duration,
    remaining_time: Duration,
    progress: f32,
    result: DecodingResult,
}

impl Segment {
    /// Get the probability of no speech.
    pub fn probability_of_no_speech(&self) -> f64 {
        self.result.no_speech_prob
    }

    /// Get the text of the segment.
    pub fn text(&self) -> &str {
        &self.result.text
    }

    /// Get the start timestamp of the segment.
    pub fn start(&self) -> f64 {
        self.start
    }

    /// Get the duration of the segment.
    pub fn duration(&self) -> f64 {
        self.duration
    }

    /// Get the elapsed time
    pub fn elapsed_time(&self) -> Duration {
        self.elapsed_time
    }

    /// Get the estimated time remaining to process the entire audio file
    pub fn remaining_time(&self) -> Duration {
        self.remaining_time
    }

    /// The progress of the transcription, from 0 to 1
    pub fn progress(&self) -> f32 {
        self.progress
    }
}

impl AsRef<str> for Segment {
    fn as_ref(&self) -> &str {
        if self.probability_of_no_speech() < 0.10 {
            self.text()
        } else {
            ""
        }
    }
}

/// An extension trait for transcribing audio streams.
pub trait TranscribeAudioStreamExt {
    /// Transcribe the audio stream.
    fn text(self, model: Whisper) -> ChannelTextStream<Segment>;
}

impl<S> TranscribeAudioStreamExt for S
where
    S: Stream + std::marker::Unpin + Send + 'static,
    <S as Stream>::Item: Source + Send + 'static,
    <<S as Stream>::Item as Iterator>::Item: rodio::Sample,
    f32: FromSample<<<S as Stream>::Item as Iterator>::Item>,
{
    fn text(self, model: Whisper) -> ChannelTextStream<Segment> {
        let mut stream = self;
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Some(source) = stream.next().await {
                let result = { model.transcribe(source) };
                match result {
                    Ok(mut stream) => {
                        while let Some(segment) = stream.next().await {
                            if let Err(err) = sender.send(segment) {
                                tracing::error!("error sending segment: {}", err);
                                return;
                            }
                        }
                    }
                    Err(err) => tracing::error!("error transcribing audio: {}", err),
                }
            }
        });
        ChannelTextStream::from(receiver)
    }
}

#[derive(Clone, Copy, Debug)]
enum Task {
    Transcribe,
    #[allow(dead_code)]
    Translate,
}

/// A builder with configuration for a Whisper model.
#[derive(Debug)]
pub struct WhisperBuilder {
    /// The model to be used, can be tiny, small, medium.
    model: WhisperSource,

    /// Language.
    language: Option<WhisperLanguage>,

    /// The cache location to use for the model (defaults DATA_DIR/kalosm/cache)
    cache: kalosm_common::Cache,
}

impl Default for WhisperBuilder {
    fn default() -> Self {
        Self {
            model: WhisperSource::default(),
            language: Some(WhisperLanguage::English),
            cache: kalosm_common::Cache::default(),
        }
    }
}

#[async_trait::async_trait]
impl ModelBuilder for WhisperBuilder {
    type Model = Whisper;

    async fn start_with_loading_handler(
        self,
        handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> anyhow::Result<Self::Model> {
        self.build_with_loading_handler(handler).await
    }

    fn requires_download(&self) -> bool {
        let whisper = self.get_whisper_model_config();
        !whisper.model.downloaded()
            || !whisper.tokenizer.downloaded()
            || !whisper.config.downloaded()
    }
}

impl WhisperBuilder {
    fn get_whisper_model_config(&self) -> WhisperModelConfig {
        let (model_id, revision) = self.model.model_and_revision();
        if self.model.is_quantized() {
            match self.model {
                WhisperSource::QuantizedTinyEn => {
                    let model = FileSource::huggingface(
                        model_id.to_owned(),
                        revision.to_owned(),
                        "model-tiny-en-q80.gguf".to_owned(),
                    );
                    let tokenizer = FileSource::huggingface(
                        model_id.to_owned(),
                        revision.to_owned(),
                        "tokenizer-tiny-en.json".to_owned(),
                    );
                    let config = FileSource::huggingface(
                        model_id.to_owned(),
                        revision.to_owned(),
                        "config-tiny-en.json".to_owned(),
                    );
                    WhisperModelConfig::new(model, tokenizer, config)
                }
                WhisperSource::QuantizedTiny => {
                    let model = FileSource::huggingface(
                        model_id.to_owned(),
                        revision.to_owned(),
                        "model-tiny-q80.gguf".to_owned(),
                    );
                    let tokenizer = FileSource::huggingface(
                        model_id.to_owned(),
                        revision.to_owned(),
                        "tokenizer-tiny.json".to_owned(),
                    );
                    let config = FileSource::huggingface(
                        model_id.to_owned(),
                        revision.to_owned(),
                        "config-tiny.json".to_owned(),
                    );
                    WhisperModelConfig::new(model, tokenizer, config)
                }
                WhisperSource::QuantizedDistilLargeV3 => {
                    let model = FileSource::huggingface(
                        model_id.to_owned(),
                        revision.to_owned(),
                        "model.gguf".to_owned(),
                    );
                    let tokenizer = FileSource::huggingface(
                        model_id.to_owned(),
                        revision.to_owned(),
                        "tokenizer.json".to_owned(),
                    );
                    let config = FileSource::huggingface(
                        model_id.to_owned(),
                        revision.to_owned(),
                        "config.json".to_owned(),
                    );
                    WhisperModelConfig::new(model, tokenizer, config)
                }
                _ => unreachable!(),
            }
        } else {
            let model = FileSource::huggingface(
                model_id.to_owned(),
                revision.to_owned(),
                "model.safetensors".to_owned(),
            );
            let tokenizer = FileSource::huggingface(
                model_id.to_owned(),
                revision.to_owned(),
                "tokenizer.json".to_owned(),
            );
            let config = FileSource::huggingface(
                model_id.to_owned(),
                revision.to_owned(),
                "config.json".to_owned(),
            );
            WhisperModelConfig::new(model, tokenizer, config)
        }
    }

    /// Build the model.
    pub async fn build(self) -> anyhow::Result<Whisper> {
        self.build_with_loading_handler(ModelLoadingProgress::multi_bar_loading_indicator())
            .await
    }

    /// Build the model with a handler for progress as the download and loading progresses.
    pub async fn build_with_loading_handler(
        self,
        mut progress_handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> anyhow::Result<Whisper> {
        // Download section
        let whisper = self.get_whisper_model_config();
        let tokenizer_source = whisper.tokenizer;
        let model_source = whisper.model;
        let config_source = whisper.config;

        let display_tokenizer_source = format!("Tokenizer ({})", tokenizer_source);
        let mut create_progress =
            ModelLoadingProgress::downloading_progress(display_tokenizer_source);
        let tokenizer_filename = self
            .cache
            .get(&tokenizer_source, |progress| {
                progress_handler(create_progress(progress))
            })
            .await?;

        let display_model_source = format!("Model ({})", model_source);
        let mut create_progress = ModelLoadingProgress::downloading_progress(display_model_source);
        let filename = self
            .cache
            .get(&model_source, |progress| {
                progress_handler(create_progress(progress))
            })
            .await?;

        let display_config_source = format!("Config ({})", config_source);
        let mut create_progress = ModelLoadingProgress::downloading_progress(display_config_source);
        let config = self
            .cache
            .get(&config_source, |progress| {
                progress_handler(create_progress(progress))
            })
            .await?;

        let (rx, tx) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async move {
                    let mut model =
                        WhisperInner::new(self, filename, tokenizer_filename, config).unwrap();
                    while let Ok(message) = tx.recv() {
                        match message {
                            WhisperMessage::Kill => return,
                            WhisperMessage::Transcribe(input, result) => {
                                model.transcribe(input, result);
                            }
                        }
                    }
                });
        });

        Ok(Whisper {
            thread: Some(thread),
            sender: rx,
        })
    }

    /// Set the model to be used.
    pub fn with_source(mut self, model: WhisperSource) -> Self {
        self.model = model;
        self
    }

    /// Set the language to be used.
    pub fn with_language(mut self, language: Option<WhisperLanguage>) -> Self {
        self.language = language;
        self
    }

    /// Set the cache location to use for the model (defaults DATA_DIR/kalosm/cache)
    pub fn with_cache(mut self, cache: kalosm_common::Cache) -> Self {
        self.cache = cache;

        self
    }
}

/// A language whisper can use
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy)]
pub enum WhisperLanguage {
    English,
    Chinese,
    German,
    Spanish,
    Russian,
    Korean,
    French,
    Japanese,
    Portuguese,
    Turkish,
    Polish,
    Catalan,
    Dutch,
    Arabic,
    Swedish,
    Italian,
    Indonesian,
    Hindi,
    Finnish,
    Vietnamese,
    Hebrew,
    Ukrainian,
    Greek,
    Malay,
    Czech,
    Romanian,
    Danish,
    Hungarian,
    Tamil,
    Norwegian,
    Thai,
    Urdu,
    Croatian,
    Bulgarian,
    Lithuanian,
    Latin,
    Maori,
    Malayalam,
    Welsh,
    Slovak,
    Telugu,
    Persian,
    Latvian,
    Bengali,
    Serbian,
    Azerbaijani,
    Slovenian,
    Kannada,
    Estonian,
    Macedonian,
    Breton,
    Basque,
    Icelandic,
    Armenian,
    Nepali,
    Mongolian,
    Bosnian,
    Kazakh,
    Albanian,
    Swahili,
    Galician,
    Marathi,
    Punjabi,
    Sinhala,
    Khmer,
    Shona,
    Yoruba,
    Somali,
    Afrikaans,
    Occitan,
    Georgian,
    Belarusian,
    Tajik,
    Sindhi,
    Gujarati,
    Amharic,
    Yiddish,
    Lao,
    Uzbek,
    Faroese,
    HaitianCreole,
    Pashto,
    Turkmen,
    Nynorsk,
    Maltese,
    Sanskrit,
    Luxembourgish,
    Myanmar,
    Tibetan,
    Tagalog,
    Malagasy,
    Assamese,
    Tatar,
    Hawaiian,
    Lingala,
    Hausa,
    Bashkir,
    Javanese,
    Sundanese,
}

/// Error that reports the unsupported value
#[derive(PartialEq, Eq)]
pub struct ParseWhisperLanguageError(String);

impl Display for ParseWhisperLanguageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Language {} not supported ", self.0)
    }
}

impl FromStr for WhisperLanguage {
    type Err = ParseWhisperLanguageError;

    fn from_str(s: &str) -> std::prelude::v1::Result<Self, Self::Err> {
        match s {
            "en" => Ok(WhisperLanguage::English),
            "zh" => Ok(WhisperLanguage::Chinese),
            "de" => Ok(WhisperLanguage::German),
            "es" => Ok(WhisperLanguage::Spanish),
            "ru" => Ok(WhisperLanguage::Russian),
            "ko" => Ok(WhisperLanguage::Korean),
            "fr" => Ok(WhisperLanguage::French),
            "ja" => Ok(WhisperLanguage::Japanese),
            "pt" => Ok(WhisperLanguage::Portuguese),
            "tr" => Ok(WhisperLanguage::Turkish),
            "pl" => Ok(WhisperLanguage::Polish),
            "ca" => Ok(WhisperLanguage::Catalan),
            "nl" => Ok(WhisperLanguage::Dutch),
            "ar" => Ok(WhisperLanguage::Arabic),
            "sv" => Ok(WhisperLanguage::Swedish),
            "it" => Ok(WhisperLanguage::Italian),
            "id" => Ok(WhisperLanguage::Indonesian),
            "hi" => Ok(WhisperLanguage::Hindi),
            "fi" => Ok(WhisperLanguage::Finnish),
            "vi" => Ok(WhisperLanguage::Vietnamese),
            "he" => Ok(WhisperLanguage::Hebrew),
            "uk" => Ok(WhisperLanguage::Ukrainian),
            "el" => Ok(WhisperLanguage::Greek),
            "ms" => Ok(WhisperLanguage::Malay),
            "cs" => Ok(WhisperLanguage::Czech),
            "ro" => Ok(WhisperLanguage::Romanian),
            "da" => Ok(WhisperLanguage::Danish),
            "hu" => Ok(WhisperLanguage::Hungarian),
            "ta" => Ok(WhisperLanguage::Tamil),
            "no" => Ok(WhisperLanguage::Norwegian),
            "th" => Ok(WhisperLanguage::Thai),
            "ur" => Ok(WhisperLanguage::Urdu),
            "hr" => Ok(WhisperLanguage::Croatian),
            "bg" => Ok(WhisperLanguage::Bulgarian),
            "lt" => Ok(WhisperLanguage::Lithuanian),
            "la" => Ok(WhisperLanguage::Latin),
            "mi" => Ok(WhisperLanguage::Maori),
            "ml" => Ok(WhisperLanguage::Malayalam),
            "cy" => Ok(WhisperLanguage::Welsh),
            "sk" => Ok(WhisperLanguage::Slovak),
            "te" => Ok(WhisperLanguage::Telugu),
            "fa" => Ok(WhisperLanguage::Persian),
            "lv" => Ok(WhisperLanguage::Latvian),
            "bn" => Ok(WhisperLanguage::Bengali),
            "sr" => Ok(WhisperLanguage::Serbian),
            "az" => Ok(WhisperLanguage::Azerbaijani),
            "sl" => Ok(WhisperLanguage::Slovenian),
            "kn" => Ok(WhisperLanguage::Kannada),
            "et" => Ok(WhisperLanguage::Estonian),
            "mk" => Ok(WhisperLanguage::Macedonian),
            "br" => Ok(WhisperLanguage::Breton),
            "eu" => Ok(WhisperLanguage::Basque),
            "is" => Ok(WhisperLanguage::Icelandic),
            "hy" => Ok(WhisperLanguage::Armenian),
            "ne" => Ok(WhisperLanguage::Nepali),
            "mn" => Ok(WhisperLanguage::Mongolian),
            "bs" => Ok(WhisperLanguage::Bosnian),
            "kk" => Ok(WhisperLanguage::Kazakh),
            "sq" => Ok(WhisperLanguage::Albanian),
            "sw" => Ok(WhisperLanguage::Swahili),
            "gl" => Ok(WhisperLanguage::Galician),
            "mr" => Ok(WhisperLanguage::Marathi),
            "pa" => Ok(WhisperLanguage::Punjabi),
            "si" => Ok(WhisperLanguage::Sinhala),
            "km" => Ok(WhisperLanguage::Khmer),
            "sn" => Ok(WhisperLanguage::Shona),
            "yo" => Ok(WhisperLanguage::Yoruba),
            "so" => Ok(WhisperLanguage::Somali),
            "af" => Ok(WhisperLanguage::Afrikaans),
            "oc" => Ok(WhisperLanguage::Occitan),
            "ka" => Ok(WhisperLanguage::Georgian),
            "be" => Ok(WhisperLanguage::Belarusian),
            "tg" => Ok(WhisperLanguage::Tajik),
            "sd" => Ok(WhisperLanguage::Sindhi),
            "gu" => Ok(WhisperLanguage::Gujarati),
            "am" => Ok(WhisperLanguage::Amharic),
            "yi" => Ok(WhisperLanguage::Yiddish),
            "lo" => Ok(WhisperLanguage::Lao),
            "uz" => Ok(WhisperLanguage::Uzbek),
            "fo" => Ok(WhisperLanguage::Faroese),
            "ht" => Ok(WhisperLanguage::HaitianCreole),
            "ps" => Ok(WhisperLanguage::Pashto),
            "tk" => Ok(WhisperLanguage::Turkmen),
            "nn" => Ok(WhisperLanguage::Nynorsk),
            "mt" => Ok(WhisperLanguage::Maltese),
            "sa" => Ok(WhisperLanguage::Sanskrit),
            "lb" => Ok(WhisperLanguage::Luxembourgish),
            "my" => Ok(WhisperLanguage::Myanmar),
            "bo" => Ok(WhisperLanguage::Tibetan),
            "tl" => Ok(WhisperLanguage::Tagalog),
            "mg" => Ok(WhisperLanguage::Malagasy),
            "as" => Ok(WhisperLanguage::Assamese),
            "tt" => Ok(WhisperLanguage::Tatar),
            "haw" => Ok(WhisperLanguage::Hawaiian),
            "ln" => Ok(WhisperLanguage::Lingala),
            "ha" => Ok(WhisperLanguage::Hausa),
            "ba" => Ok(WhisperLanguage::Bashkir),
            "jw" => Ok(WhisperLanguage::Javanese),
            "su" => Ok(WhisperLanguage::Sundanese),
            _ => Err(ParseWhisperLanguageError(s.to_owned())),
        }
    }
}

impl Display for WhisperLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WhisperLanguage::English => write!(f, "en"),
            WhisperLanguage::Chinese => write!(f, "zh"),
            WhisperLanguage::German => write!(f, "de"),
            WhisperLanguage::Spanish => write!(f, "es"),
            WhisperLanguage::Russian => write!(f, "ru"),
            WhisperLanguage::Korean => write!(f, "ko"),
            WhisperLanguage::French => write!(f, "fr"),
            WhisperLanguage::Japanese => write!(f, "ja"),
            WhisperLanguage::Portuguese => write!(f, "pt"),
            WhisperLanguage::Turkish => write!(f, "tr"),
            WhisperLanguage::Polish => write!(f, "pl"),
            WhisperLanguage::Catalan => write!(f, "ca"),
            WhisperLanguage::Dutch => write!(f, "nl"),
            WhisperLanguage::Arabic => write!(f, "ar"),
            WhisperLanguage::Swedish => write!(f, "sv"),
            WhisperLanguage::Italian => write!(f, "it"),
            WhisperLanguage::Indonesian => write!(f, "id"),
            WhisperLanguage::Hindi => write!(f, "hi"),
            WhisperLanguage::Finnish => write!(f, "fi"),
            WhisperLanguage::Vietnamese => write!(f, "vi"),
            WhisperLanguage::Hebrew => write!(f, "he"),
            WhisperLanguage::Ukrainian => write!(f, "uk"),
            WhisperLanguage::Greek => write!(f, "el"),
            WhisperLanguage::Malay => write!(f, "ms"),
            WhisperLanguage::Czech => write!(f, "cs"),
            WhisperLanguage::Romanian => write!(f, "ro"),
            WhisperLanguage::Danish => write!(f, "da"),
            WhisperLanguage::Hungarian => write!(f, "hu"),
            WhisperLanguage::Tamil => write!(f, "ta"),
            WhisperLanguage::Norwegian => write!(f, "no"),
            WhisperLanguage::Thai => write!(f, "th"),
            WhisperLanguage::Urdu => write!(f, "ur"),
            WhisperLanguage::Croatian => write!(f, "hr"),
            WhisperLanguage::Bulgarian => write!(f, "bg"),
            WhisperLanguage::Lithuanian => write!(f, "lt"),
            WhisperLanguage::Latin => write!(f, "la"),
            WhisperLanguage::Maori => write!(f, "mi"),
            WhisperLanguage::Malayalam => write!(f, "ml"),
            WhisperLanguage::Welsh => write!(f, "cy"),
            WhisperLanguage::Slovak => write!(f, "sk"),
            WhisperLanguage::Telugu => write!(f, "te"),
            WhisperLanguage::Persian => write!(f, "fa"),
            WhisperLanguage::Latvian => write!(f, "lv"),
            WhisperLanguage::Bengali => write!(f, "bn"),
            WhisperLanguage::Serbian => write!(f, "sr"),
            WhisperLanguage::Azerbaijani => write!(f, "az"),
            WhisperLanguage::Slovenian => write!(f, "sl"),
            WhisperLanguage::Kannada => write!(f, "kn"),
            WhisperLanguage::Estonian => write!(f, "et"),
            WhisperLanguage::Macedonian => write!(f, "mk"),
            WhisperLanguage::Breton => write!(f, "br"),
            WhisperLanguage::Basque => write!(f, "eu"),
            WhisperLanguage::Icelandic => write!(f, "is"),
            WhisperLanguage::Armenian => write!(f, "hy"),
            WhisperLanguage::Nepali => write!(f, "ne"),
            WhisperLanguage::Mongolian => write!(f, "mn"),
            WhisperLanguage::Bosnian => write!(f, "bs"),
            WhisperLanguage::Kazakh => write!(f, "kk"),
            WhisperLanguage::Albanian => write!(f, "sq"),
            WhisperLanguage::Swahili => write!(f, "sw"),
            WhisperLanguage::Galician => write!(f, "gl"),
            WhisperLanguage::Marathi => write!(f, "mr"),
            WhisperLanguage::Punjabi => write!(f, "pa"),
            WhisperLanguage::Sinhala => write!(f, "si"),
            WhisperLanguage::Khmer => write!(f, "km"),
            WhisperLanguage::Shona => write!(f, "sn"),
            WhisperLanguage::Yoruba => write!(f, "yo"),
            WhisperLanguage::Somali => write!(f, "so"),
            WhisperLanguage::Afrikaans => write!(f, "af"),
            WhisperLanguage::Occitan => write!(f, "oc"),
            WhisperLanguage::Georgian => write!(f, "ka"),
            WhisperLanguage::Belarusian => write!(f, "be"),
            WhisperLanguage::Tajik => write!(f, "tg"),
            WhisperLanguage::Sindhi => write!(f, "sd"),
            WhisperLanguage::Gujarati => write!(f, "gu"),
            WhisperLanguage::Amharic => write!(f, "am"),
            WhisperLanguage::Yiddish => write!(f, "yi"),
            WhisperLanguage::Lao => write!(f, "lo"),
            WhisperLanguage::Uzbek => write!(f, "uz"),
            WhisperLanguage::Faroese => write!(f, "fo"),
            WhisperLanguage::HaitianCreole => write!(f, "ht"),
            WhisperLanguage::Pashto => write!(f, "ps"),
            WhisperLanguage::Turkmen => write!(f, "tk"),
            WhisperLanguage::Nynorsk => write!(f, "nn"),
            WhisperLanguage::Maltese => write!(f, "mt"),
            WhisperLanguage::Sanskrit => write!(f, "sa"),
            WhisperLanguage::Luxembourgish => write!(f, "lb"),
            WhisperLanguage::Myanmar => write!(f, "my"),
            WhisperLanguage::Tibetan => write!(f, "bo"),
            WhisperLanguage::Tagalog => write!(f, "tl"),
            WhisperLanguage::Malagasy => write!(f, "mg"),
            WhisperLanguage::Assamese => write!(f, "as"),
            WhisperLanguage::Tatar => write!(f, "tt"),
            WhisperLanguage::Hawaiian => write!(f, "haw"),
            WhisperLanguage::Lingala => write!(f, "ln"),
            WhisperLanguage::Hausa => write!(f, "ha"),
            WhisperLanguage::Bashkir => write!(f, "ba"),
            WhisperLanguage::Javanese => write!(f, "jw"),
            WhisperLanguage::Sundanese => write!(f, "su"),
        }
    }
}

/// A quantized whisper audio transcription model.
pub struct Whisper {
    thread: Option<std::thread::JoinHandle<()>>,
    sender: std::sync::mpsc::Sender<WhisperMessage>,
}

impl Whisper {
    /// Create a builder for a Whisper model.
    pub fn builder() -> WhisperBuilder {
        WhisperBuilder::default()
    }

    /// Create a new default whisper model.
    pub async fn new() -> Result<Self, anyhow::Error> {
        let model = Self::builder().build().await?;
        Ok(model)
    }

    /// Transcribe some audio into text.
    ///
    /// Dropping the returned channel will stop the transcription early.
    pub fn transcribe<S: Source>(&self, input: S) -> Result<ChannelTextStream<Segment>>
    where
        <S as Iterator>::Item: rodio::Sample,
        f32: FromSample<<S as Iterator>::Item>,
    {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        self.transcribe_into(input, sender)?;
        Ok(ChannelTextStream::from(receiver))
    }

    /// Transcribe some audio into a stream of text
    ///
    /// Dropping the receiver will stop the transcription early.
    pub fn transcribe_into<S: Source>(
        &self,
        input: S,
        sender: tokio::sync::mpsc::UnboundedSender<Segment>,
    ) -> Result<()>
    where
        <S as Iterator>::Item: rodio::Sample,
        f32: FromSample<<S as Iterator>::Item>,
    {
        let pcm_data: Vec<_> = normalize_audio(input)?;
        self.sender
            .send(WhisperMessage::Transcribe(pcm_data, sender))?;
        Ok(())
    }
}

impl Drop for Whisper {
    fn drop(&mut self) {
        self.sender.send(WhisperMessage::Kill).unwrap();
        self.thread.take().unwrap().join().unwrap();
    }
}

enum WhisperMessage {
    Kill,
    Transcribe(Vec<f32>, tokio::sync::mpsc::UnboundedSender<Segment>),
}

pub(crate) fn normalize_audio<S: Source>(input: S) -> Result<Vec<f32>>
where
    <S as Iterator>::Item: rodio::Sample,
    f32: FromSample<<S as Iterator>::Item>,
{
    let resample = UniformSourceIterator::new(input, 1, m::SAMPLE_RATE as u32);
    let pass_filter = resample.low_pass(3000).high_pass(200).convert_samples();

    let samples = pass_filter.collect::<Vec<f32>>();

    Ok(samples)
}
