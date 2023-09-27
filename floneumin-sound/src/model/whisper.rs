use cpal::FromSample;
use floneumin_streams::sender::ChannelTextStream;
use rodio::{source::UniformSourceIterator, Source};
use std::fmt::Display;

use anyhow::{Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::{distributions::Distribution, SeedableRng};
use tokenizers::Tokenizer;

use candle_transformers::models::whisper::{self as m, audio, model};
use model::{Config, Whisper};

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Segment {
    start: f64,
    duration: f64,
    result: DecodingResult,
}

impl Segment {
    pub fn probability_of_no_speech(&self) -> f64 {
        self.result.no_speech_prob
    }

    pub fn text(&self) -> &str {
        &self.result.text
    }

    pub fn start(&self) -> f64 {
        self.start
    }

    pub fn duration(&self) -> f64 {
        self.duration
    }
}

impl AsRef<str> for Segment {
    fn as_ref(&self) -> &str {
        self.text()
    }
}

struct Decoder {
    model: Whisper,
    rng: rand::rngs::StdRng,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Whisper,
        tokenizer: Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..model.config.vocab_size as u32)
            .map(|i| {
                if model.config.suppress_tokens.contains(&i) {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = token_id(&tokenizer, m::NO_SPEECH_TOKEN)?;
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
        })
    }

    fn decode(&mut self, mel: &Tensor, t: f64, task: Task) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder.forward(mel, true)?;
        let sample_len = model.config.max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match task {
            Task::Transcribe => tokens.push(self.transcribe_token),
            Task::Translate => tokens.push(self.translate_token),
        }
        tokens.push(self.no_timestamps_token);
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder.forward(&tokens_t, &audio_features, i == 0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            // TODO: Besides suppress tokens, we should apply the heuristics from
            // ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than any other tokens,
            //   only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle_core::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > model.config.max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor, task: Task) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t, task);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    tracing::error!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    fn run(
        &mut self,
        mel: &Tensor,
        task: Task,
        result: tokio::sync::mpsc::UnboundedSender<Segment>,
    ) {
        let (_, _, content_frames) = mel.dims3().unwrap();
        let mut seek = 0;
        while seek < content_frames {
            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size).unwrap();
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment, task).unwrap();
            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                tracing::trace!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                result: dr,
            };

            if let Err(err) = result.send(segment) {
                tracing::error!("Error sending segment: {err}");
                break;
            }
        }
    }
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle_core::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle_core::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

#[derive(Clone, Copy, Debug)]
enum Task {
    Transcribe,
    #[allow(dead_code)]
    Translate,
}

#[derive(Clone, Copy, Debug)]
pub enum WhichModel {
    Tiny,
    TinyEn,
    Base,
    BaseEn,
    Small,
    SmallEn,
    Medium,
    MediumEn,
    Large,
    LargeV2,
}

impl WhichModel {
    pub fn is_multilingual(&self) -> bool {
        match self {
            Self::Tiny | Self::Base | Self::Small | Self::Medium | Self::Large | Self::LargeV2 => {
                true
            }
            Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn => false,
        }
    }

    fn model_and_revision(&self) -> (&'static str, &'static str) {
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
        }
    }
}

#[derive(Debug)]
pub struct WhisperBuilder {
    /// Run on CPU rather than on GPU.
    cpu: bool,

    /// The model to be used, can be tiny, small, medium.
    model: WhichModel,

    /// Language.
    language: Option<WhisperLanguage>,
}

impl Default for WhisperBuilder {
    fn default() -> Self {
        Self {
            cpu: false,
            model: WhichModel::LargeV2,
            language: Some(WhisperLanguage::English),
        }
    }
}

impl WhisperBuilder {
    pub fn build(self) -> anyhow::Result<WhisperModel> {
        let (rx, tx) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async move {
                    let mut model = WhisperModelInner::new(self).unwrap();
                    while let Ok(message) = tx.recv() {
                        match message {
                            WhisperMessage::Kill => return,
                            WhisperMessage::Transcribe(input, result) => {
                                let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                                model.transcribe(input, tx);
                                if result.send(rx).is_err() {
                                    break;
                                }
                            }
                        }
                    }
                });
        });

        Ok(WhisperModel {
            thread: Some(thread),
            sender: rx,
        })
    }

    pub fn cpu(mut self, cpu: bool) -> Self {
        self.cpu = cpu;
        self
    }

    pub fn model(mut self, model: WhichModel) -> Self {
        self.model = model;
        self
    }

    pub fn language(mut self, language: Option<WhisperLanguage>) -> Self {
        self.language = language;
        self
    }
}

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

pub struct WhisperModel {
    thread: Option<std::thread::JoinHandle<()>>,
    sender: std::sync::mpsc::Sender<WhisperMessage>,
}

impl WhisperModel {
    pub fn builder() -> WhisperBuilder {
        WhisperBuilder::default()
    }

    pub async fn transcribe<S: Source>(&mut self, input: S) -> Result<ChannelTextStream<Segment>>
    where
        <S as Iterator>::Item: rodio::Sample,
        f32: FromSample<<S as Iterator>::Item>,
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let pcm_data: Vec<_> = normalize_audio(input)?;
        self.sender.send(WhisperMessage::Transcribe(pcm_data, tx))?;
        Ok(rx.await?.into())
    }
}

impl Drop for WhisperModel {
    fn drop(&mut self) {
        self.sender.send(WhisperMessage::Kill).unwrap();
        self.thread.take().unwrap().join().unwrap();
    }
}

enum WhisperMessage {
    Kill,
    Transcribe(
        Vec<f32>,
        tokio::sync::oneshot::Sender<tokio::sync::mpsc::UnboundedReceiver<Segment>>,
    ),
}

struct WhisperModelInner {
    mel_filters: Vec<f32>,
    device: Device,
    decoder: Decoder,
}

impl WhisperModelInner {
    fn new(settings: WhisperBuilder) -> anyhow::Result<Self> {
        let device = device(settings.cpu)?;
        let (default_model, _) = settings.model.model_and_revision();
        let default_model = default_model.to_string();
        let path = std::path::PathBuf::from(default_model.clone());
        let (model_id, revision) = (default_model, "main".to_string());

        let (config_filename, tokenizer_filename, weights_filename) = if path.exists() {
            let mut config_filename = path.clone();
            config_filename.push("config.json");
            let mut tokenizer_filename = path.clone();
            tokenizer_filename.push("tokenizer.json");
            let mut model_filename = path;
            model_filename.push("model.safetensors");
            (config_filename, tokenizer_filename, model_filename)
        } else {
            let api = Api::new()?;
            let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
            (
                repo.get("config.json")?,
                repo.get("tokenizer.json")?,
                repo.get("model.safetensors")?,
            )
        };
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let mel_bytes = include_bytes!("melfilters.bytes");
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            mel_bytes,
            &mut mel_filters,
        );

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)? };
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
        let model = Whisper::load(&vb, config)?;
        let language_token = if settings.model.is_multilingual() {
            let language = settings.language.unwrap_or(WhisperLanguage::English);
            match token_id(&tokenizer, &format!("<|{language}|>")) {
                Ok(token_id) => Some(token_id),
                Err(_) => anyhow::bail!("language {language} is not supported"),
            }
        } else {
            None
        };
        let decoder = Decoder::new(model, tokenizer, 0, &device, language_token)?;

        Ok(Self {
            mel_filters,
            device,
            decoder,
        })
    }

    fn transcribe(
        &mut self,
        pcm_data: Vec<f32>,
        result: tokio::sync::mpsc::UnboundedSender<Segment>,
    ) {
        let mel = audio::pcm_to_mel(&pcm_data, &self.mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(mel, (1, m::N_MELS, mel_len / m::N_MELS), &self.device).unwrap();

        self.decoder.run(&mel, Task::Transcribe, result);
    }
}

pub fn device(cpu: bool) -> anyhow::Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else {
        let device = Device::cuda_if_available(0)?;
        if !device.is_cuda() {
            tracing::warn!(
                "Running on CPU, to run on GPU, build this example with `--features cuda`"
            );
        }
        Ok(device)
    }
}

pub fn normalize_audio<S: Source>(input: S) -> Result<Vec<f32>>
where
    <S as Iterator>::Item: rodio::Sample,
    f32: FromSample<<S as Iterator>::Item>,
{
    let resample = UniformSourceIterator::new(input, 1, m::SAMPLE_RATE as u32);
    let pass_filter = resample.low_pass(3000).high_pass(200).convert_samples();

    let samples = pass_filter.collect::<Vec<f32>>();

    Ok(samples)
}
