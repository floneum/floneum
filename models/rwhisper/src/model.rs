use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{anyhow, Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::ops::softmax;
use rand::{distributions::Distribution, SeedableRng};
use tokenizers::Tokenizer;

use candle_transformers::models::whisper::{self as m, audio, Config};
use kalosm_common::accelerated_device_if_available;

use crate::{quantized::TextDecoderCache, Task, WhisperBuilder, WhisperLanguage};

use super::{DecodingResult, Segment};

enum ModelType {
    Quantized(crate::quantized::Whisper),
    Unquantized(m::model::Whisper),
}

impl ModelType {
    fn load(
        weights_filename: &PathBuf,
        device: &Device,
        config: Config,
        quantized: bool,
    ) -> Result<Self> {
        if quantized {
            let vb = crate::m::quantized_model::VarBuilder::from_gguf(weights_filename, device)?;
            Ok(Self::Quantized(crate::quantized::Whisper::load(
                &vb, config,
            )?))
        } else {
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[weights_filename],
                    m::DTYPE,
                    device,
                )?
            };
            Ok(Self::Unquantized(m::model::Whisper::load(&vb, config)?))
        }
    }

    fn config(&self) -> &Config {
        match self {
            Self::Quantized(model) => &model.config,
            Self::Unquantized(model) => &model.config,
        }
    }
}

pub(crate) struct WhisperInner {
    mel_filters: Vec<f32>,
    device: Device,
    decoder: Decoder,
    config: Config,
}

impl WhisperInner {
    pub(crate) fn new(
        settings: WhisperBuilder,
        weights_filename: PathBuf,
        tokenizer_filename: PathBuf,
        config_filename: PathBuf,
    ) -> anyhow::Result<Self> {
        let device = accelerated_device_if_available()?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;

        let mel_bytes = match config.num_mel_bins {
            80 => include_bytes!("melfilters.bytes").as_slice(),
            128 => include_bytes!("melfilters128.bytes").as_slice(),
            nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            mel_bytes,
            &mut mel_filters,
        );

        let model = ModelType::load(
            &weights_filename,
            &device,
            config.clone(),
            settings.model.is_quantized(),
        )?;
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
            config,
        })
    }

    pub(crate) fn transcribe(
        &mut self,
        pcm_data: Vec<f32>,
        result: tokio::sync::mpsc::UnboundedSender<Segment>,
    ) {
        let mel = audio::pcm_to_mel(&self.config, &pcm_data, &self.mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (
                1,
                self.config.num_mel_bins,
                mel_len / self.config.num_mel_bins,
            ),
            &self.device,
        )
        .unwrap();

        self.decoder.run(&mel, Task::Transcribe, result);
    }
}

struct Decoder {
    model: ModelType,
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
        model: ModelType,
        tokenizer: Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i) {
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
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok())
            .ok_or(anyhow!("no_speech_token not found"))?;
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
        let audio_features = match model {
            ModelType::Quantized(model) => model.encoder.forward(mel)?,
            ModelType::Unquantized(model) => model.encoder.forward(mel, true)?,
        };
        let sample_len = model.config().max_target_positions / 2;
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
        // The tokens that are queued for decoding
        let mut queued_tokens = tokens.clone();
        let mut cache = TextDecoderCache::new();
        for i in 0..sample_len {
            let ys = match model {
                ModelType::Quantized(model) => {
                    let result =
                        model
                            .decoder
                            .forward(&queued_tokens, &audio_features, &mut cache)?;
                    // The quantized model caches tokens so it we can remove any old tokens
                    queued_tokens.clear();
                    result
                }
                ModelType::Unquantized(model) => {
                    let tokens_t = Tensor::new(queued_tokens.as_slice(), mel.device())?;
                    // The model expects a batch dim but this inference loop does not handle
                    // it so we add it at this point.
                    let tokens_t = tokens_t.unsqueeze(0)?;
                    model.decoder.forward(&tokens_t, &audio_features, i == 0)?
                }
            };

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = match model {
                    ModelType::Quantized(model) => model.decoder.final_linear(&ys.i(..1)?)?,
                    ModelType::Unquantized(model) => model.decoder.final_linear(&ys.i(..1)?)?,
                }
                .i(0)?
                .i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = match model {
                ModelType::Quantized(model) => {
                    model.decoder.final_linear(&ys.i((..1, seq_len - 1..))?)?
                }
                ModelType::Unquantized(model) => {
                    model.decoder.final_linear(&ys.i((..1, seq_len - 1..))?)?
                }
            }
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
            queued_tokens.push(next_token);
            let prob = softmax(&logits, candle_core::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > model.config().max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            text,
            avg_logprob,
            no_speech_prob,
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
        let start_time = Instant::now();
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
            let elapsed = start_time.elapsed();
            let remaining = Duration::from_millis(
                ((elapsed.as_millis() as usize / seek) * (content_frames - seek)) as u64,
            );
            let progress = seek as f32 / content_frames as f32;
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                remaining_time: remaining,
                elapsed_time: elapsed,
                progress,
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
