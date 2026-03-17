use std::{
    collections::VecDeque,
    fs,
    path::{Path, PathBuf},
};

use flate2::read::GzDecoder;
use fusor::VarBuilder;
use fusor_gguf::GgufValue;
use midly::{
    Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind,
    num::{u4, u7, u15, u24, u28},
};
use rand::{Rng, rngs::StdRng};
use tar::Archive;

use crate::config::RuntimeConfig;

const NESMDB_ARCHIVE_NAME: &str = "nesmdb_midi.tar.gz";
const NESMDB_DOWNLOAD_URL: &str = "https://drive.usercontent.google.com/download?id=1w2uo1Cmio4gz6nGUhZOtzF54kPkoKyo7&export=download&confirm=t";
const BOS_TOKEN: &str = "<bos>";
const EOT_TOKEN: &str = "<eot>";
const WAIT_CHUNK_MAX: u32 = 100;
const MIDI_TICKS_PER_BEAT: u16 = 100;
const MIDI_TEMPO_US_PER_BEAT: u32 = 1_000_000;
const QUANTIZATION_US_PER_STEP: u64 = 10_000;
const DEFAULT_TEMPO_US_PER_BEAT: u32 = 500_000;
const VELOCITY_CONTROLLER: u8 = 11;
const TIMBRE_CONTROLLER: u8 = 12;

#[derive(Clone)]
pub struct MidiTokenizer {
    tokens: Vec<String>,
    bos_token: u32,
    eot_token: u32,
    wait_start: u32,
    voice_offsets: [VoiceTokenOffsets; 4],
}

#[derive(Clone, Copy)]
struct VoiceTokenOffsets {
    note_on_start: u32,
    note_off_start: u32,
    velocity_start: Option<u32>,
    timbre_start: Option<u32>,
}

pub struct SourceDataset {
    files: Vec<SourceFile>,
}

#[derive(Clone)]
pub struct SourceFile {
    path: String,
    tokens: Vec<u32>,
    prompt_tokens: Vec<u32>,
    target_tokens: Vec<u32>,
    target_start: usize,
}

pub struct DatasetSplit {
    pub train: SourceDataset,
    pub validation: SourceDataset,
    pub test: SourceDataset,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum Voice {
    P1,
    P2,
    Tr,
    No,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenKind {
    Bos,
    Eot,
    Wait(u32),
    NoteOn(Voice, u8),
    NoteOff(Voice, u8),
    Velocity(Voice, u8),
    Timbre(Voice, u8),
}

#[derive(Clone)]
struct MidiEvent {
    step: u32,
    sequence: usize,
    kind: RenderEventKind,
}

#[derive(Clone)]
enum RenderEventKind {
    NoteOn { voice: Voice, note: u8 },
    NoteOff { voice: Voice, note: u8 },
    Velocity { voice: Voice, value: u8 },
    Timbre { voice: Voice, value: u8 },
}

#[derive(Clone, Copy)]
struct TempoSegment {
    start_tick: u64,
    start_us: u64,
    tempo_us_per_beat: u32,
}

impl Voice {
    fn all() -> [Self; 4] {
        [Self::P1, Self::P2, Self::Tr, Self::No]
    }

    fn index(self) -> usize {
        match self {
            Self::P1 => 0,
            Self::P2 => 1,
            Self::Tr => 2,
            Self::No => 3,
        }
    }

    fn channel(self) -> u8 {
        match self {
            Self::P1 => 0,
            Self::P2 => 1,
            Self::Tr => 2,
            Self::No => 3,
        }
    }

    fn track_name(self) -> &'static [u8] {
        match self {
            Self::P1 => b"P1",
            Self::P2 => b"P2",
            Self::Tr => b"TR",
            Self::No => b"NO",
        }
    }

    fn token_prefix(self) -> &'static str {
        match self {
            Self::P1 => "P1",
            Self::P2 => "P2",
            Self::Tr => "TR",
            Self::No => "NO",
        }
    }

    fn supports_expression(self) -> bool {
        !matches!(self, Self::Tr)
    }

    fn from_channel(channel: u8) -> Option<Self> {
        match channel {
            0 => Some(Self::P1),
            1 => Some(Self::P2),
            2 => Some(Self::Tr),
            3 | 9 => Some(Self::No),
            _ => None,
        }
    }

    fn from_track_name(name: &str) -> Option<Self> {
        let normalized = name
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        if normalized.contains("pulse1") || normalized.contains("square1") || normalized == "p1" {
            Some(Self::P1)
        } else if normalized.contains("pulse2")
            || normalized.contains("square2")
            || normalized == "p2"
        {
            Some(Self::P2)
        } else if normalized.contains("triangle") || normalized == "tr" {
            Some(Self::Tr)
        } else if normalized.contains("noise") || normalized == "no" {
            Some(Self::No)
        } else {
            None
        }
    }
}

impl MidiTokenizer {
    pub fn new() -> Self {
        let mut tokens = vec![BOS_TOKEN.to_string(), EOT_TOKEN.to_string()];
        let bos_token = 0;
        let eot_token = 1;
        let wait_start = tokens.len() as u32;
        for wait in 1..=WAIT_CHUNK_MAX {
            tokens.push(format!("WAIT_{wait}"));
        }

        let mut voice_offsets = [VoiceTokenOffsets {
            note_on_start: 0,
            note_off_start: 0,
            velocity_start: None,
            timbre_start: None,
        }; 4];

        for voice in Voice::all() {
            let note_on_start = tokens.len() as u32;
            for note in 0..=127 {
                tokens.push(format!("{}_NOTE_ON_{note}", voice.token_prefix()));
            }

            let note_off_start = tokens.len() as u32;
            for note in 0..=127 {
                tokens.push(format!("{}_NOTE_OFF_{note}", voice.token_prefix()));
            }

            let velocity_start = if voice.supports_expression() {
                let start = tokens.len() as u32;
                for value in 0..=127 {
                    tokens.push(format!("{}_VELOCITY_{value}", voice.token_prefix()));
                }
                Some(start)
            } else {
                None
            };

            let timbre_start = if voice.supports_expression() {
                let start = tokens.len() as u32;
                for value in 0..=127 {
                    tokens.push(format!("{}_TIMBRE_{value}", voice.token_prefix()));
                }
                Some(start)
            } else {
                None
            };

            voice_offsets[voice.index()] = VoiceTokenOffsets {
                note_on_start,
                note_off_start,
                velocity_start,
                timbre_start,
            };
        }

        Self {
            tokens,
            bos_token,
            eot_token,
            wait_start,
            voice_offsets,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn bos_token(&self) -> u32 {
        self.bos_token
    }

    pub fn eot_token(&self) -> u32 {
        self.eot_token
    }

    pub fn token_name(&self, token: u32) -> &str {
        self.tokens
            .get(token as usize)
            .map(String::as_str)
            .unwrap_or("<invalid>")
    }

    pub fn describe_tokens(&self, tokens: &[u32], limit: usize) -> String {
        let shown = tokens
            .iter()
            .take(limit)
            .map(|token| self.token_name(*token))
            .collect::<Vec<_>>()
            .join(" ");
        if tokens.len() > limit {
            format!("{shown} ...")
        } else {
            shown
        }
    }

    pub fn push_wait_tokens(&self, out: &mut Vec<u32>, mut steps: u32) {
        while steps > 0 {
            let chunk = steps.min(WAIT_CHUNK_MAX);
            out.push(self.wait_token(chunk));
            steps -= chunk;
        }
    }

    fn wait_token(&self, steps: u32) -> u32 {
        assert!((1..=WAIT_CHUNK_MAX).contains(&steps));
        self.wait_start + steps - 1
    }

    fn note_on_token(&self, voice: Voice, note: u8) -> u32 {
        self.voice_offsets[voice.index()].note_on_start + note as u32
    }

    fn note_off_token(&self, voice: Voice, note: u8) -> u32 {
        self.voice_offsets[voice.index()].note_off_start + note as u32
    }

    fn velocity_token(&self, voice: Voice, value: u8) -> u32 {
        self.voice_offsets[voice.index()]
            .velocity_start
            .expect("voice does not support velocity")
            + value as u32
    }

    fn timbre_token(&self, voice: Voice, value: u8) -> u32 {
        self.voice_offsets[voice.index()]
            .timbre_start
            .expect("voice does not support timbre")
            + value as u32
    }

    fn render_event_token(&self, kind: &RenderEventKind) -> u32 {
        match *kind {
            RenderEventKind::NoteOn { voice, note } => self.note_on_token(voice, note),
            RenderEventKind::NoteOff { voice, note } => self.note_off_token(voice, note),
            RenderEventKind::Velocity { voice, value } => self.velocity_token(voice, value),
            RenderEventKind::Timbre { voice, value } => self.timbre_token(voice, value),
        }
    }

    fn decode_token_kind(&self, token: u32) -> TokenKind {
        if token == self.bos_token {
            return TokenKind::Bos;
        }
        if token == self.eot_token {
            return TokenKind::Eot;
        }
        let wait_end = self.wait_start + WAIT_CHUNK_MAX;
        if (self.wait_start..wait_end).contains(&token) {
            return TokenKind::Wait(token - self.wait_start + 1);
        }

        for voice in Voice::all() {
            let offsets = self.voice_offsets[voice.index()];
            if token >= offsets.note_on_start && token < offsets.note_on_start + 128 {
                return TokenKind::NoteOn(voice, (token - offsets.note_on_start) as u8);
            }
            if token >= offsets.note_off_start && token < offsets.note_off_start + 128 {
                return TokenKind::NoteOff(voice, (token - offsets.note_off_start) as u8);
            }
            if let Some(velocity_start) = offsets.velocity_start {
                if token >= velocity_start && token < velocity_start + 128 {
                    return TokenKind::Velocity(voice, (token - velocity_start) as u8);
                }
            }
            if let Some(timbre_start) = offsets.timbre_start {
                if token >= timbre_start && token < timbre_start + 128 {
                    return TokenKind::Timbre(voice, (token - timbre_start) as u8);
                }
            }
        }

        panic!("unknown MIDI token id {token}");
    }

    pub fn gguf_metadata(&self) -> Vec<(String, GgufValue)> {
        let token_types: Box<[GgufValue]> = self
            .tokens
            .iter()
            .enumerate()
            .map(|(index, _)| {
                if index as u32 == self.bos_token || index as u32 == self.eot_token {
                    GgufValue::U32(3)
                } else {
                    GgufValue::U32(1)
                }
            })
            .collect();

        vec![
            (
                "tokenizer.ggml.model".to_string(),
                GgufValue::String("word".into()),
            ),
            (
                "tokenizer.ggml.pre".to_string(),
                GgufValue::String("default".into()),
            ),
            (
                "tokenizer.ggml.add_bos_token".to_string(),
                GgufValue::Bool(false),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                GgufValue::Array(
                    self.tokens
                        .iter()
                        .cloned()
                        .map(|token| GgufValue::String(token.into_boxed_str()))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.token_type".to_string(),
                GgufValue::Array(token_types),
            ),
            (
                "tokenizer.ggml.bos_token_id".to_string(),
                GgufValue::U32(self.bos_token),
            ),
            (
                "tokenizer.ggml.eos_token_id".to_string(),
                GgufValue::U32(self.eot_token),
            ),
        ]
    }

    pub fn from_var_builder(vb: &VarBuilder) -> fusor::Result<Self> {
        let tokenizer = Self::new();
        let tokens = vb
            .get_metadata("tokenizer.ggml.tokens")
            .ok_or_else(|| fusor::Error::msg("tokenizer.ggml.tokens metadata missing"))?
            .to_array()
            .map_err(|error| fusor::Error::msg(error.to_string()))?;
        let loaded_tokens = tokens
            .iter()
            .map(|token| {
                token
                    .to_string()
                    .map(String::from)
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(fusor::Error::msg)?;
        if loaded_tokens != tokenizer.tokens {
            return Err(fusor::Error::msg(
                "GGUF tokenizer vocabulary does not match the expected fixed MIDI vocabulary",
            ));
        }

        let bos_token = vb
            .get_metadata("tokenizer.ggml.bos_token_id")
            .ok_or_else(|| fusor::Error::msg("tokenizer.ggml.bos_token_id metadata missing"))?
            .to_u32()
            .map_err(|error| fusor::Error::msg(error.to_string()))?;
        let eot_token = vb
            .get_metadata("tokenizer.ggml.eos_token_id")
            .ok_or_else(|| fusor::Error::msg("tokenizer.ggml.eos_token_id metadata missing"))?
            .to_u32()
            .map_err(|error| fusor::Error::msg(error.to_string()))?;
        if bos_token != tokenizer.bos_token || eot_token != tokenizer.eot_token {
            return Err(fusor::Error::msg(
                "GGUF tokenizer special token ids do not match the expected MIDI tokenizer ids",
            ));
        }

        Ok(tokenizer)
    }
}

impl SourceDataset {
    pub fn num_docs(&self) -> usize {
        self.files.len()
    }

    pub fn num_tokens(&self) -> usize {
        self.files.iter().map(|file| file.tokens.len()).sum()
    }

    pub fn num_training_windows(&self, block_size: usize) -> usize {
        self.files
            .iter()
            .map(|file| file.training_window_count(block_size))
            .sum()
    }

    pub fn max_tokens_per_example(&self) -> usize {
        self.files
            .iter()
            .map(|file| file.tokens.len())
            .max()
            .unwrap_or(0)
    }

    pub fn files(&self) -> &[SourceFile] {
        &self.files
    }

    pub fn sample_batch(&self, rng: &mut StdRng, pad_token: u32, config: &RuntimeConfig) -> Batch {
        let (file_offsets, total_training_windows) = self.window_offsets(config.block_size);
        let sampled = (0..config.batch_size)
            .map(|_| {
                let global_position = rng.random_range(0..total_training_windows.max(1));
                self.sample_window_at(global_position, &file_offsets, pad_token, config.block_size)
            })
            .collect::<Vec<_>>();

        Batch {
            windows: sampled
                .iter()
                .map(|(window, _, _)| window.clone())
                .collect(),
            masks: sampled.iter().map(|(_, mask, _)| mask.clone()).collect(),
            valid_tokens: sampled.iter().map(|(_, _, valid)| *valid).sum(),
        }
    }

    pub fn evaluation_batches(&self, pad_token: u32, config: &RuntimeConfig) -> Vec<Batch> {
        let (file_offsets, total_training_windows) = self.window_offsets(config.block_size);
        if total_training_windows == 0 {
            return Vec::new();
        }

        let steps = config
            .eval_batches
            .saturating_mul(config.batch_size)
            .min(total_training_windows);
        let mut batches = Vec::new();
        let mut current_windows = Vec::new();
        let mut current_masks = Vec::new();
        let mut current_valid_tokens = 0.0;

        for global_position in 0..steps {
            let (window, mask, valid_tokens) =
                self.sample_window_at(global_position, &file_offsets, pad_token, config.block_size);
            current_windows.push(window);
            current_masks.push(mask);
            current_valid_tokens += valid_tokens;

            if current_windows.len() == config.batch_size {
                batches.push(Batch {
                    windows: std::mem::take(&mut current_windows),
                    masks: std::mem::take(&mut current_masks),
                    valid_tokens: current_valid_tokens,
                });
                current_valid_tokens = 0.0;
            }
        }

        if !current_windows.is_empty() {
            batches.push(Batch {
                windows: current_windows,
                masks: current_masks,
                valid_tokens: current_valid_tokens,
            });
        }

        batches
    }

    fn sample_window_at(
        &self,
        global_position: usize,
        file_offsets: &[usize],
        pad_token: u32,
        block_size: usize,
    ) -> (Vec<u32>, Vec<f32>, f32) {
        let file_index = file_offsets
            .partition_point(|offset| *offset <= global_position)
            .saturating_sub(1);
        let file = &self.files[file_index];
        let start =
            file.training_window_start(block_size) + (global_position - file_offsets[file_index]);
        let end = (start + block_size + 1).min(file.tokens.len());
        let slice = &file.tokens[start..end];

        let mut window = vec![pad_token; block_size + 1];
        let mut mask = vec![0.0; block_size];
        window[..slice.len()].copy_from_slice(slice);

        let mut valid = 0.0;
        for local_index in 0..slice.len().saturating_sub(1) {
            mask[local_index] = 1.0;
            valid += 1.0;
        }

        (window, mask, valid)
    }

    fn window_offsets(&self, block_size: usize) -> (Vec<usize>, usize) {
        let mut running_total = 0;
        let mut file_offsets = Vec::with_capacity(self.files.len());
        for file in &self.files {
            file_offsets.push(running_total);
            running_total += file.training_window_count(block_size);
        }
        (file_offsets, running_total)
    }
}

impl SourceFile {
    fn training_window_start(&self, block_size: usize) -> usize {
        self.target_start
            .saturating_sub(block_size.saturating_sub(1))
    }

    fn training_window_count(&self, block_size: usize) -> usize {
        self.tokens
            .len()
            .saturating_sub(1)
            .saturating_sub(self.training_window_start(block_size))
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn target_tokens(&self) -> &[u32] {
        &self.target_tokens
    }

    pub fn completion_prompt_tokens(
        &self,
        prompt_tokens: usize,
        block_size: usize,
        sample_tokens: usize,
    ) -> &[u32] {
        let max_prompt_tokens = prompt_tokens.min(block_size.saturating_sub(sample_tokens + 1));
        &self.prompt_tokens[..self.prompt_tokens.len().min(max_prompt_tokens)]
    }
}

pub struct Batch {
    pub windows: Vec<Vec<u32>>,
    pub masks: Vec<Vec<f32>>,
    pub valid_tokens: f32,
}

pub async fn bootstrap_dataset(cache_dir: &Path) -> PathBuf {
    fs::create_dir_all(cache_dir).unwrap_or_else(|error| {
        panic!(
            "failed to create dataset cache dir {}: {error}",
            cache_dir.display()
        )
    });
    if let Some(root) = find_dataset_root(cache_dir) {
        return root;
    }

    let archive_path = cache_dir.join(NESMDB_ARCHIVE_NAME);
    if !archive_path.exists() {
        download_dataset_archive(&archive_path).await;
    }

    extract_archive(&archive_path, cache_dir);
    find_dataset_root(cache_dir).unwrap_or_else(|| {
        panic!(
            "downloaded NES-MDB archive but could not find train/valid/test directories under {}",
            cache_dir.display()
        )
    })
}

pub fn load_dataset_split(root: &Path, tokenizer: &MidiTokenizer) -> DatasetSplit {
    DatasetSplit {
        train: load_split(root, "train", tokenizer),
        validation: load_split(root, "valid", tokenizer),
        test: load_split(root, "test", tokenizer),
    }
}

pub fn write_tokens_to_midi_file(tokenizer: &MidiTokenizer, tokens: &[u32], path: &Path) {
    let smf = tokens_to_smf(tokenizer, tokens);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|error| {
            panic!(
                "failed to create MIDI output dir {}: {error}",
                parent.display()
            )
        });
    }
    let mut bytes = Vec::new();
    smf.write_std(&mut bytes).unwrap_or_else(|error| {
        panic!(
            "failed to serialize MIDI sample {}: {error}",
            path.display()
        )
    });
    fs::write(path, bytes)
        .unwrap_or_else(|error| panic!("failed to write MIDI sample {}: {error}", path.display()));
}

pub fn windows_to_token_inputs(windows: &[Vec<u32>]) -> Vec<Vec<u32>> {
    windows
        .iter()
        .map(|window| window[..window.len().saturating_sub(1)].to_vec())
        .collect()
}

pub fn windows_to_token_targets(windows: &[Vec<u32>]) -> Vec<Vec<u32>> {
    windows.iter().map(|window| window[1..].to_vec()).collect()
}

pub fn autoregressive_context(
    tokens: &[u32],
    pad_token: u32,
    block_size: usize,
) -> (Vec<u32>, usize) {
    let mut context = vec![pad_token; block_size];
    let slice = if tokens.len() > block_size {
        &tokens[tokens.len() - block_size..]
    } else {
        tokens
    };
    context[..slice.len()].copy_from_slice(slice);
    (context, slice.len().saturating_sub(1))
}

pub fn position_indexes(batch_size: usize, block_size: usize) -> Vec<Vec<u32>> {
    (0..batch_size)
        .map(|_| (0..block_size).map(|position| position as u32).collect())
        .collect()
}

async fn download_dataset_archive(path: &Path) {
    println!("downloading NES-MDB MIDI archive to {}...", path.display());
    let response = reqwest::get(NESMDB_DOWNLOAD_URL)
        .await
        .unwrap_or_else(|error| panic!("failed to download NES-MDB archive: {error}"));
    let status = response.status();
    if !status.is_success() {
        panic!("failed to download NES-MDB archive: HTTP {status}");
    }
    let bytes = response
        .bytes()
        .await
        .unwrap_or_else(|error| panic!("failed to read NES-MDB archive response body: {error}"));
    if bytes.starts_with(b"<!DOCTYPE html") || bytes.starts_with(b"<html") {
        let parent_display = path.parent().unwrap_or(Path::new(".")).display();
        panic!(
            "NES-MDB download returned HTML instead of a tar.gz archive; place {NESMDB_ARCHIVE_NAME} under {} manually or update the built-in URL",
            parent_display
        );
    }
    fs::write(path, &bytes).unwrap_or_else(|error| {
        panic!(
            "failed to write downloaded archive {}: {error}",
            path.display()
        )
    });
}

fn extract_archive(archive_path: &Path, cache_dir: &Path) {
    println!("extracting NES-MDB MIDI archive...");
    let archive = fs::File::open(archive_path).unwrap_or_else(|error| {
        panic!(
            "failed to open dataset archive {}: {error}",
            archive_path.display()
        )
    });
    let decoder = GzDecoder::new(archive);
    let mut archive = Archive::new(decoder);
    archive.unpack(cache_dir).unwrap_or_else(|error| {
        panic!(
            "failed to extract dataset archive {}: {error}",
            archive_path.display()
        )
    });
}

fn load_split(root: &Path, split_name: &str, tokenizer: &MidiTokenizer) -> SourceDataset {
    let split_root = root.join(split_name);
    let mut paths = Vec::new();
    collect_midi_files(&split_root, &mut paths);
    paths.sort();
    let files = paths
        .into_iter()
        .map(|path| {
            let bytes = fs::read(&path).unwrap_or_else(|error| {
                panic!("failed to read MIDI file {}: {error}", path.display())
            });
            let content_tokens = midi_bytes_to_tokens(tokenizer, &bytes, &path);
            let mut tokens = Vec::with_capacity(content_tokens.len() + 2);
            tokens.push(tokenizer.bos_token());
            tokens.extend_from_slice(&content_tokens);
            tokens.push(tokenizer.eot_token());
            let relative_path = path
                .strip_prefix(root)
                .unwrap_or(path.as_path())
                .to_string_lossy()
                .into_owned();

            SourceFile {
                path: relative_path,
                tokens,
                prompt_tokens: content_tokens.clone(),
                target_tokens: content_tokens,
                target_start: 0,
            }
        })
        .collect();
    SourceDataset { files }
}

fn midi_bytes_to_tokens(tokenizer: &MidiTokenizer, bytes: &[u8], path: &Path) -> Vec<u32> {
    let smf = Smf::parse(bytes)
        .unwrap_or_else(|error| panic!("failed to parse MIDI file {}: {error}", path.display()));
    let ticks_per_beat = match smf.header.timing {
        Timing::Metrical(ticks_per_beat) => ticks_per_beat.as_int() as u16,
        _ => panic!(
            "unsupported MIDI timing in {}; NES-MDB examples are expected to use metrical timing",
            path.display()
        ),
    };
    let tempo_segments = collect_tempo_segments(&smf, ticks_per_beat);

    let mut events = Vec::new();
    let mut sequence = 0usize;
    for track in &smf.tracks {
        let track_voice = detect_track_voice(track);
        let mut absolute_tick = 0u64;
        for event in track {
            absolute_tick += event.delta.as_int() as u64;
            let TrackEventKind::Midi { channel, message } = event.kind else {
                continue;
            };
            let Some(voice) = track_voice.or_else(|| Voice::from_channel(channel.as_int())) else {
                continue;
            };
            let Some(kind) = midi_message_to_render_event(voice, message) else {
                continue;
            };
            let step = tick_to_step(absolute_tick, ticks_per_beat, &tempo_segments);
            events.push(MidiEvent {
                step,
                sequence,
                kind,
            });
            sequence += 1;
        }
    }

    events.sort_by_key(|event| (event.step, event.sequence));
    let mut tokens = Vec::new();
    let mut current_step = 0u32;
    for event in events {
        if event.step > current_step {
            tokenizer.push_wait_tokens(&mut tokens, event.step - current_step);
            current_step = event.step;
        }
        tokens.push(tokenizer.render_event_token(&event.kind));
    }
    tokens
}

fn tokens_to_smf(tokenizer: &MidiTokenizer, tokens: &[u32]) -> Smf<'static> {
    let mut current_step = 0u32;
    let mut events = [
        Vec::<(u32, TrackEventKind<'static>)>::new(),
        Vec::<(u32, TrackEventKind<'static>)>::new(),
        Vec::<(u32, TrackEventKind<'static>)>::new(),
        Vec::<(u32, TrackEventKind<'static>)>::new(),
    ];

    for voice in Voice::all() {
        let track_events = &mut events[voice.index()];
        track_events.push((
            0,
            TrackEventKind::Meta(MetaMessage::TrackName(voice.track_name())),
        ));
    }
    events[0].push((
        0,
        TrackEventKind::Meta(MetaMessage::Tempo(u24::new(MIDI_TEMPO_US_PER_BEAT))),
    ));

    for token in tokens {
        match tokenizer.decode_token_kind(*token) {
            TokenKind::Bos | TokenKind::Eot => {}
            TokenKind::Wait(steps) => current_step += steps,
            TokenKind::NoteOn(voice, note) => events[voice.index()].push((
                current_step,
                TrackEventKind::Midi {
                    channel: u4::new(voice.channel()),
                    message: MidiMessage::NoteOn {
                        key: u7::new(note),
                        vel: u7::new(default_velocity(voice)),
                    },
                },
            )),
            TokenKind::NoteOff(voice, note) => events[voice.index()].push((
                current_step,
                TrackEventKind::Midi {
                    channel: u4::new(voice.channel()),
                    message: MidiMessage::NoteOff {
                        key: u7::new(note),
                        vel: u7::new(0),
                    },
                },
            )),
            TokenKind::Velocity(voice, value) => events[voice.index()].push((
                current_step,
                TrackEventKind::Midi {
                    channel: u4::new(voice.channel()),
                    message: MidiMessage::Controller {
                        controller: u7::new(VELOCITY_CONTROLLER),
                        value: u7::new(value),
                    },
                },
            )),
            TokenKind::Timbre(voice, value) => events[voice.index()].push((
                current_step,
                TrackEventKind::Midi {
                    channel: u4::new(voice.channel()),
                    message: MidiMessage::Controller {
                        controller: u7::new(TIMBRE_CONTROLLER),
                        value: u7::new(value),
                    },
                },
            )),
        }
    }

    let tracks = events
        .into_iter()
        .map(encode_track_events)
        .collect::<Vec<_>>();
    Smf {
        header: Header::new(
            Format::Parallel,
            Timing::Metrical(u15::new(MIDI_TICKS_PER_BEAT)),
        ),
        tracks,
    }
}

fn detect_track_voice(track: &[TrackEvent<'_>]) -> Option<Voice> {
    for event in track {
        match &event.kind {
            TrackEventKind::Meta(MetaMessage::TrackName(name))
            | TrackEventKind::Meta(MetaMessage::InstrumentName(name)) => {
                if let Ok(name) = std::str::from_utf8(name) {
                    if let Some(voice) = Voice::from_track_name(name) {
                        return Some(voice);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn midi_message_to_render_event(voice: Voice, message: MidiMessage) -> Option<RenderEventKind> {
    match message {
        MidiMessage::NoteOn { key, vel } if vel.as_int() == 0 => Some(RenderEventKind::NoteOff {
            voice,
            note: key.as_int(),
        }),
        MidiMessage::NoteOn { key, .. } => Some(RenderEventKind::NoteOn {
            voice,
            note: key.as_int(),
        }),
        MidiMessage::NoteOff { key, .. } => Some(RenderEventKind::NoteOff {
            voice,
            note: key.as_int(),
        }),
        MidiMessage::Controller { controller, value }
            if controller.as_int() == VELOCITY_CONTROLLER && voice.supports_expression() =>
        {
            Some(RenderEventKind::Velocity {
                voice,
                value: value.as_int(),
            })
        }
        MidiMessage::Controller { controller, value }
            if controller.as_int() == TIMBRE_CONTROLLER && voice.supports_expression() =>
        {
            Some(RenderEventKind::Timbre {
                voice,
                value: value.as_int(),
            })
        }
        _ => None,
    }
}

fn collect_tempo_segments(smf: &Smf<'_>, ticks_per_beat: u16) -> Vec<TempoSegment> {
    let mut changes = vec![(0u64, DEFAULT_TEMPO_US_PER_BEAT)];
    for track in &smf.tracks {
        let mut absolute_tick = 0u64;
        for event in track {
            absolute_tick += event.delta.as_int() as u64;
            if let TrackEventKind::Meta(MetaMessage::Tempo(tempo)) = event.kind {
                changes.push((absolute_tick, tempo.as_int()));
            }
        }
    }
    changes.sort_by_key(|(tick, _)| *tick);

    let mut deduped: Vec<(u64, u32)> = Vec::new();
    for (tick, tempo) in changes {
        if let Some(last) = deduped.last_mut() {
            if last.0 == tick {
                last.1 = tempo;
                continue;
            }
        }
        deduped.push((tick, tempo));
    }

    let mut segments = Vec::with_capacity(deduped.len());
    let mut start_us = 0u64;
    let mut previous_tick = deduped[0].0;
    let mut previous_tempo = deduped[0].1;
    segments.push(TempoSegment {
        start_tick: previous_tick,
        start_us,
        tempo_us_per_beat: previous_tempo,
    });
    for (tick, tempo) in deduped.into_iter().skip(1) {
        start_us += ticks_to_us(tick - previous_tick, ticks_per_beat, previous_tempo);
        segments.push(TempoSegment {
            start_tick: tick,
            start_us,
            tempo_us_per_beat: tempo,
        });
        previous_tick = tick;
        previous_tempo = tempo;
    }
    segments
}

fn tick_to_step(tick: u64, ticks_per_beat: u16, tempo_segments: &[TempoSegment]) -> u32 {
    let mut active_segment = tempo_segments
        .first()
        .copied()
        .expect("tempo map must contain at least one segment");
    for segment in tempo_segments.iter().copied().skip(1) {
        if segment.start_tick > tick {
            break;
        }
        active_segment = segment;
    }

    let micros = active_segment.start_us
        + ticks_to_us(
            tick.saturating_sub(active_segment.start_tick),
            ticks_per_beat,
            active_segment.tempo_us_per_beat,
        );
    ((micros + (QUANTIZATION_US_PER_STEP / 2)) / QUANTIZATION_US_PER_STEP) as u32
}

fn ticks_to_us(delta_ticks: u64, ticks_per_beat: u16, tempo_us_per_beat: u32) -> u64 {
    delta_ticks * tempo_us_per_beat as u64 / ticks_per_beat.max(1) as u64
}

fn encode_track_events(
    mut events: Vec<(u32, TrackEventKind<'static>)>,
) -> Vec<TrackEvent<'static>> {
    events.sort_by_key(|(step, _)| *step);
    let mut current_step = 0u32;
    let mut track = Vec::with_capacity(events.len() + 1);
    for (step, kind) in events {
        let delta = step.saturating_sub(current_step);
        track.push(TrackEvent {
            delta: u28::new(delta),
            kind,
        });
        current_step = step;
    }
    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });
    track
}

fn default_velocity(voice: Voice) -> u8 {
    match voice {
        Voice::Tr => 96,
        _ => 100,
    }
}

fn find_dataset_root(base: &Path) -> Option<PathBuf> {
    let mut queue = VecDeque::from([base.to_path_buf()]);
    while let Some(path) = queue.pop_front() {
        if path.join("train").is_dir() && path.join("valid").is_dir() && path.join("test").is_dir()
        {
            return Some(path);
        }

        let Ok(entries) = fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() {
                queue.push_back(child);
            }
        }
    }
    None
}

fn collect_midi_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|error| {
        panic!(
            "failed to read dataset split directory {}: {error}",
            dir.display()
        )
    });
    for entry in entries {
        let entry = entry.unwrap();
        let path = entry.path();
        if entry.file_type().unwrap().is_dir() {
            collect_midi_files(&path, out);
            continue;
        }

        let extension = path.extension().and_then(|extension| extension.to_str());
        if matches!(
            extension,
            Some("mid") | Some("midi") | Some("MID") | Some("MIDI")
        ) {
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor_gguf::{GgufMetadata, GgufVersion};
    use std::{
        io::Cursor,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn synthetic_tokens(tokenizer: &MidiTokenizer) -> Vec<u32> {
        vec![
            tokenizer.velocity_token(Voice::P1, 90),
            tokenizer.note_on_token(Voice::P1, 60),
            tokenizer.wait_token(3),
            tokenizer.note_off_token(Voice::P1, 60),
            tokenizer.timbre_token(Voice::No, 12),
            tokenizer.note_on_token(Voice::No, 36),
            tokenizer.wait_token(2),
            tokenizer.note_off_token(Voice::No, 36),
        ]
    }

    #[test]
    fn midi_tokens_round_trip_through_smf() {
        let tokenizer = MidiTokenizer::new();
        let tokens = synthetic_tokens(&tokenizer);
        let smf = tokens_to_smf(&tokenizer, &tokens);
        let mut bytes = Vec::new();
        smf.write_std(&mut bytes).unwrap();
        let reparsed = midi_bytes_to_tokens(&tokenizer, &bytes, Path::new("synthetic.mid"));
        assert_eq!(reparsed, tokens);
        Smf::parse(&bytes).unwrap();
    }

    #[test]
    fn wait_tokens_round_trip_timing() {
        let tokenizer = MidiTokenizer::new();
        let tokens = vec![
            tokenizer.wait_token(100),
            tokenizer.wait_token(17),
            tokenizer.note_on_token(Voice::P2, 64),
            tokenizer.wait_token(5),
            tokenizer.note_off_token(Voice::P2, 64),
        ];
        let smf = tokens_to_smf(&tokenizer, &tokens);
        let mut bytes = Vec::new();
        smf.write_std(&mut bytes).unwrap();
        let reparsed = midi_bytes_to_tokens(&tokenizer, &bytes, Path::new("timing.mid"));
        assert_eq!(reparsed, tokens);
    }

    #[test]
    fn dataset_split_loader_discovers_train_valid_test() {
        let tokenizer = MidiTokenizer::new();
        let root = temp_test_dir("dataset");
        fs::create_dir_all(root.join("train")).unwrap();
        fs::create_dir_all(root.join("valid/nested")).unwrap();
        fs::create_dir_all(root.join("test")).unwrap();

        let smf = tokens_to_smf(&tokenizer, &synthetic_tokens(&tokenizer));
        let mut bytes = Vec::new();
        smf.write_std(&mut bytes).unwrap();
        fs::write(root.join("train/song_a.mid"), &bytes).unwrap();
        fs::write(root.join("valid/nested/song_b.mid"), &bytes).unwrap();
        fs::write(root.join("test/song_c.mid"), &bytes).unwrap();

        let split = load_dataset_split(&root, &tokenizer);
        assert_eq!(split.train.num_docs(), 1);
        assert_eq!(split.validation.num_docs(), 1);
        assert_eq!(split.test.num_docs(), 1);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn tokenizer_survives_gguf_metadata_round_trip() {
        let tokenizer = MidiTokenizer::new();
        let gguf = GgufMetadata {
            version: GgufVersion::V3,
            metadata: tokenizer
                .gguf_metadata()
                .into_iter()
                .map(|(key, value)| (key.into_boxed_str(), value))
                .collect(),
            tensor_infos: Default::default(),
            tensor_data_offset: 0,
        };
        let mut bytes = Vec::new();
        gguf.write(&mut bytes, std::iter::empty::<(&str, &[u8])>())
            .unwrap();
        let mut reader = Cursor::new(bytes);
        let vb = VarBuilder::from_gguf(&mut reader).unwrap();
        let restored = MidiTokenizer::from_var_builder(&vb).unwrap();
        assert_eq!(restored.vocab_size(), tokenizer.vocab_size());
        assert_eq!(restored.bos_token(), tokenizer.bos_token());
        assert_eq!(restored.eot_token(), tokenizer.eot_token());
    }

    #[test]
    fn sample_export_produces_valid_midi_bytes() {
        let tokenizer = MidiTokenizer::new();
        let tokens = synthetic_tokens(&tokenizer);
        let root = temp_test_dir("sample");
        let output = root.join("sample.mid");
        write_tokens_to_midi_file(&tokenizer, &tokens, &output);
        let bytes = fs::read(&output).unwrap();
        Smf::parse(&bytes).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    fn temp_test_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("nanochat-{label}-{nanos}"))
    }
}
