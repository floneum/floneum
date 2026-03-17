use std::{env, fmt::Display, path::PathBuf, str::FromStr};

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub train_steps: usize,
    pub warmup_steps: usize,
    pub learning_rate: f32,
    pub min_learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub adam_eps: f32,
    pub weight_decay: f32,
    pub log_every: usize,
    pub eval_batches: usize,
    pub sample_tokens: usize,
    pub sample_prefix_tokens: usize,
    pub sample_temperature: f32,
    pub sample_top_k: usize,
    pub block_size: usize,
    pub batch_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_ff: usize,
    pub n_layer: usize,
    pub conv_kernel_size: usize,
    pub attention_period: usize,
    pub eps: f32,
    pub init_scale: f32,
    pub seed: u64,
    pub save_every_steps: usize,
    pub save_final_model: bool,
    pub save_quantization: SaveQuantization,
    pub dataset_cache_dir: PathBuf,
    pub gguf_path: PathBuf,
    pub sample_output_path: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SaveQuantization {
    F32,
    F16,
    Q4_0,
    Q8_0,
}

impl SaveQuantization {
    pub fn as_str(self) -> &'static str {
        match self {
            SaveQuantization::F32 => "f32",
            SaveQuantization::F16 => "f16",
            SaveQuantization::Q4_0 => "q4_0",
            SaveQuantization::Q8_0 => "q8_0",
        }
    }
}

impl FromStr for SaveQuantization {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "q4_0" | "q40" | "q4" => Ok(Self::Q4_0),
            "q8_0" | "q80" | "q8" => Ok(Self::Q8_0),
            other => Err(format!(
                "unsupported save quantization {other:?}; expected one of: f32, f16, q4_0, q8_0"
            )),
        }
    }
}

impl RuntimeConfig {
    pub fn load() -> Self {
        let env_path = env_path();
        dotenvy::from_path(&env_path).unwrap_or_else(|error| {
            panic!(
                "failed to load nanochat config from {}: {error}",
                env_path.display()
            )
        });

        let batch_size = read_env("NANOCHAT_BATCH_SIZE", 12);

        Self {
            train_steps: read_env("NANOCHAT_TRAIN_STEPS", 540),
            warmup_steps: read_env("NANOCHAT_WARMUP_STEPS", 20),
            learning_rate: read_env("NANOCHAT_LEARNING_RATE", 1e-3),
            min_learning_rate: read_env("NANOCHAT_MIN_LEARNING_RATE", 1e-4),
            beta1: read_env("NANOCHAT_BETA1", 0.9),
            beta2: read_env("NANOCHAT_BETA2", 0.95),
            adam_eps: read_env("NANOCHAT_ADAM_EPS", 1e-8),
            weight_decay: read_env("NANOCHAT_WEIGHT_DECAY", 0.1),
            log_every: read_env("NANOCHAT_LOG_EVERY", 80),
            eval_batches: read_env("NANOCHAT_EVAL_BATCHES", 2),
            sample_tokens: read_env("NANOCHAT_SAMPLE_TOKENS", 256),
            sample_prefix_tokens: read_env("NANOCHAT_SAMPLE_PREFIX_TOKENS", 128),
            sample_temperature: read_env("NANOCHAT_SAMPLE_TEMPERATURE", 0.7),
            sample_top_k: read_env("NANOCHAT_SAMPLE_TOP_K", 8),
            block_size: read_env("NANOCHAT_BLOCK_SIZE", 32),
            batch_size,
            n_embd: read_env("NANOCHAT_N_EMBD", 64),
            n_head: read_env("NANOCHAT_N_HEAD", 4),
            n_ff: read_env("NANOCHAT_N_FF", 256),
            n_layer: read_env("NANOCHAT_N_LAYER", 4),
            conv_kernel_size: read_env("NANOCHAT_CONV_KERNEL_SIZE", 3),
            attention_period: read_env("NANOCHAT_ATTENTION_PERIOD", 1),
            eps: read_env("NANOCHAT_EPS", 1e-5),
            init_scale: read_env("NANOCHAT_INIT_SCALE", 0.08),
            seed: read_env("NANOCHAT_SEED", 1337),
            save_every_steps: read_env("NANOCHAT_SAVE_EVERY_STEPS", 0),
            save_final_model: read_env("NANOCHAT_SAVE_FINAL_MODEL", true),
            save_quantization: read_env("NANOCHAT_SAVE_QUANTIZATION", SaveQuantization::F32),
            dataset_cache_dir: read_env_path_with_default(
                "NANOCHAT_DATASET_CACHE_DIR",
                std::env::temp_dir().join("nanochat-midi"),
            ),
            gguf_path: read_env_path("NANOCHAT_GGUF_PATH", "nanochat.gguf"),
            sample_output_path: read_env_path("NANOCHAT_SAMPLE_OUTPUT_PATH", "nanochat-sample.mid"),
        }
    }
}

fn env_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".env")
}

fn read_env<T>(key: &str, default: T) -> T
where
    T: FromStr + Copy,
    T::Err: Display,
{
    match env::var(key) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|error| panic!("invalid value for {key}: {value:?} ({error})")),
        Err(env::VarError::NotPresent) => default,
        Err(error) => panic!("failed to read {key}: {error}"),
    }
}

fn read_env_path(key: &str, default: &str) -> PathBuf {
    match env::var(key) {
        Ok(value) => PathBuf::from(value),
        Err(env::VarError::NotPresent) => PathBuf::from(default),
        Err(error) => panic!("failed to read {key}: {error}"),
    }
}

fn read_env_path_with_default(key: &str, default: PathBuf) -> PathBuf {
    match env::var(key) {
        Ok(value) => PathBuf::from(value),
        Err(env::VarError::NotPresent) => default,
        Err(error) => panic!("failed to read {key}: {error}"),
    }
}
