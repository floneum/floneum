pub const CHARSET: &str = "\n !,.:?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
pub const BOS_TOKEN: u32 = CHARSET.len() as u32;
pub const EOT_TOKEN: u32 = CHARSET.len() as u32 + 1;
pub const VOCAB_SIZE: usize = CHARSET.len() + 2;

pub const BLOCK_SIZE: usize = 224;
pub const BATCH_SIZE: usize = 8;
pub const N_EMBD: usize = 64;
pub const N_FF: usize = 128;
pub const N_LAYER: usize = 4;

pub const TRAIN_STEPS: usize = 2200;
pub const LEARNING_RATE: f32 = 0.03;
pub const EPS: f32 = 1e-5;
pub const LOG_EVERY: usize = 100;

pub const SAMPLE_TOKENS: usize = 80;

pub const SYSTEM_PROMPT: &str =
    "You are nanochat, a tiny helpful assistant trained with fusor.";
