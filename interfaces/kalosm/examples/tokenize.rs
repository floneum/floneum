use candle_core::IndexOp;
use candle_core::Tensor;
use candle_core::{DType, Device};
use candle_datasets::Batcher;
use candle_nn::Optimizer;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama2_c::Cache;
use candle_transformers::models::llama2_c::{Config, Llama};
use std::io::Read;
use std::io::Write;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::Tokenizer;
use tokenizers::TokenizerBuilder;
use std::path::Path;

const LEARNING_RATE: f64 = 0.001;
const BATCH_SIZE: usize = 32;
const VOCAB_SIZE: usize = 1000;
const SEQ_LEN: usize = 512;
const N_LAYERS: usize = 5;
const N_HEADS: usize = 8;
const N_KV_HEADS: usize = 4;
const DIM: usize = 64;
const HIDDEN_DIM: usize = 128;

fn main() -> anyhow::Result<()> {
    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(VOCAB_SIZE)
        .min_frequency(2)
        .special_tokens(vec![])
        .build();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into(),
        ])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let pretty = false;
    let mut paths = Vec::new();
    // find all rust files in the current directory
    find_files(
        "/Users/evanalmloff/Desktop/Github/floneum/interfaces/kalosm/html",
        &mut paths,
    )?;
    tokenizer
        .train_from_files(&mut trainer, paths.clone())
        .map_err(|e| anyhow::anyhow!("{}", e))?
        .save("tokenizer.json", pretty)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut dataset = Vec::new();

    for path in paths {
        let mut file = std::fs::File::open(&path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let mut tokens = tokenizer
            .encode(&*contents, false)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let mut ids = Vec::new();
        ids.extend(tokens.get_ids().into_iter());
        dataset.push(ids);
    }

    run(dataset, tokenizer.into())?;

    Ok(())
}

fn find_files<P: AsRef<Path>>(path: P, paths: &mut Vec<String>) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().map(|s| s.to_str().unwrap()) == Some("html") {
            paths.push(path.canonicalize().unwrap().to_str().unwrap().to_owned());
        } else if path.is_dir() {
            find_files(path, paths)?;
        }
    }
    Ok(())
}

fn valid_loss(dataset: &Dataset, model: &Llama, device: &Device) -> anyhow::Result<f64> {
    let iter = DatasetRandomIter::new(dataset, true, model.config.seq_len, device.clone());
    let batch_iter = Batcher::new_r2(iter).batch_size(BATCH_SIZE);
    let mut sum_ce = 0f64;
    let mut cnt = 0usize;
    for inp_tgt in batch_iter.take(50) {
        let (inp, tgt) = inp_tgt?;
        let logits = model.forward(&inp, 0)?;
        let loss = candle_nn::loss::cross_entropy(&logits.flatten_to(1)?, &tgt.flatten_to(1)?)?;
        sum_ce += loss.to_vec0::<f32>()? as f64;
        cnt += 1;
    }
    Ok(sum_ce / cnt as f64)
}

pub fn run(dataset: Vec<Vec<u32>>, tokenizer: Tokenizer) -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;
    let dataset = Dataset::new(&dataset);
    println!(
        "loaded dataset, train: {} files, valid: {} files",
        dataset.train_tokens(),
        dataset.valid_tokens()
    );
    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let config = Config {
        dim: DIM,
        hidden_dim: HIDDEN_DIM,
        n_layers: N_LAYERS,
        n_heads: N_HEADS,
        n_kv_heads: N_KV_HEADS,
        vocab_size: VOCAB_SIZE,
        seq_len: SEQ_LEN,
        norm_eps: 1e-5,
    };
    let iter = DatasetRandomIter::new(&dataset, false, config.seq_len, device.clone());
    let batch_iter = candle_datasets::Batcher::new_r2(iter).batch_size(BATCH_SIZE);

    let cache = Cache::new(false, &config, vb.pp("rot"))?;
    let model = Llama::load(vb, &cache, config.clone())?;
    if Path::new("checkpoint.safetensors").exists() {
        varmap.load("checkpoint.safetensors")?;
    }

    let params = candle_nn::ParamsAdamW {
        lr: LEARNING_RATE,
        ..Default::default()
    };
    let mut opt = candle_nn::AdamW::new(varmap.all_vars(), params)?;
    for (batch_index, batch) in batch_iter.enumerate() {
        let (inp, tgt) = batch?;
        let logits = model.forward(&inp, 0)?;
        let loss = candle_nn::loss::cross_entropy(&logits.flatten_to(1)?, &tgt.flatten_to(1)?)?;
        opt.backward_step(&loss)?;
        println!("{batch_index} {loss}");

        if batch_index > 0 && batch_index % 100 == 0 {
            // TODO: Add a way to deactivate the backprop graph tracking when computing the
            // validation loss.
            let loss = valid_loss(&dataset, &model, &device)?;
            println!("validation: {batch_index} {loss}");
            run_inference(&tokenizer, &cache, &model, &config)?;
            varmap.save("checkpoint.safetensors")?;
        }
    }
    Ok(())
}

pub struct Dataset {
    valid_tokens: Vec<Vec<u32>>,
    train_tokens: Vec<Vec<u32>>,
}

impl Dataset {
    pub fn new(tokens: &[Vec<u32>]) -> Self {
        let mut valid_tokens = Vec::new();
        let mut train_tokens = Vec::new();
        for (i, tokens) in tokens.iter().enumerate() {
            if i % 10 == 0 {
                valid_tokens.push(tokens.clone());
            } else {
                train_tokens.push(tokens.clone());
            }
        }
        Self {
            valid_tokens,
            train_tokens,
        }
    }

    pub fn train_tokens(&self) -> usize {
        self.train_tokens.len()
    }

    pub fn valid_tokens(&self) -> usize {
        self.valid_tokens.len()
    }
}

pub struct DatasetRandomIter<'a> {
    all_tokens: &'a [Vec<u32>],
    tokens: Vec<&'a [u32]>,
    current_tokens: &'a [u32],
    indexes_in_bytes: Vec<usize>,
    seq_len: usize,
    device: Device,
}

impl<'a> DatasetRandomIter<'a> {
    pub fn new(ds: &'a Dataset, valid: bool, seq_len: usize, device: Device) -> Self {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let all_tokens = if valid {
            &ds.valid_tokens
        } else {
            &ds.train_tokens
        };
        let mut tokens = all_tokens.iter().map(|s| &**s).collect::<Vec<_>>();
        tokens.shuffle(&mut thread_rng());
        let current_tokens = tokens.pop().unwrap();
        let mut indexes_in_bytes = (0..current_tokens.len().saturating_sub(seq_len))
            .step_by(seq_len)
            .collect::<Vec<_>>();
        indexes_in_bytes.shuffle(&mut thread_rng());
        Self {
            all_tokens,
            tokens,
            current_tokens,
            indexes_in_bytes,
            seq_len,
            device,
        }
    }
}

impl<'a> Iterator for DatasetRandomIter<'a> {
    type Item = candle_core::Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let seq_len = self.seq_len;
        while self.indexes_in_bytes.is_empty() {
            if self.tokens.is_empty() {
                self.tokens = self.all_tokens.iter().map(|s| &**s).collect::<Vec<_>>();
                self.tokens.shuffle(&mut thread_rng());
            }
            self.current_tokens = self.tokens.pop().unwrap();
            self.indexes_in_bytes = (0..self.current_tokens.len().saturating_sub(seq_len))
                .step_by(seq_len)
                .collect::<Vec<_>>();
            self.indexes_in_bytes.shuffle(&mut thread_rng());
        }
        let start_idx = self.indexes_in_bytes.pop().unwrap();
        let tokens = &self.current_tokens[start_idx..start_idx + (seq_len + 1)];
        let inputs = Tensor::new(&tokens[..seq_len], &self.device);
        let targets = Tensor::new(&tokens[1..], &self.device);
        Some(candle_core::error::zip(inputs, targets))
    }
}

fn run_inference(
    tokenizer: &Tokenizer,
    cache: &Cache,
    model: &Llama,
    config: &Config,
) -> anyhow::Result<()> {
    println!("starting the inference loop");
    let mut logits_processor = LogitsProcessor::new(299792458, None, None);
    let mut index_pos = 0;
    let device = Device::cuda_if_available(0)?;

    let prompt = r#"<html lang="en"><body><body>"#;
    print!("{}", prompt);
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    let start_gen = std::time::Instant::now();
    for index in 0.. {
        if tokens.len() >= config.seq_len {
            break;
        }
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, index_pos)?;
        let logits = logits.i((0, logits.dim(1)? - 1))?;
        let start_at = tokens.len().saturating_sub(64);
        let logits =
            candle_transformers::utils::apply_repeat_penalty(&logits, 1.1, &tokens[start_at..])?;
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        // Extracting the last token as a string is complicated, here we just apply some simple
        // heuristics as it seems to work well enough for this example. See the following for more
        // details:
        // https://github.com/huggingface/tokenizers/issues/1141#issuecomment-1562644141
        if let Some(text) = tokenizer.id_to_token(next_token) {
            let text = text.replace('▁', " ").replace("<0x0A>", "\n");
            print!("{text}");
            std::io::stdout().flush()?;
        }
    }
    let dt = start_gen.elapsed();
    println!(
        "\n{} tokens generated ({:.2} token/s)\n",
        tokens.len(),
        tokens.len() as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
