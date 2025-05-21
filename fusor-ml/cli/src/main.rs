use std::{collections::HashMap, error::Error, path::PathBuf};

use clap::Parser;
use fusor_gguf::GgufValue;
use parse::parse_key_val;

mod parse;

#[derive(clap::Parser, Clone)]
#[clap(name = "fusor-ml", version = "0.1.0", author = "Fusor ML Team")]
enum Command {
    /// Add metadata to a gguf file
    AddMetadata {
        /// Path to the input gguf file
        #[clap(short, long)]
        input: PathBuf,
        /// Path to the output gguf file
        #[clap(short, long)]
        output: PathBuf,
        /// Metadata to add. Format: KEY=VALUE
        #[arg(short = 'D', value_parser = parse_key_val::<String>)]
        data: Vec<(String, GgufValue)>,
    },
    /// Fuse the tokenizer metadata into the gguf file
    FuseTokenizer {
        /// Path to the input gguf file
        #[clap(short, long)]
        input: PathBuf,
        /// Path to the tokenizer file
        #[clap(short, long)]
        tokenizer: PathBuf,
        /// Path to the tokenizer config file
        #[clap(short, long)]
        tokenizer_config: Option<PathBuf>,
        /// Path to the output gguf file
        #[clap(short, long)]
        output: PathBuf,
    },
}

fn main() {
    let command = Command::parse();

    match command {
        Command::AddMetadata {
            input,
            output,
            data,
        } => add_metadata(input, output, data).unwrap(),
        Command::FuseTokenizer {
            input,
            tokenizer,
            output,
            tokenizer_config,
        } => fuse_tokenizer(input, tokenizer, output, tokenizer_config).unwrap(),
    }
}

fn fuse_tokenizer(
    input: PathBuf,
    tokenizer: PathBuf,
    output: PathBuf,
    tokenizer_config: Option<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    let mut metadata = Vec::new();

    let tokenizer_json: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(&tokenizer)?)?;
    // Store model.vocab in tokenizer.ggml.tokens
    let token_map = tokenizer_json["model"]["vocab"].as_object().unwrap();
    let added_tokens = tokenizer_json["added_tokens"].as_array().unwrap();
    let mut tokens = token_map
        .iter()
        .map(|(token, id)| (token.clone(), id.as_u64().unwrap()))
        .collect::<Vec<_>>();
    // Add added_tokens to the token map
    for token in added_tokens {
        let id = token["id"].as_u64().unwrap();
        let token_str = token["content"].as_str().unwrap();
        tokens.push((token_str.to_string(), id));
    }
    // Find the max id
    let max_id = tokens.iter().map(|(_, id)| *id).max().unwrap();
    let mut tokens_array = vec!["<conversion-error>".to_string(); (max_id + 1) as usize];
    for (token, id) in &tokens {
        tokens_array[*id as usize] = token.clone();
    }
    let tokens = GgufValue::Array(
        tokens_array
            .iter()
            .map(|s| GgufValue::String(s.clone().into_boxed_str()))
            .collect(),
    );
    metadata.push(("tokenizer.ggml.tokens".into(), tokens));

    // Try to find bos_token and eos_token in tokenizer_config
    if let Some(config) = tokenizer_config {
        let config_json: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&config)?)?;
        if let Some(bos_token) = config_json["bos_token"].as_str() {
            let id = tokens_array
                .iter()
                .position(|s| s == bos_token)
                .ok_or_else(|| format!("Invalid bos_token: {bos_token}"))?;
            metadata.push((
                "tokenizer.ggml.bos_token_id".into(),
                GgufValue::U32(id as u32),
            ));
        }
        if let Some(eos_token) = config_json["eos_token"].as_str() {
            let id = tokens_array
                .iter()
                .position(|s| s == eos_token)
                .ok_or_else(|| format!("Invalid eos_token: {eos_token}"))?;
            metadata.push((
                "tokenizer.ggml.eos_token_id".into(),
                GgufValue::U32(id as u32),
            ));
        }
    }

    add_metadata(input, output, metadata)?;

    Ok(())
}

fn add_metadata(
    input: PathBuf,
    output: PathBuf,
    data: Vec<(String, GgufValue)>,
) -> Result<(), Box<dyn Error>> {
    let mut tensor;
    let mut gguf;
    {
        let reader = std::fs::File::open(&input)?;
        let mut reader = std::io::BufReader::new(reader);
        gguf = fusor_gguf::GgufMetadata::read(&mut reader)?;
        gguf.metadata
            .extend(data.into_iter().map(|(k, v)| (k.into(), v)));
        tensor = HashMap::new();
        for (key, value) in gguf.tensor_infos.iter() {
            tensor.insert(
                key.clone(),
                value.read_tensor_bytes(&mut reader, gguf.tensor_data_offset)?,
            );
        }
    }
    let writer = std::fs::File::create(&output)?;
    let mut writer = std::io::BufWriter::new(writer);
    gguf.write(
        &mut writer,
        tensor.iter().map(|(k, v)| (k.as_ref(), v.as_ref())),
    )?;
    Ok(())
}
