use std::{collections::HashMap, error::Error, path::PathBuf};

use clap::Parser;
use fusor_gguf::GgufValue;
use nom::{
    AsChar, IResult, Input, Parser as _,
    branch::alt,
    bytes::complete::tag,
    character::complete::{char, digit1, multispace0, one_of},
    combinator::{map, opt, recognize},
    error::ErrorKind,
    multi::separated_list0,
    sequence::delimited,
};

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

fn parse_key_val<T>(s: &str) -> Result<(T, GgufValue), Box<dyn Error + Send + Sync + 'static>>
where
    T: std::str::FromStr,
    T::Err: Error + Send + Sync + 'static,
{
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid KEY=value: no `=` found in `{s}`"))?;
    Ok((
        s[..pos].parse()?,
        parse_gguf_value(&s[pos + 1..])
            .map_err(|e| format!("invalid KEY=value: {e}"))?
            .1,
    ))
}

fn parse_int(input: &str) -> IResult<&str, GgufValue> {
    let (input, (num_str, suffix)) = (
        recognize((opt(one_of("+-")), digit1)),
        alt((
            tag("u8"),
            tag("i8"),
            tag("u16"),
            tag("i16"),
            tag("u32"),
            tag("i32"),
            tag("u64"),
            tag("i64"),
        )),
    )
        .parse(input)?;
    let value = match suffix {
        "u8" => GgufValue::U8(num_str.parse().unwrap()),
        "i8" => GgufValue::I8(num_str.parse().unwrap()),
        "u16" => GgufValue::U16(num_str.parse().unwrap()),
        "i16" => GgufValue::I16(num_str.parse().unwrap()),
        "u32" => GgufValue::U32(num_str.parse().unwrap()),
        "i32" => GgufValue::I32(num_str.parse().unwrap()),
        "u64" => GgufValue::U64(num_str.parse().unwrap()),
        "i64" => GgufValue::I64(num_str.parse().unwrap()),
        _ => unreachable!(),
    };
    Ok((input, value))
}

fn parse_float(input: &str) -> IResult<&str, GgufValue> {
    let (input, (num_str, suffix)) = (
        recognize((opt(one_of("+-")), digit1, opt((char('.'), digit1)))),
        opt(alt((tag("f32"), tag("f64")))),
    )
        .parse(input)?;
    let value = match suffix {
        Some("f32") => GgufValue::F32(num_str.parse().unwrap()),
        Some("f64") => GgufValue::F64(num_str.parse().unwrap()),
        None => {
            if num_str.contains('.') {
                GgufValue::F64(num_str.parse().unwrap())
            } else {
                GgufValue::I32(num_str.parse().unwrap())
            }
        }
        _ => unreachable!(),
    };
    Ok((input, value))
}

fn parse_bool(input: &str) -> IResult<&str, GgufValue> {
    alt((
        map(tag("true"), |_| GgufValue::Bool(true)),
        map(tag("false"), |_| GgufValue::Bool(false)),
    ))
    .parse(input)
}

fn parse_string(input: &str) -> IResult<&str, GgufValue> {
    let (input, s) = input.split_at_position1_complete(
        |item| !item.is_alphanum() && item != '.',
        ErrorKind::AlphaNumeric,
    )?;
    Ok((input, GgufValue::String(s.to_string().into_boxed_str())))
}

fn parse_array(input: &str) -> IResult<&str, GgufValue> {
    let (input, elems) = delimited(
        char('['),
        separated_list0(
            delimited(multispace0, char(','), multispace0),
            parse_gguf_value,
        ),
        char(']'),
    )
    .parse(input)?;
    Ok((input, GgufValue::Array(elems.into_boxed_slice())))
}

fn parse_gguf_value(input: &str) -> IResult<&str, GgufValue> {
    delimited(
        multispace0,
        alt((
            parse_array,
            parse_bool,
            parse_int,
            parse_float,
            parse_string,
        )),
        multispace0,
    )
    .parse(input)
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
            .extend(data.into_iter().map(|(k, v)| (k.into(), v.into())));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_values() {
        assert_eq!(parse_gguf_value("0").unwrap().1, GgufValue::I32(0));
        assert_eq!(parse_gguf_value("-100").unwrap().1, GgufValue::I32(-100));
        assert_eq!(parse_gguf_value("0u8").unwrap().1, GgufValue::U8(0));
        assert_eq!(parse_gguf_value("1.1").unwrap().1, GgufValue::F64(1.1));
        assert_eq!(parse_gguf_value("1.1f32").unwrap().1, GgufValue::F32(1.1));
        assert_eq!(parse_gguf_value("true").unwrap().1, GgufValue::Bool(true));
        assert_eq!(
            parse_gguf_value("hello").unwrap().1,
            GgufValue::String("hello".into())
        );
        assert_eq!(
            parse_gguf_value("hello.world").unwrap().1,
            GgufValue::String("hello.world".into())
        );
        if let GgufValue::Array(vals) = parse_gguf_value("[hello, world]").unwrap().1 {
            assert_eq!(vals[0], GgufValue::String("hello".into()));
            assert_eq!(vals[1], GgufValue::String("world".into()));
        } else {
            panic!("Expected array");
        }
    }
}
