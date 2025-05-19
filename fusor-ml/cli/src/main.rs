use std::{error::Error, path::PathBuf};

use clap::Parser;
use fusor_gguf::GgufValue;
use nom::{
    IResult, Parser as _,
    branch::alt,
    bytes::complete::tag,
    character::complete::{char, digit1, multispace0, one_of},
    combinator::{map, opt, recognize},
    multi::separated_list0,
    sequence::delimited,
};

#[derive(clap::Parser, Clone)]
#[clap(name = "fusor-ml", version = "0.1.0", author = "Fusor ML Team")]
enum Command {
    /// Add metadata to a gguf file
    AddMetadata {
        /// Path to the gguf file
        #[clap(short, long)]
        path: PathBuf,
        /// Metadata to add. Format: KEY=VALUE
        #[arg(short = 'D', value_parser = parse_key_val::<String>)]
        data: Vec<(String, GgufValue)>,
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
    let (input, s) = nom::character::complete::alphanumeric1.parse(input)?;
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
        Command::AddMetadata { path, data } => {
            let reader = std::fs::File::open(&path).unwrap();
            let mut reader = std::io::BufReader::new(reader);
            let mut gguf = fusor_gguf::GgufMetadata::read(&mut reader).unwrap();
            gguf.metadata
                .extend(data.into_iter().map(|(k, v)| (k.into(), v.into())));
        }
    }
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
        if let GgufValue::Array(vals) = parse_gguf_value("[hello, world]").unwrap().1 {
            assert_eq!(vals[0], GgufValue::String("hello".into()));
            assert_eq!(vals[1], GgufValue::String("world".into()));
        } else {
            panic!("Expected array");
        }
    }
}
