use std::error::Error;

use kalosm::{language::*, *};
use language::kalosm_sample::*;

/// One line of text with some non-whitespace characters
#[derive(Debug, Clone, Copy)]
pub struct OneAsciiLine;

/// The state of the [`OneLine`] parser
#[derive(Debug, Clone)]
pub struct OneAsciiLineState {
    all_whitespace: bool,
    bytes: Vec<u8>,
}

impl CreateParserState for OneAsciiLine {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        OneAsciiLineState {
            all_whitespace: true,
            bytes: Vec::new(),
        }
    }
}

/// An error that can occur when parsing a [`OneLine`]
#[derive(Debug, Clone)]
pub struct OneAsciiLineError;

impl std::fmt::Display for OneAsciiLineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OneLineError")
    }
}

impl Error for OneAsciiLineError {}

impl Parser for OneAsciiLine {
    type Error = OneAsciiLineError;
    type Output = String;
    type PartialState = OneAsciiLineState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<kalosm_sample::ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        if input.is_empty() {
            return Ok(ParseResult::Incomplete {
                new_state: state.clone(),
                required_next: Default::default(),
            });
        }
        let mut state = state.clone();
        let mut iter = input.iter();
        while let Some(&c) = iter.next() {
            if !c.is_ascii() {
                return Err(OneAsciiLineError);
            }
            if state.all_whitespace {
                let c = char::from(c);
                if !c.is_whitespace() {
                    state.all_whitespace = false;
                }
            }
            if c == b'\n' || c == b'\r' {
                if state.all_whitespace {
                    return Err(OneAsciiLineError);
                } else {
                    return Ok(ParseResult::Finished {
                        result: String::from_utf8_lossy(&state.bytes).to_string(),
                        remaining: iter.as_slice(),
                    });
                }
            }
            state.bytes.push(c);
        }
        Ok(ParseResult::Incomplete {
            new_state: state,
            required_next: Default::default(),
        })
    }
}

#[tokio::main]
async fn main() {
    let character_description =
        "The assistant will respond like a cute cat named Kittens. The assistant will never reveal that it is an AI or assistant.";
    let character_name = "Kittens";

    let mut model = Llama::new_chat();
    let mut chat = Chat::builder(&mut model)
        .with_system_prompt(character_description)
        .constrain_response(move |_, _| {
            LiteralParser::new(format!("(Responding as {}) ", character_name)).then(OneAsciiLine)
        })
        .map_bot_response(move |response, _| {
            response
                .trim_start_matches(&format!("(Responding as {}) ", character_name))
                .trim()
        })
        .build();

    loop {
        let output_stream = chat
            .add_message(prompt_input("\n> ").unwrap())
            .await
            .unwrap();
        print!("Bot: ");
        output_stream.to_std_out().await.unwrap();
    }
}
