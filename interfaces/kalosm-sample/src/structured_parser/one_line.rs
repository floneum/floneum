use crate::{CreateParserState, ParseResult, ParseStatus, Parser};

/// One line of text with some non-whitespace characters
#[derive(Debug, Clone, Copy)]
pub struct OneLine;

/// The state of the [`OneLine`] parser
#[derive(Debug, Clone)]
pub struct OneLineState {
    all_whitespace: bool,
    bytes: Vec<u8>,
}

impl CreateParserState for OneLine {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        OneLineState {
            all_whitespace: true,
            bytes: Vec::new(),
        }
    }
}

/// An error that can occur when parsing a [`OneLine`]
#[derive(Debug, Clone)]
pub struct OneLineError;

impl std::fmt::Display for OneLineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OneLineError")
    }
}

impl std::error::Error for OneLineError {}

impl Parser for OneLine {
    type Output = String;
    type PartialState = OneLineState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        if input.is_empty() {
            return Ok(ParseStatus::Incomplete {
                new_state: state.clone(),
                required_next: Default::default(),
            });
        }
        let mut state = state.clone();
        let mut iter = input.iter();
        while let Some(&c) = iter.next() {
            if !c.is_ascii_alphanumeric() || matches!(c, b' ' | b'.' | b'\n') {
                crate::bail!(OneLineError);
            }
            if state.all_whitespace {
                let c = char::from(c);
                if !c.is_whitespace() {
                    state.all_whitespace = false;
                }
            }
            if c == b'\n' {
                if state.all_whitespace {
                    crate::bail!(OneLineError);
                } else {
                    return Ok(ParseStatus::Finished {
                        result: String::from_utf8_lossy(&state.bytes).to_string(),
                        remaining: iter.as_slice(),
                    });
                }
            }
            state.bytes.push(c);
        }
        Ok(ParseStatus::Incomplete {
            new_state: state,
            required_next: Default::default(),
        })
    }
}
