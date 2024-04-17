use std::borrow::Cow;

use crate::bail;

use crate::{CreateParserState, ParseStatus, Parser};

/// A parser for a literal.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct LiteralParser {
    literal: Cow<'static, str>,
}

impl CreateParserState for LiteralParser {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        LiteralParserOffset::default()
    }
}

impl<S: Into<Cow<'static, str>>> From<S> for LiteralParser {
    fn from(literal: S) -> Self {
        Self {
            literal: literal.into(),
        }
    }
}

impl LiteralParser {
    /// Create a new literal parser.
    pub fn new<S: Into<Cow<'static, str>>>(literal: S) -> Self {
        Self {
            literal: literal.into(),
        }
    }
}

/// The state of a literal parser.
#[derive(Default, Debug, PartialEq, Eq, Copy, Clone)]
pub struct LiteralParserOffset {
    pub(crate) offset: usize,
}

impl LiteralParserOffset {
    /// Create a new literal parser state.
    pub fn new(offset: usize) -> Self {
        Self { offset }
    }
}

/// The error type for a literal parser.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct LiteralMismatchError;

impl std::fmt::Display for LiteralMismatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Literal mismatch")
    }
}

impl std::error::Error for LiteralMismatchError {}

impl Parser for LiteralParser {
    type Output = ();
    type PartialState = LiteralParserOffset;

    fn parse<'a>(
        &self,
        state: &LiteralParserOffset,
        input: &'a [u8],
    ) -> crate::ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        let mut bytes_consumed = 0;

        for (input_byte, literal_byte) in input
            .iter()
            .zip(self.literal.as_bytes()[state.offset..].iter())
        {
            if input_byte != literal_byte {
                bail!(LiteralMismatchError);
            }
            bytes_consumed += 1;
        }

        if state.offset + bytes_consumed == self.literal.len() {
            Ok(ParseStatus::Finished {
                result: (),
                remaining: &input[bytes_consumed..],
            })
        } else {
            Ok(ParseStatus::Incomplete {
                new_state: LiteralParserOffset {
                    offset: state.offset + bytes_consumed,
                },
                required_next: {
                    match &self.literal {
                        Cow::Borrowed(cow) => {
                            Cow::Borrowed(cow.split_at(state.offset + bytes_consumed).1)
                        }
                        Cow::Owned(cow) => {
                            Cow::Owned(cow.split_at(state.offset + bytes_consumed).1.to_string())
                        }
                    }
                },
            })
        }
    }
}

#[test]
fn literal_parser() {
    let parser = LiteralParser::new("Hello, world!");
    let state = LiteralParserOffset { offset: 0 };
    assert_eq!(
        parser.parse(&state, b"Hello, world!").unwrap(),
        ParseStatus::Finished {
            result: (),
            remaining: &[]
        }
    );
    assert_eq!(
        parser.parse(&state, b"Hello, ").unwrap(),
        ParseStatus::Incomplete {
            new_state: LiteralParserOffset { offset: 7 },
            required_next: "world!".into()
        }
    );
    assert_eq!(
        parser
            .parse(
                &parser
                    .parse(&state, b"Hello, ")
                    .unwrap()
                    .unwrap_incomplete()
                    .0,
                b"world!"
            )
            .unwrap(),
        ParseStatus::Finished {
            result: (),
            remaining: &[]
        }
    );
    assert!(parser.parse(&state, b"Goodbye, world!").is_err(),);
}
