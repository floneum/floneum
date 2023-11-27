use crate::{CreateParserState, ParseResult, Parser};

/// A parser for a literal.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct LiteralParser<S: AsRef<str>> {
    literal: S,
}

impl<S: AsRef<str>> CreateParserState for LiteralParser<S> {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        LiteralParserOffset::default()
    }
}

impl<S: AsRef<str>> From<S> for LiteralParser<S> {
    fn from(literal: S) -> Self {
        Self { literal }
    }
}

impl<S: AsRef<str>> LiteralParser<S> {
    /// Create a new literal parser.
    pub fn new(literal: S) -> Self {
        Self { literal }
    }
}

/// The state of a literal parser.
#[derive(Default, Debug, PartialEq, Eq, Copy, Clone)]
pub struct LiteralParserOffset {
    offset: usize,
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

impl<S: AsRef<str>> Parser for LiteralParser<S> {
    type Error = LiteralMismatchError;
    type Output = ();
    type PartialState = LiteralParserOffset;

    fn parse<'a>(
        &self,
        state: &LiteralParserOffset,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        let mut bytes_consumed = 0;

        for (input_byte, literal_byte) in input
            .iter()
            .zip(self.literal.as_ref().as_bytes()[state.offset..].iter())
        {
            if input_byte != literal_byte {
                return Err(LiteralMismatchError);
            }
            bytes_consumed += 1;
        }

        if state.offset + bytes_consumed == self.literal.as_ref().len() {
            Ok(ParseResult::Finished {
                result: (),
                remaining: &input[bytes_consumed..],
            })
        } else {
            Ok(ParseResult::Incomplete {
                new_state: LiteralParserOffset {
                    offset: state.offset + bytes_consumed,
                },
                required_next: self
                    .literal
                    .as_ref()
                    .split_at(state.offset + bytes_consumed)
                    .1
                    .to_string()
                    .into(),
            })
        }
    }
}

#[test]
fn literal_parser() {
    let parser = LiteralParser {
        literal: "Hello, world!",
    };
    let state = LiteralParserOffset { offset: 0 };
    assert_eq!(
        parser.parse(&state, b"Hello, world!"),
        Ok(ParseResult::Finished {
            result: (),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"Hello, "),
        Ok(ParseResult::Incomplete {
            new_state: LiteralParserOffset { offset: 7 },
            required_next: "world!".into()
        })
    );
    assert_eq!(
        parser.parse(
            &parser
                .parse(&state, b"Hello, ")
                .unwrap()
                .unwrap_incomplete()
                .0,
            b"world!"
        ),
        Ok(ParseResult::Finished {
            result: (),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"Goodbye, world!"),
        Err(LiteralMismatchError)
    );
}
