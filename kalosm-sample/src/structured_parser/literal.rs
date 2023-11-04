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

impl<S: AsRef<str>> Parser for LiteralParser<S> {
    type Error = ();
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
                return Err(());
            }
            bytes_consumed += 1;
        }

        if state.offset + bytes_consumed == self.literal.as_ref().len() {
            Ok(ParseResult::Finished {
                result: (),
                remaining: &input[bytes_consumed..],
            })
        } else {
            Ok(ParseResult::Incomplete(LiteralParserOffset {
                offset: state.offset + bytes_consumed,
            }))
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
        Ok(ParseResult::Incomplete(LiteralParserOffset { offset: 7 }))
    );
    assert_eq!(
        parser.parse(
            &parser
                .parse(&state, b"Hello, ")
                .unwrap()
                .unwrap_incomplete(),
            b"world!"
        ),
        Ok(ParseResult::Finished {
            result: (),
            remaining: &[]
        })
    );
    assert_eq!(parser.parse(&state, b"Goodbye, world!"), Err(()));
}
