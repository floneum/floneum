use crate::{CreateParserState, ParseResult, Parser};

/// A parser that parses until a literal is found.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct StopOn<S: AsRef<str>> {
    literal: S,
}

impl<S: AsRef<str>> CreateParserState for StopOn<S> {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        StopOnOffset::default()
    }
}

impl<S: AsRef<str>> From<S> for StopOn<S> {
    fn from(literal: S) -> Self {
        Self { literal }
    }
}

impl<S: AsRef<str>> StopOn<S> {
    /// Create a new stop on literal parser.
    pub fn new(literal: S) -> Self {
        Self { literal }
    }

    /// Get the literal that this parser stops on.
    pub fn literal(&self) -> &str {
        self.literal.as_ref()
    }
}

/// The state of a stop on literal parser.
#[derive(Default, Debug, PartialEq, Eq, Copy, Clone)]
pub struct StopOnOffset {
    offset: usize,
}

impl StopOnOffset {
    /// Create a new stop on literal parser state.
    pub fn new(offset: usize) -> Self {
        Self { offset }
    }
}

impl<S: AsRef<str>> Parser for StopOn<S> {
    type Error = std::convert::Infallible;
    type Output = ();
    type PartialState = StopOnOffset;

    fn parse<'a>(
        &self,
        state: &StopOnOffset,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        let mut new_offset = state.offset;
        let mut input_offset = 0;

        for (input_byte, literal_byte) in input
            .iter()
            .zip(self.literal.as_ref().as_bytes()[state.offset..].iter())
        {
            if input_byte == literal_byte {
                new_offset += 1;
            } else {
                new_offset = 0;
            }
            input_offset += 1;
        }

        if new_offset == self.literal.as_ref().len() {
            Ok(ParseResult::Finished {
                result: (),
                remaining: &input[input_offset..],
            })
        } else {
            Ok(ParseResult::Incomplete {
                new_state: StopOnOffset { offset: new_offset },
                required_next: "".into(),
            })
        }
    }
}

#[test]
fn literal_parser() {
    let parser = StopOn {
        literal: "Hello, world!",
    };
    let state = StopOnOffset { offset: 0 };
    assert_eq!(
        parser.parse(&state, b"Hello, world!"),
        Ok(ParseResult::Finished {
            result: (),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"Hello, world! This is a test"),
        Ok(ParseResult::Finished {
            result: (),
            remaining: b" This is a test"
        })
    );
    assert_eq!(
        parser.parse(&state, b"Hello, "),
        Ok(ParseResult::Incomplete {
            new_state: StopOnOffset { offset: 7 },
            required_next: "".into()
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
        Ok(ParseResult::Incomplete {
            new_state: StopOnOffset { offset: 0 },
            required_next: "".into()
        })
    );
}
