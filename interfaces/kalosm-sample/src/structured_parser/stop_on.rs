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
#[derive(Default, Debug, PartialEq, Eq, Clone)]
pub struct StopOnOffset {
    offset: usize,
    text: String,
}

impl StopOnOffset {
    /// Create a new stop on literal parser state.
    pub fn new(offset: usize) -> Self {
        Self {
            offset,
            text: String::new(),
        }
    }
}

impl<S: AsRef<str>> Parser for StopOn<S> {
    type Error = std::convert::Infallible;
    type Output = String;
    type PartialState = StopOnOffset;

    fn parse<'a>(
        &self,
        state: &StopOnOffset,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        let mut new_offset = state.offset;
        let mut text = state.text.clone();

        for (i, (input_byte, literal_byte)) in input
            .iter()
            .zip(self.literal.as_ref().as_bytes()[state.offset..].iter())
            .enumerate()
        {
            if input_byte == literal_byte {
                new_offset += 1;
            } else {
                new_offset = 0;
            }
            if new_offset == self.literal.as_ref().len() {
                text += std::str::from_utf8(&input[..i + 1]).unwrap();
                return Ok(ParseResult::Finished {
                    result: state.text[..state.offset].to_string(),
                    remaining: &input[i..],
                });
            }
        }

        text.push_str(std::str::from_utf8(&input).unwrap());

        Ok(ParseResult::Incomplete {
            new_state: StopOnOffset {
                offset: new_offset,
                text,
            },
            required_next: "".into(),
        })
    }
}

#[test]
fn literal_parser() {
    let parser = StopOn {
        literal: "Hello, world!",
    };
    let state = StopOnOffset {
        offset: 0,
        text: String::new(),
    };
    assert_eq!(
        parser.parse(&state, b"Hello, world!"),
        Ok(ParseResult::Finished {
            result: "".to_string(),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"Hello, world! This is a test"),
        Ok(ParseResult::Finished {
            result: "".to_string(),
            remaining: b" This is a test"
        })
    );
    assert_eq!(
        parser.parse(&state, b"Hello, "),
        Ok(ParseResult::Incomplete {
            new_state: StopOnOffset {
                offset: 7,
                text: "Hello, ".into()
            },
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
            result: "Hello, ".to_string(),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"Goodbye, world!"),
        Ok(ParseResult::Incomplete {
            new_state: StopOnOffset {
                offset: 0,
                text: "Goodbye, world".into()
            },
            required_next: "".into()
        })
    );
}
