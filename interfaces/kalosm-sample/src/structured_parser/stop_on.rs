use crate::{CreateParserState, ParseStatus, Parser};

type CharFilter = fn(char) -> bool;

/// A parser that parses until a literal is found.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct StopOn<S: AsRef<str> = &'static str, F: Fn(char) -> bool + 'static = CharFilter> {
    literal: S,
    character_filter: F,
}

impl<S: AsRef<str>> CreateParserState for StopOn<S> {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        StopOnOffset::default()
    }
}

impl<S: AsRef<str>> From<S> for StopOn<S> {
    fn from(literal: S) -> Self {
        Self {
            literal,
            character_filter: |_| true,
        }
    }
}

impl<S: AsRef<str>> StopOn<S> {
    /// Create a new literal parser.
    pub fn new(literal: S) -> Self {
        Self {
            literal,
            character_filter: |_| true,
        }
    }
}

impl<S: AsRef<str>, F: Fn(char) -> bool + 'static> StopOn<S, F> {
    /// Only allow characters that pass the filter.
    pub fn filter_characters(self, character_filter: F) -> StopOn<S, F> {
        StopOn {
            literal: self.literal,
            character_filter,
        }
    }

    /// Get the literal that this parser stops on.
    pub fn literal(&self) -> &str {
        self.literal.as_ref()
    }
}

/// An error that can occur while parsing a string literal.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct StopOnParseError;

impl std::fmt::Display for StopOnParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        "StopOnParseError".fmt(f)
    }
}

impl std::error::Error for StopOnParseError {}

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

impl<S: AsRef<str>, F: Fn(char) -> bool + 'static> Parser for StopOn<S, F> {
    type Output = String;
    type PartialState = StopOnOffset;

    fn parse<'a>(
        &self,
        state: &StopOnOffset,
        input: &'a [u8],
    ) -> crate::ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        let mut new_offset = state.offset;
        let mut text = state.text.clone();

        let input_str = std::str::from_utf8(input).unwrap();
        let literal_length = self.literal.as_ref().len();
        let mut literal_iter = self.literal.as_ref()[state.offset..].chars();

        for (i, input_char) in input_str.char_indices() {
            if !(self.character_filter)(input_char) {
                crate::bail!(StopOnParseError);
            }

            let literal_char = literal_iter.next();

            if Some(input_char) == literal_char {
                new_offset += 1;

                if new_offset == literal_length {
                    text += std::str::from_utf8(&input[..i + 1]).unwrap();
                    return Ok(ParseStatus::Finished {
                        result: text,
                        remaining: &input[i + 1..],
                    });
                }
            } else {
                literal_iter = self.literal.as_ref()[state.offset..].chars();
                new_offset = 0;
            }
        }

        text.push_str(input_str);

        Ok(ParseStatus::Incomplete {
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
    let parser = StopOn::new("Hello, world!");
    let state = StopOnOffset {
        offset: 0,
        text: String::new(),
    };
    assert_eq!(
        parser.parse(&state, b"Hello, world!"),
        Ok(ParseStatus::Finished {
            result: "Hello, world!".to_string(),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"Hello, world! This is a test"),
        Ok(ParseStatus::Finished {
            result: "Hello, world!".to_string(),
            remaining: b" This is a test"
        })
    );
    assert_eq!(
        parser.parse(&state, b"Hello, "),
        Ok(ParseStatus::Incomplete {
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
        Ok(ParseStatus::Finished {
            result: "Hello, world!".to_string(),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"Goodbye, world!"),
        Ok(ParseStatus::Incomplete {
            new_state: StopOnOffset {
                offset: 0,
                text: "Goodbye, world!".into()
            },
            required_next: "".into()
        })
    );
}
