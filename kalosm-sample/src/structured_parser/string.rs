use crate::{CreateParserState, ParseResult, Parser};

/// A parser for an ascii string.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct StringParser<F: Fn(char) -> bool + 'static> {
    len_range: std::ops::RangeInclusive<usize>,
    character_filter: F,
}

impl CreateParserState for StringParser<fn(char) -> bool> {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        StringParserState::default()
    }
}

impl StringParser<fn(char) -> bool> {
    /// Create a new string parser.
    pub fn new(len_range: std::ops::RangeInclusive<usize>,) -> Self {
        Self { len_range, character_filter: |_| true }
    }
}

impl<F: Fn(char) -> bool + 'static> StringParser<F> {
    /// Only allow characters that pass the filter.
    pub fn with_allowed_characters<F2: Fn(char) -> bool + 'static>(
        self,
        character_filter: F2,
    ) -> StringParser<F2> {
        StringParser {
            len_range: self.len_range,
            character_filter,
        }
    }
}

#[derive(Default, Debug, PartialEq, Eq, Clone)]
enum StringParserProgress {
    #[default]
    BeforeQuote,
    InString,
}

/// The state of a literal parser.
#[derive(Default, Debug, PartialEq, Eq, Clone)]
pub struct StringParserState {
    progress: StringParserProgress,
    string: String,
    next_char_escaped: bool,
}

impl StringParserState {
    /// Create a new literal parser state.
    pub fn new(string: String) -> Self {
        let progress = if string.starts_with('"') {
            StringParserProgress::InString
        } else {
            StringParserProgress::BeforeQuote
        };
        Self {
            progress,
            next_char_escaped: string.ends_with('\\'),
            string,
        }
    }
}

impl<F: Fn(char) -> bool + 'static> Parser for StringParser<F> {
    type Error = ();
    type Output = String;
    type PartialState = StringParserState;

    fn parse<'a>(
        &self,
        state: &StringParserState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        let StringParserState {
            mut progress,
            mut string,
            mut next_char_escaped,
        } = state.clone();

        for (i, byte) in input.iter().enumerate() {
            match progress {
                StringParserProgress::BeforeQuote => {
                    if *byte == b'"' {
                        progress = StringParserProgress::InString;
                    } else {
                        return Err(());
                    }
                }
                StringParserProgress::InString => {
                    if (state.next_char_escaped || *byte != b'"') && !(self.character_filter)(*byte as char){
                        return Err(());
                    }

                    if string.len() == *self.len_range.end() && *byte != b'"' {
                        return Err(());
                    }

                    if next_char_escaped {
                        next_char_escaped = false;
                        string.push(*byte as char);
                    } else if *byte == b'"' {
                        if !self.len_range.contains(&string.len()) {
                            return Err(());
                        }
                        return Ok(ParseResult::Finished {
                            remaining: &input[i + 1..],
                            result: string,
                        });
                    } else if *byte == b'\\' {
                        next_char_escaped = true;
                    } else {
                        string.push(*byte as char);
                    }
                }
            }
        }

        Ok(ParseResult::Incomplete(StringParserState {
            progress,
            string,
            next_char_escaped,
        }))
    }
}

#[test]
fn literal_parser() {
    let parser = StringParser::new(1..=10);
    let state = StringParserState::default();
    assert_eq!(
        parser.parse(&state, b"\"Hello, \\\"world!\""),
        Ok(ParseResult::Finished {
            result: "Hello, \"world!".to_string(),
            remaining: &[]
        })
    );

    assert_eq!(
        parser.parse(&state, b"\"Hello, "),
        Ok(ParseResult::Incomplete(StringParserState {
            progress: StringParserProgress::InString,
            string: "Hello, ".to_string(),
            next_char_escaped: false,
        }))
    );

    assert_eq!(
        parser.parse(
            &parser
                .parse(&state, b"\"Hello, ")
                .unwrap()
                .unwrap_incomplete(),
            b"world!\""
        ),
        Ok(ParseResult::Finished {
            result: "Hello, world!".to_string(),
            remaining: &[]
        })
    );
}
