use std::ops::{Deref, DerefMut};

use crate::{CreateParserState, Parse, SendCreateParserState};
use crate::{ParseStatus, Parser, StringParser};

#[derive(Clone, Debug)]
/// A single word.
pub struct Sentence<const MIN_LENGTH: usize = 1, const MAX_LENGTH: usize = 200>(pub String);

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> Sentence<MIN_LENGTH, MAX_LENGTH> {
    /// Create a new word.
    pub fn new(word: String) -> Self {
        assert!(word.len() >= MIN_LENGTH);
        assert!(word.len() <= MAX_LENGTH);
        Self(word)
    }
}

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> From<Sentence<MIN_LENGTH, MAX_LENGTH>>
    for String
{
    fn from(word: Sentence<MIN_LENGTH, MAX_LENGTH>) -> Self {
        word.0
    }
}

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> From<String>
    for Sentence<MIN_LENGTH, MAX_LENGTH>
{
    fn from(word: String) -> Self {
        Self(word)
    }
}

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> Deref for Sentence<MIN_LENGTH, MAX_LENGTH> {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> DerefMut
    for Sentence<MIN_LENGTH, MAX_LENGTH>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A parser for a word.
pub struct SentenceParser<const MIN_LENGTH: usize = 1, const MAX_LENGTH: usize = 200> {
    parser: StringParser<fn(char) -> bool>,
}

impl SentenceParser {
    /// Create a new default sentence parser
    pub fn new() -> Self {
        Self::default()
    }
}

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> Default
    for SentenceParser<MIN_LENGTH, MAX_LENGTH>
{
    fn default() -> Self {
        Self {
            parser: StringParser::new(MIN_LENGTH..=MAX_LENGTH).with_allowed_characters(|c| {
                c.is_ascii_alphanumeric() || matches!(c, ' ' | '-' | ';' | ',')
            }),
        }
    }
}

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> CreateParserState
    for SentenceParser<MIN_LENGTH, MAX_LENGTH>
{
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        self.parser.create_parser_state()
    }
}

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> Parser
    for SentenceParser<MIN_LENGTH, MAX_LENGTH>
{
    type Output = Sentence<MIN_LENGTH, MAX_LENGTH>;
    type PartialState = <StringParser<fn(char) -> bool> as Parser>::PartialState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> crate::ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        self.parser
            .parse(state, input)
            .map(|result| result.map(Into::into))
    }
}

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> Parse for Sentence<MIN_LENGTH, MAX_LENGTH> {
    fn new_parser() -> impl SendCreateParserState<Output = Self> {
        SentenceParser::default()
    }
}
