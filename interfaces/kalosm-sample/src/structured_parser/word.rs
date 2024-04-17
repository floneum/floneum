use std::ops::{Deref, DerefMut};

use crate::{CreateParserState, HasParser};
use crate::{ParseStatus, Parser, StringParser};

#[derive(Clone, Debug)]
/// A single word.
pub struct Word<const MIN_LENGTH: usize = 1, const MAX_LENGTH: usize = 20>(pub String);

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> Word<MIN_LENGTH, MAX_LENGTH> {
    /// Create a new word.
    pub fn new(word: String) -> Self {
        assert!(word.len() >= MIN_LENGTH);
        assert!(word.len() <= MAX_LENGTH);
        Self(word)
    }
}

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> From<Word<MIN_LENGTH, MAX_LENGTH>>
    for String
{
    fn from(word: Word<MIN_LENGTH, MAX_LENGTH>) -> Self {
        word.0
    }
}

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> From<String>
    for Word<MIN_LENGTH, MAX_LENGTH>
{
    fn from(word: String) -> Self {
        Self(word)
    }
}

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> Deref for Word<MIN_LENGTH, MAX_LENGTH> {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> DerefMut for Word<MIN_LENGTH, MAX_LENGTH> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A parser for a word.
pub struct WordParser<const MIN_LENGTH: usize = 1, const MAX_LENGTH: usize = 20> {
    parser: StringParser<fn(char) -> bool>,
}

impl WordParser {
    /// Create a new default word parser
    pub fn new() -> Self {
        Self::default()
    }
}

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> Default
    for WordParser<MIN_LENGTH, MAX_LENGTH>
{
    fn default() -> Self {
        Self {
            parser: StringParser::new(MIN_LENGTH..=MAX_LENGTH)
                .with_allowed_characters(|c| c.is_ascii_alphanumeric()),
        }
    }
}

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> CreateParserState
    for WordParser<MIN_LENGTH, MAX_LENGTH>
{
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        self.parser.create_parser_state()
    }
}

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> Parser
    for WordParser<MIN_LENGTH, MAX_LENGTH>
{
    type Output = Word<MIN_LENGTH, MAX_LENGTH>;
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

impl<const MIN_LENGTH: usize, const MAX_LENGTH: usize> HasParser for Word<MIN_LENGTH, MAX_LENGTH> {
    type Parser = WordParser<MIN_LENGTH, MAX_LENGTH>;

    fn new_parser() -> Self::Parser {
        WordParser::default()
    }

    fn create_parser_state() -> <Self::Parser as Parser>::PartialState {
        Default::default()
    }
}
