use std::borrow::Cow;

use crate::{
    CreateParserState, ParseResult, ParseStatus, Parser, 
};

/// The state of the [`IndexParser`] parser
#[derive(Debug, Clone)]
pub struct IndexParserState<PA> {
    states: Vec<ParseResult<PA>>,
}

/// A parser that parses a sequence of parsers and returns the index of the first parser that succeeds
#[derive(Debug, Clone)]
pub struct IndexParser<S: Parser> {
    parsers: Vec<S>,
}

impl<S: Parser> IndexParser<S> {
    /// Create a new index parser
    pub fn new(parsers: Vec<S>) -> Self {
        Self { parsers }
    }
}

impl<S: CreateParserState> CreateParserState for IndexParser<S> {
    fn create_parser_state(&self) -> Self::PartialState {
        IndexParserState {
            states: self
                .parsers
                .iter()
                .map(|s| Ok(s.create_parser_state()))
                .collect(),
        }
    }
}

impl<S: Parser> Parser for IndexParser<S> {
    type Output = (usize, S::Output);
    type PartialState = IndexParserState<S::PartialState>;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        let mut states = state.states.clone();
        let mut has_incomplete_option = false;
        let mut required_next: Option<Cow<'static, str>> = None;
        let last_index = self.parsers.len() - 1;
        for (i, parser) in self.parsers.iter().enumerate() {
            match &states[i] {
                Ok(state) => {
                    let result = parser.parse(state, input);
                    match result {
                        Ok(ParseStatus::Finished {
                            result,
                            remaining: r,
                        }) => {
                            return Ok(ParseStatus::Finished {
                                result: (i, result),
                                remaining: r,
                            })
                        }
                        Ok(ParseStatus::Incomplete {
                            new_state: s,
                            required_next: new_required_next,
                        }) => {
                            states[i] = Ok(s);
                            has_incomplete_option = true;
                            match required_next {
                                Some(r) => {
                                    let mut common_bytes = 0;
                                    for (byte1, byte2) in r.bytes().zip(new_required_next.bytes()) {
                                        if byte1 != byte2 {
                                            break;
                                        }
                                        common_bytes += 1;
                                    }
                                    required_next = Some(match (r, new_required_next) {
                                        (Cow::Borrowed(required_next), _) => {
                                            Cow::Borrowed(&required_next[common_bytes..])
                                        }
                                        (_, Cow::Borrowed(required_next)) => {
                                            Cow::Borrowed(&required_next[common_bytes..])
                                        }
                                        (Cow::Owned(mut required_next), _) => {
                                            required_next.truncate(common_bytes);
                                            Cow::Owned(required_next)
                                        }
                                    });
                                }
                                None => {
                                    required_next = Some(new_required_next);
                                }
                            }
                        }
                        Err(e) => {
                            if !has_incomplete_option && i == last_index {
                                return Err(e);
                            }
                            states[i] = Err(e);
                        }
                    }
                }
                Err(err) => {
                    if !has_incomplete_option && i == last_index {
                        return Err(err.clone());
                    }
                }
            }
        }
        Ok(ParseStatus::Incomplete {
            new_state: IndexParserState { states },
            required_next: required_next.unwrap_or_default(),
        })
    }
}
