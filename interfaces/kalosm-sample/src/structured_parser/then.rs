use std::sync::Arc;

use crate::{CreateParserState, ParseResult, ParseStatus, Parser};

/// State of a sequence parser.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum SequenceParserState<P1, P2, O1> {
    /// The first parser is incomplete.
    FirstParser(P1),
    /// The first parser is finished, and the second parser is incomplete.
    SecondParser(P2, O1),
}

impl<P1, P2, O1> SequenceParserState<P1, P2, O1> {
    /// Create a new sequence parser state.
    pub fn new(state1: P1) -> Self {
        Self::FirstParser(state1)
    }
}

impl<P1: Default, P2, O1> Default for SequenceParserState<P1, P2, O1> {
    fn default() -> Self {
        SequenceParserState::FirstParser(Default::default())
    }
}

impl<P1: CreateParserState, P2: CreateParserState> CreateParserState for SequenceParser<P1, P2> {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        SequenceParserState::FirstParser(self.parser1.create_parser_state())
    }
}

/// A parser for a sequence of two parsers.
#[derive(Default, Debug, PartialEq, Eq, Copy, Clone)]
pub struct SequenceParser<P1, P2> {
    parser1: P1,
    parser2: P2,
}

impl<P1, P2> SequenceParser<P1, P2> {
    /// Create a new sequence parser.
    pub fn new(parser1: P1, parser2: P2) -> Self {
        Self { parser1, parser2 }
    }
}

impl<P1: Parser, P2: CreateParserState> Parser for SequenceParser<P1, P2> {
    type Output = (P1::Output, P2::Output);
    type PartialState = SequenceParserState<P1::PartialState, P2::PartialState, P1::Output>;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> crate::ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        match state {
            SequenceParserState::FirstParser(p1) => {
                let result = self.parser1.parse(p1, input)?;
                match result {
                    ParseStatus::Finished {
                        result: o1,
                        remaining,
                    } => {
                        let second_parser_state = self.parser2.create_parser_state();
                        let result = self.parser2.parse(&second_parser_state, remaining)?;
                        match result {
                            ParseStatus::Finished { result, remaining } => {
                                Ok(ParseStatus::Finished {
                                    result: (o1, result),
                                    remaining,
                                })
                            }
                            ParseStatus::Incomplete {
                                new_state: p2,
                                required_next,
                            } => {
                                let new_state = SequenceParserState::SecondParser(p2, o1);
                                Ok(ParseStatus::Incomplete {
                                    new_state,
                                    required_next,
                                })
                            }
                        }
                    }
                    ParseStatus::Incomplete {
                        new_state: p1,
                        required_next,
                    } => {
                        let new_state = SequenceParserState::FirstParser(p1);
                        Ok(ParseStatus::Incomplete {
                            new_state,
                            required_next,
                        })
                    }
                }
            }
            SequenceParserState::SecondParser(p2, o1) => {
                let result = self.parser2.parse(p2, input)?;
                match result {
                    ParseStatus::Finished { result, remaining } => Ok(ParseStatus::Finished {
                        result: (o1.clone(), result),
                        remaining,
                    }),
                    ParseStatus::Incomplete {
                        new_state: p2,
                        required_next,
                    } => {
                        let new_state = SequenceParserState::SecondParser(p2, o1.clone());
                        Ok(ParseStatus::Incomplete {
                            new_state,
                            required_next,
                        })
                    }
                }
            }
        }
    }
}

#[test]
fn sequence_parser() {
    use crate::{LiteralParser, LiteralParserOffset};
    let parser = SequenceParser {
        parser1: LiteralParser::new("Hello, "),
        parser2: LiteralParser::new("world!"),
    };
    let state = SequenceParserState::FirstParser(LiteralParserOffset::default());
    assert_eq!(
        parser.parse(&state, b"Hello, world!"),
        Ok(ParseStatus::Finished {
            result: ((), ()),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"Hello, "),
        Ok(ParseStatus::Incomplete {
            new_state: SequenceParserState::SecondParser(LiteralParserOffset::new(0), ()),
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
        Ok(ParseStatus::Finished {
            result: ((), ()),
            remaining: &[]
        })
    );
    assert!(parser.parse(&state, b"Goodbye, world!").is_err(),);
}

/// State of a then lazy parser.
#[derive(Debug, PartialEq, Eq)]
pub enum ThenLazyParserState<P1: Parser, P2: Parser> {
    /// The first parser is incomplete.
    FirstParser(P1::PartialState),
    /// The first parser is finished, and the second parser is incomplete.
    SecondParser {
        /// The result of the first parser.
        first_output: P1::Output,
        /// The second parser.
        second_parser: Arc<P2>,
        /// The state of the second parser.
        second_state: P2::PartialState,
    },
}

impl<P1: Parser, P2: Parser> Clone for ThenLazyParserState<P1, P2>
where
    P1::PartialState: Clone,
    P2::PartialState: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Self::FirstParser(first_state) => Self::FirstParser(first_state.clone()),
            Self::SecondParser {
                first_output,
                second_parser,
                second_state,
            } => Self::SecondParser {
                first_output: first_output.clone(),
                second_parser: second_parser.clone(),
                second_state: second_state.clone(),
            },
        }
    }
}

impl<P1: Parser, P2: Parser> ThenLazyParserState<P1, P2> {
    /// Create a new then lazy parser state.
    pub fn new(first_state: P1::PartialState) -> Self {
        Self::FirstParser(first_state)
    }
}

impl<P1: Parser, P2: Parser> Default for ThenLazyParserState<P1, P2>
where
    P1::PartialState: Default,
{
    fn default() -> Self {
        Self::FirstParser(Default::default())
    }
}

/// A parser that is initialized lazily based on the state of the previous parser.
pub struct ThenLazy<P1, F> {
    parser1: P1,
    parser_fn: F,
}

impl<P1: Parser, P2: CreateParserState, F: Fn(&P1::Output) -> P2> ThenLazy<P1, F> {
    /// Create a new parser that is lazily initialized based on the output of the first parser.
    pub fn new(parser1: P1, parser_fn: F) -> Self {
        Self { parser1, parser_fn }
    }
}

impl<P1: CreateParserState, P2: CreateParserState, F: Fn(&P1::Output) -> P2> CreateParserState
    for ThenLazy<P1, F>
{
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        ThenLazyParserState::FirstParser(self.parser1.create_parser_state())
    }
}

impl<P1: Parser, P2: CreateParserState, F: Fn(&P1::Output) -> P2> Parser for ThenLazy<P1, F> {
    type Output = (P1::Output, P2::Output);
    type PartialState = ThenLazyParserState<P1, P2>;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        match state {
            ThenLazyParserState::FirstParser(p1) => {
                let result = self.parser1.parse(p1, input)?;
                match result {
                    ParseStatus::Finished {
                        result: o1,
                        remaining,
                    } => {
                        let parser2 = Arc::new((self.parser_fn)(&o1));
                        let second_parser_state = parser2.create_parser_state();
                        let result = parser2.parse(&second_parser_state, remaining)?;
                        match result {
                            ParseStatus::Finished { result, remaining } => {
                                Ok(ParseStatus::Finished {
                                    result: (o1, result),
                                    remaining,
                                })
                            }
                            ParseStatus::Incomplete {
                                new_state: p2,
                                required_next,
                            } => {
                                let new_state = ThenLazyParserState::SecondParser {
                                    first_output: o1.clone(),
                                    second_parser: parser2.clone(),
                                    second_state: p2,
                                };
                                Ok(ParseStatus::Incomplete {
                                    new_state,
                                    required_next,
                                })
                            }
                        }
                    }
                    ParseStatus::Incomplete {
                        new_state: p1,
                        required_next,
                    } => {
                        let new_state = ThenLazyParserState::FirstParser(p1);
                        Ok(ParseStatus::Incomplete {
                            new_state,
                            required_next,
                        })
                    }
                }
            }
            ThenLazyParserState::SecondParser {
                first_output,
                second_parser,
                second_state,
            } => {
                let result = second_parser.parse(second_state, input)?;
                match result {
                    ParseStatus::Finished { result, remaining } => Ok(ParseStatus::Finished {
                        result: (first_output.clone(), result),
                        remaining,
                    }),
                    ParseStatus::Incomplete {
                        new_state: p2,
                        required_next,
                    } => {
                        let new_state = ThenLazyParserState::SecondParser {
                            first_output: first_output.clone(),
                            second_parser: second_parser.clone(),
                            second_state: p2,
                        };
                        Ok(ParseStatus::Incomplete {
                            new_state,
                            required_next,
                        })
                    }
                }
            }
        }
    }
}
