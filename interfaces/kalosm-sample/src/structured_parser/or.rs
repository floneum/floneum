use std::borrow::Cow;

use crate::{CreateParserState, ParseResult, Parser};

/// State of a choice parser.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct ChoiceParserState<P1, P2, E1, E2> {
    pub(crate) state1: Result<P1, E1>,
    pub(crate) state2: Result<P2, E2>,
}

impl<P1, P2, E1, E2> ChoiceParserState<P1, P2, E1, E2> {
    /// Create a new choice parser state.
    pub fn new(state1: P1, state2: P2) -> Self {
        Self {
            state1: Ok(state1),
            state2: Ok(state2),
        }
    }
}

impl<P1: Default, P2: Default, E1, E2> Default for ChoiceParserState<P1, P2, E1, E2> {
    fn default() -> Self {
        ChoiceParserState {
            state1: Ok(Default::default()),
            state2: Ok(Default::default()),
        }
    }
}

/// A parser for a choice of two parsers.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Default)]
pub struct ChoiceParser<P1, P2> {
    pub(crate) parser1: P1,
    pub(crate) parser2: P2,
}

impl<P1, P2> ChoiceParser<P1, P2> {
    /// Create a new choice parser.
    pub fn new(parser1: P1, parser2: P2) -> Self {
        Self { parser1, parser2 }
    }
}

impl<
        E1: Clone,
        E2: Clone,
        O1,
        O2,
        PA1,
        PA2,
        P1: Parser<Error = E1, Output = O1, PartialState = PA1> + CreateParserState,
        P2: Parser<Error = E2, Output = O2, PartialState = PA2> + CreateParserState,
    > CreateParserState for ChoiceParser<P1, P2>
{
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        ChoiceParserState {
            state1: Ok(self.parser1.create_parser_state()),
            state2: Ok(self.parser2.create_parser_state()),
        }
    }
}

/// A value that can be one of two types.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Either<L, R> {
    /// The value is the left type.
    Left(L),
    /// The value is the right type.
    Right(R),
}

impl<
        E1: Clone,
        E2: Clone,
        O1,
        O2,
        PA1,
        PA2,
        P1: Parser<Error = E1, Output = O1, PartialState = PA1>,
        P2: Parser<Error = E2, Output = O2, PartialState = PA2>,
    > Parser for ChoiceParser<P1, P2>
{
    type Error = Either<E1, E2>;
    type Output = Either<O1, O2>;
    type PartialState = ChoiceParserState<PA1, PA2, E1, E2>;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        match (&state.state1, &state.state2) {
            (Ok(p1), Ok(p2)) => {
                match (self.parser1.parse(p1, input), self.parser2.parse(p2, input)) {
                    // If one parser finishes, we return the result of that parser
                    (Ok(ParseResult::Finished { result, remaining }), _) => {
                        Ok(ParseResult::Finished {
                            result: Either::Left(result),
                            remaining,
                        })
                    }
                    (_, Ok(ParseResult::Finished { result, remaining })) => {
                        Ok(ParseResult::Finished {
                            result: Either::Right(result),
                            remaining,
                        })
                    }
                    // If either parser is incomplete, we return the incomplete state
                    (
                        Ok(ParseResult::Incomplete {
                            new_state: p1,
                            required_next: required_next1,
                        }),
                        Ok(ParseResult::Incomplete {
                            new_state: p2,
                            required_next: required_next2,
                        }),
                    ) => {
                        let new_state = ChoiceParserState {
                            state1: Ok(p1),
                            state2: Ok(p2),
                        };
                        let mut common_bytes = 0;
                        for (byte1, byte2) in required_next1.bytes().zip(required_next2.bytes()) {
                            if byte1 != byte2 {
                                break;
                            }
                            common_bytes += 1;
                        }
                        Ok(ParseResult::Incomplete {
                            new_state,
                            required_next: match (required_next1, required_next2) {
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
                            },
                        })
                    }
                    (
                        Ok(ParseResult::Incomplete {
                            new_state: p1,
                            required_next,
                        }),
                        Err(err2),
                    ) => {
                        let new_state = ChoiceParserState {
                            state1: Ok(p1),
                            state2: Err(err2),
                        };
                        Ok(ParseResult::Incomplete {
                            new_state,
                            required_next,
                        })
                    }
                    (
                        Err(err1),
                        Ok(ParseResult::Incomplete {
                            new_state: p2,
                            required_next,
                        }),
                    ) => {
                        let new_state = ChoiceParserState {
                            state1: Err(err1),
                            state2: Ok(p2),
                        };
                        Ok(ParseResult::Incomplete {
                            new_state,
                            required_next,
                        })
                    }

                    // If both parsers fail, we return the error from the first parser
                    (Err(err1), Err(_)) => Err(Either::Left(err1.clone())),
                }
            }
            (Ok(p1), Err(err2)) => {
                let result = self.parser1.parse(p1, input).map_err(Either::Left)?;
                match result {
                    ParseResult::Finished { result, remaining } => Ok(ParseResult::Finished {
                        result: Either::Left(result),
                        remaining,
                    }),
                    ParseResult::Incomplete {
                        new_state: p1,
                        required_next,
                    } => {
                        let new_state = ChoiceParserState {
                            state1: Ok(p1),
                            state2: Err(err2.clone()),
                        };
                        Ok(ParseResult::Incomplete {
                            new_state,
                            required_next,
                        })
                    }
                }
            }
            (Err(err1), Ok(p2)) => {
                let result = self.parser2.parse(p2, input).map_err(Either::Right)?;
                match result {
                    ParseResult::Finished { result, remaining } => Ok(ParseResult::Finished {
                        result: Either::Right(result),
                        remaining,
                    }),
                    ParseResult::Incomplete {
                        new_state: p2,
                        required_next,
                    } => {
                        let new_state = ChoiceParserState {
                            state1: Err(err1.clone()),
                            state2: Ok(p2),
                        };
                        Ok(ParseResult::Incomplete {
                            new_state,
                            required_next,
                        })
                    }
                }
            }
            (Err(_), Err(_)) => {
                unreachable!()
            }
        }
    }
}

#[test]
fn choice_parser() {
    use crate::{LiteralParser, LiteralParserOffset};
    let parser = ChoiceParser {
        parser1: LiteralParser::new("Hello, "),
        parser2: LiteralParser::new("world!"),
    };
    let state = ChoiceParserState::default();
    assert_eq!(
        parser.parse(&state, b"Hello, "),
        Ok(ParseResult::Finished {
            result: Either::Left(()),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"Hello, "),
        Ok(ParseResult::Finished {
            result: Either::Left(()),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"world!"),
        Ok(ParseResult::Finished {
            result: Either::Right(()),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"Goodbye, world!"),
        Err(Either::Left(()))
    );

    let parser = ChoiceParser::new(
        LiteralParser::new("This isn't a test"),
        LiteralParser::new("This is a test"),
    );
    let state = ChoiceParserState::default();
    assert_eq!(
        parser.parse(&state, b"This isn"),
        Ok(ParseResult::Incomplete {
            new_state: ChoiceParserState {
                state1: Ok(LiteralParserOffset::new(8)),
                state2: Err(()),
            },
            required_next: "'t a test".into()
        })
    );
}
