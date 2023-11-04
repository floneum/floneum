use crate::{CreateParserState, ParseResult, Parser};

/// State of a choice parser.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct ChoiceParserState<P1, P2, E> {
    pub(crate) state1: Result<P1, E>,
    pub(crate) state2: Result<P2, E>,
}

impl<P1, P2, E> ChoiceParserState<P1, P2, E> {
    /// Create a new choice parser state.
    pub fn new(state1: P1, state2: P2) -> Self {
        Self {
            state1: Ok(state1),
            state2: Ok(state2),
        }
    }
}

impl<P1: Default, P2: Default, E> Default for ChoiceParserState<P1, P2, E> {
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
        E: Clone,
        O1,
        O2,
        PA1,
        PA2,
        P1: Parser<Error = E, Output = O1, PartialState = PA1> + CreateParserState,
        P2: Parser<Error = E, Output = O2, PartialState = PA2> + CreateParserState,
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
        E: Clone,
        O1,
        O2,
        PA1,
        PA2,
        P1: Parser<Error = E, Output = O1, PartialState = PA1>,
        P2: Parser<Error = E, Output = O2, PartialState = PA2>,
    > Parser for ChoiceParser<P1, P2>
{
    type Error = E;
    type Output = Either<O1, O2>;
    type PartialState = ChoiceParserState<PA1, PA2, E>;

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
                    (Ok(ParseResult::Incomplete(p1)), Ok(ParseResult::Incomplete(p2))) => {
                        let new_state = ChoiceParserState {
                            state1: Ok(p1),
                            state2: Ok(p2),
                        };
                        Ok(ParseResult::Incomplete(new_state))
                    }
                    (Ok(ParseResult::Incomplete(p1)), Err(err2)) => {
                        let new_state = ChoiceParserState {
                            state1: Ok(p1),
                            state2: Err(err2),
                        };
                        Ok(ParseResult::Incomplete(new_state))
                    }
                    (Err(err1), Ok(ParseResult::Incomplete(p2))) => {
                        let new_state = ChoiceParserState {
                            state1: Err(err1),
                            state2: Ok(p2),
                        };
                        Ok(ParseResult::Incomplete(new_state))
                    }

                    // If both parsers fail, we return the error from the first parser
                    (Err(err1), Err(_)) => Err(err1),
                }
            }
            (Ok(p1), Err(err2)) => {
                let result = self.parser1.parse(p1, input)?;
                match result {
                    ParseResult::Finished { result, remaining } => Ok(ParseResult::Finished {
                        result: Either::Left(result),
                        remaining,
                    }),
                    ParseResult::Incomplete(p1) => {
                        let new_state = ChoiceParserState {
                            state1: Ok(p1),
                            state2: Err(err2.clone()),
                        };
                        Ok(ParseResult::Incomplete(new_state))
                    }
                }
            }
            (Err(err1), Ok(p2)) => {
                let result = self.parser2.parse(p2, input)?;
                match result {
                    ParseResult::Finished { result, remaining } => Ok(ParseResult::Finished {
                        result: Either::Right(result),
                        remaining,
                    }),
                    ParseResult::Incomplete(p2) => {
                        let new_state = ChoiceParserState {
                            state1: Err(err1.clone()),
                            state2: Ok(p2),
                        };
                        Ok(ParseResult::Incomplete(new_state))
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
    assert_eq!(parser.parse(&state, b"Goodbye, world!"), Err(()));

    let parser = ChoiceParser::new(
        LiteralParser::new("This isn't a test"),
        LiteralParser::new("This is a test"),
    );
    let state = ChoiceParserState::default();
    assert_eq!(
        parser.parse(&state, b"This isn"),
        Ok(ParseResult::Incomplete(ChoiceParserState {
            state1: Ok(LiteralParserOffset::new(8)),
            state2: Err(()),
        }))
    );
}
