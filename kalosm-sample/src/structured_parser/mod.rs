mod integer;
pub use integer::*;
mod float;
pub use float::*;
mod literal;
pub use literal::*;
mod or;
pub use or::*;
mod then;
pub use then::*;
mod string;
pub use string::*;
mod repeat;
pub use repeat::*;
mod has_parser;
pub use has_parser::*;

/// A trait for a parser with a default state.
pub trait CreateParserState: Parser {
    /// Create the default state of the parser.
    fn create_parser_state(&self) -> <Self as Parser>::PartialState;
}

impl<P: CreateParserState> CreateParserState for &P {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        (*self).create_parser_state()
    }
}

impl<P: CreateParserState> CreateParserState for Box<P> {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        (**self).create_parser_state()
    }
}

/// An incremental parser for a structured input.
pub trait Parser {
    /// The error type of the parser.
    type Error;
    /// The output of the parser.
    type Output;
    /// The state of the parser.
    type PartialState;

    /// Parse the given input.
    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error>
    where
        Self: Sized;

    /// Parse this parser, or another other parser.
    fn or<V: Parser<Error = E, Output = O, PartialState = PA>, E, O, PA>(
        self,
        other: V,
    ) -> ChoiceParser<Self, V>
    where
        Self: Sized,
    {
        ChoiceParser {
            parser1: self,
            parser2: other,
        }
    }

    /// Parse this parser, then the other parser.
    fn then<V: Parser<Error = E, Output = O, PartialState = PA>, E, O, PA>(
        self,
        other: V,
    ) -> SequenceParser<Self, V>
    where
        Self: Sized,
    {
        SequenceParser::new(self, other)
    }

    /// Repeat this parser a number of times.
    fn repeat(self, length_range: std::ops::RangeInclusive<usize>) -> RepeatParser<Self>
    where
        Self: Sized,
    {
        RepeatParser::new(self, length_range)
    }
}

impl<P: Parser> Parser for &P {
    type Error = P::Error;
    type Output = P::Output;
    type PartialState = P::PartialState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        (*self).parse(state, input)
    }
}

impl<P: Parser> Parser for Box<P> {
    type Error = P::Error;
    type Output = P::Output;
    type PartialState = P::PartialState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        let _self: &P = &*self;
        _self.parse(state, input)
    }
}

/// A parser for a choice between two parsers.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum OwnedParseResult<P, R> {
    /// The parser is incomplete.
    Incomplete(P),
    /// The parser is finished.
    Finished {
        /// The result of the parser.
        result: R,
        /// The remaining input.
        remaining: Vec<u8>,
    },
}

impl<P, R> From<ParseResult<'_, P, R>> for OwnedParseResult<P, R> {
    fn from(result: ParseResult<P, R>) -> Self {
        match result {
            ParseResult::Incomplete(parser) => OwnedParseResult::Incomplete(parser),
            ParseResult::Finished { result, remaining } => OwnedParseResult::Finished {
                result,
                remaining: remaining.to_vec(),
            },
        }
    }
}

/// The state of a parser.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum ParseResult<'a, P, R> {
    /// The parser is incomplete.
    Incomplete(P),
    /// The parser is finished.
    Finished {
        /// The result of the parser.
        result: R,
        /// The remaining input.
        remaining: &'a [u8],
    },
}

impl<'a, P, R> ParseResult<'a, P, R> {
    /// Take the remaining bytes from the parser.
    pub fn without_remaining(self) -> ParseResult<'static, P, R> {
        match self {
            ParseResult::Finished { result, .. } => ParseResult::Finished {
                result,
                remaining: &[],
            },
            ParseResult::Incomplete(parser) => ParseResult::Incomplete(parser),
        }
    }

    /// Unwrap the parser to a finished result.
    pub fn unwrap_finished(self) -> R {
        match self {
            ParseResult::Finished { result, .. } => result,
            ParseResult::Incomplete(_) => {
                panic!("called `ParseResult::unwrap_finished()` on an `Incomplete` value")
            }
        }
    }

    /// Unwrap the parser to an incomplete result.
    pub fn unwrap_incomplete(self) -> P {
        match self {
            ParseResult::Finished { .. } => {
                panic!("called `ParseResult::unwrap_incomplete()` on a `Finished` value")
            }
            ParseResult::Incomplete(parser) => parser,
        }
    }

    /// Map the result of the parser.
    pub fn map<F, O>(self, f: F) -> ParseResult<'a, P, O>
    where
        F: FnOnce(R) -> O,
    {
        match self {
            ParseResult::Finished { result, remaining } => ParseResult::Finished {
                result: f(result),
                remaining,
            },
            ParseResult::Incomplete(parser) => ParseResult::Incomplete(parser),
        }
    }
}

/// A validator for a string
#[derive(Debug, Clone)]
pub enum StructureParser {
    /// A literal string
    Literal(String),
    /// A number
    Num {
        /// The minimum value of the number
        min: f64,
        /// The maximum value of the number
        max: f64,
        /// If the number must be an integer
        integer: bool,
    },
    /// Either the first or the second parser
    Either {
        /// The first parser
        first: Box<StructureParser>,
        /// The second parser
        second: Box<StructureParser>,
    },
    /// The first parser, then the second parser
    Then {
        /// The first parser
        first: Box<StructureParser>,
        /// The second parser
        second: Box<StructureParser>,
    },
}

/// The state of a structure parser.
#[allow(missing_docs)]
#[derive(Debug, PartialEq, Clone)]
pub enum StructureParserState {
    Literal(LiteralParserOffset),
    NumInt(IntegerParserState),
    Num(FloatParserState),
    Either(ChoiceParserState<Box<StructureParserState>, Box<StructureParserState>, ()>),
    Then(SequenceParserState<Box<StructureParserState>, Box<StructureParserState>, ()>),
}

impl CreateParserState for StructureParser {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        match self {
            StructureParser::Literal(literal) => {
                StructureParserState::Literal(LiteralParser::from(literal).create_parser_state())
            }
            StructureParser::Num { min, max, integer } => {
                if *integer {
                    StructureParserState::NumInt(
                        IntegerParser::new(*min as i64..=*max as i64).create_parser_state(),
                    )
                } else {
                    StructureParserState::Num(FloatParser::new(*min..=*max).create_parser_state())
                }
            }
            StructureParser::Either { first, second } => {
                StructureParserState::Either(ChoiceParserState::new(
                    Box::new(first.create_parser_state()),
                    Box::new(second.create_parser_state()),
                ))
            }
            StructureParser::Then { first, .. } => StructureParserState::Then(
                SequenceParserState::FirstParser(Box::new(first.create_parser_state())),
            ),
        }
    }
}

impl Parser for StructureParser {
    type Error = ();
    type Output = ();
    type PartialState = StructureParserState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error>
    where
        Self: Sized,
    {
        match (self, state) {
            (StructureParser::Literal(lit_parser), StructureParserState::Literal(state)) => {
                LiteralParser::from(lit_parser)
                    .parse(state, input)
                    .map(|result| match result {
                        ParseResult::Finished { result, remaining } => {
                            ParseResult::Finished { result, remaining }
                        }
                        ParseResult::Incomplete(parser) => {
                            ParseResult::Incomplete(StructureParserState::Literal(parser))
                        }
                    })
            }
            (
                StructureParser::Num {
                    min,
                    max,
                    integer: false,
                },
                StructureParserState::Num(state),
            ) => FloatParser::new(*min..=*max)
                .parse(state, input)
                .map(|result| match result {
                    ParseResult::Finished { remaining, .. } => ParseResult::Finished {
                        result: (),
                        remaining,
                    },
                    ParseResult::Incomplete(parser) => {
                        ParseResult::Incomplete(StructureParserState::Num(parser))
                    }
                }),
            (
                StructureParser::Num {
                    min,
                    max,
                    integer: true,
                },
                StructureParserState::NumInt(int),
            ) => IntegerParser::new(*min as i64..=*max as i64)
                .parse(int, input)
                .map(|result| match result {
                    ParseResult::Finished { remaining, .. } => ParseResult::Finished {
                        result: (),
                        remaining,
                    },
                    ParseResult::Incomplete(parser) => {
                        ParseResult::Incomplete(StructureParserState::NumInt(parser))
                    }
                }),
            (StructureParser::Either { first, second }, StructureParserState::Either(state)) => {
                let state = ChoiceParserState {
                    state1: match &state.state1 {
                        Ok(state) => Ok((**state).clone()),
                        Err(()) => Err(()),
                    },
                    state2: match &state.state2 {
                        Ok(state) => Ok((**state).clone()),
                        Err(()) => Err(()),
                    },
                };
                let parser = ChoiceParser::new(first.clone(), second.clone());
                match parser.parse(&state, input) {
                    Ok(ParseResult::Incomplete(state)) => Ok(ParseResult::Incomplete(
                        StructureParserState::Either(ChoiceParserState {
                            state1: state.state1.map(Box::new),
                            state2: state.state2.map(Box::new),
                        }),
                    )),
                    Ok(ParseResult::Finished { remaining, .. }) => Ok(ParseResult::Finished {
                        result: (),
                        remaining,
                    }),
                    Err(_) => Err(()),
                }
            }
            (StructureParser::Then { first, second }, StructureParserState::Then(state)) => {
                let state = SequenceParserState::FirstParser(match &state {
                    SequenceParserState::FirstParser(state) => (**state).clone(),
                    SequenceParserState::SecondParser(state, _) => (**state).clone(),
                });
                let parser = SequenceParser::new(first.clone(), second.clone());
                match parser.parse(&state, input) {
                    Ok(ParseResult::Incomplete(state)) => Ok(ParseResult::Incomplete(
                        StructureParserState::Then(match state {
                            SequenceParserState::FirstParser(state) => {
                                SequenceParserState::FirstParser(Box::new(state))
                            }
                            SequenceParserState::SecondParser(state, _) => {
                                SequenceParserState::SecondParser(Box::new(state), ())
                            }
                        }),
                    )),
                    Ok(ParseResult::Finished { remaining, .. }) => Ok(ParseResult::Finished {
                        result: (),
                        remaining,
                    }),
                    Err(_) => Err(()),
                }
            }
            _ => unreachable!(),
        }
    }
}
