#![allow(clippy::type_complexity)]

pub use kalosm_parse_macro::Parse;
mod integer;
use std::{
    any::Any,
    borrow::Cow,
    error::Error,
    fmt::{Debug, Display},
    ops::Deref,
    sync::{Arc, OnceLock},
};

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
mod separated;
pub use separated::*;
mod parse;
pub use parse::*;
mod word;
pub use word::*;
mod sentence;
pub use sentence::*;
mod stop_on;
pub use stop_on::*;
mod map;
pub use map::*;
mod regex;
pub use regex::*;
mod arc_linked_list;
pub(crate) use arc_linked_list::*;
mod schema;
pub use schema::*;

/// An error that occurred while parsing.
#[derive(Debug, Clone)]
pub struct ParserError(Arc<anyhow::Error>);

/// Bail out with the given error.
#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return $crate::ParseResult::Err($crate::ParserError::msg($msg))
    };
    ($err:expr $(,)?) => {
        return $crate::ParseResult::Err($crate::ParserError::from($err))
    };
    ($fmt:expr, $($arg:tt)*) => {
        return $crate::ParseResult::Err($crate::ParserError::msg(format!($fmt, $($arg)*)))
    };
}

impl ParserError {
    /// Create a new error with the given message.
    pub fn msg(msg: impl Display + Debug + Send + Sync + 'static) -> Self {
        Self(Arc::new(anyhow::Error::msg(msg)))
    }
}

impl PartialEq for ParserError {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ParserError {}

impl AsRef<dyn Error> for ParserError {
    fn as_ref(&self) -> &(dyn Error + 'static) {
        let err: &anyhow::Error = self.0.as_ref();
        err.as_ref()
    }
}

impl AsRef<dyn std::error::Error + Send + Sync + 'static> for ParserError {
    fn as_ref(&self) -> &(dyn std::error::Error + Send + Sync + 'static) {
        let err: &anyhow::Error = self.0.as_ref();
        err.as_ref()
    }
}

impl Deref for ParserError {
    type Target = (dyn Error + Send + Sync + 'static);

    fn deref(&self) -> &(dyn Error + Send + Sync + 'static) {
        let err: &anyhow::Error = self.0.as_ref();
        err.deref()
    }
}

impl<E> From<E> for ParserError
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn from(value: E) -> Self {
        Self(Arc::new(anyhow::Error::from(value)))
    }
}

/// A result type for parsers.
pub type ParseResult<T> = std::result::Result<T, ParserError>;

/// An auto trait for a Send parser with a default state.
pub trait SendCreateParserState:
    Send + Sync + CreateParserState<PartialState: Send + Sync, Output: Send + Sync>
{
}

impl<P: CreateParserState<PartialState: Send + Sync, Output: Send + Sync> + Send + Sync>
    SendCreateParserState for P
{
}

/// A trait for a parser with a default state.
pub trait CreateParserState: Parser {
    /// Create the default state of the parser.
    fn create_parser_state(&self) -> <Self as Parser>::PartialState;
}

impl<P: ?Sized + CreateParserState> CreateParserState for &P {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        (*self).create_parser_state()
    }
}

impl<P: ?Sized + CreateParserState> CreateParserState for Box<P> {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        (**self).create_parser_state()
    }
}

impl<P: ?Sized + CreateParserState> CreateParserState for Arc<P> {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        (**self).create_parser_state()
    }
}

impl<O: Clone> CreateParserState for ArcParser<O> {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        self.0.create_parser_state()
    }
}

/// An incremental parser for a structured input.
pub trait Parser {
    /// The output of the parser.
    type Output: Clone;
    /// The state of the parser.
    type PartialState: Clone;

    /// Parse the given input.
    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>>;
}

impl Parser for () {
    type Output = ();
    type PartialState = ();

    fn parse<'a>(
        &self,
        _state: &Self::PartialState,
        input: &'a [u8],
    ) -> ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        Ok(ParseStatus::Finished {
            result: (),
            remaining: input,
        })
    }
}

impl<P: ?Sized + Parser> Parser for &P {
    type Output = P::Output;
    type PartialState = P::PartialState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        (*self).parse(state, input)
    }
}

impl<P: ?Sized + Parser> Parser for Box<P> {
    type Output = P::Output;
    type PartialState = P::PartialState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        let _self: &P = self;
        _self.parse(state, input)
    }
}

impl<P: ?Sized + Parser> Parser for Arc<P> {
    type Output = P::Output;
    type PartialState = P::PartialState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        let _self: &P = self;
        _self.parse(state, input)
    }
}

trait AnyCreateParserState:
    Parser<PartialState = Arc<dyn Any + Send + Sync>> + CreateParserState + Send + Sync
{
}

impl<P: Parser<PartialState = Arc<dyn Any + Send + Sync>> + CreateParserState + Send + Sync>
    AnyCreateParserState for P
{
}

/// A boxed parser.
pub struct ArcParser<O = ()>(Arc<dyn AnyCreateParserState<Output = O> + Send + Sync>);

impl<O> Clone for ArcParser<O> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<O> ArcParser<O> {
    fn new<P>(parser: P) -> Self
    where
        P: Parser<Output = O, PartialState = Arc<dyn Any + Send + Sync>>
            + CreateParserState
            + Send
            + Sync
            + 'static,
    {
        ArcParser(Arc::new(parser))
    }
}

impl<O: Clone> Parser for ArcParser<O> {
    type Output = O;
    type PartialState = Arc<dyn Any + Send + Sync>;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        let _self: &dyn Parser<Output = O, PartialState = Arc<dyn Any + Send + Sync>> = &self.0;
        _self.parse(state, input)
    }
}

/// A wrapper for a parser that implements an easily boxable version of Parser.
struct AnyParser<P>(P);

impl<P> Parser for AnyParser<P>
where
    P: Parser,
    P::PartialState: Send + Sync + 'static,
{
    type Output = P::Output;
    type PartialState = Arc<dyn Any + Sync + Send>;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        let state = state.downcast_ref::<P::PartialState>().ok_or_else(|| {
            struct StateIsNotOfTheCorrectType;
            impl std::fmt::Display for StateIsNotOfTheCorrectType {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "State is not of the correct type")
                }
            }
            impl std::fmt::Debug for StateIsNotOfTheCorrectType {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "State is not of the correct type")
                }
            }
            impl Error for StateIsNotOfTheCorrectType {}
            StateIsNotOfTheCorrectType
        })?;
        self.0
            .parse(state, input)
            .map(|result| result.map_state(|state| Arc::new(state) as Arc<dyn Any + Sync + Send>))
    }
}

impl<P: CreateParserState> CreateParserState for AnyParser<P>
where
    P: Parser,
    P::Output: Send + Sync + 'static,
    P::PartialState: Send + Sync + 'static,
{
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        Arc::new(self.0.create_parser_state())
    }
}

/// An extension trait for parsers.
pub trait ParserExt: Parser {
    /// Parse this parser, or another other parser.
    fn otherwise<V: Parser>(self, other: V) -> ChoiceParser<Self, V>
    where
        Self: Sized,
    {
        ChoiceParser {
            parser1: self,
            parser2: other,
        }
    }

    /// Parse this parser, or another other parser with the same type
    fn or<V: Parser<Output = Self::Output>>(
        self,
        other: V,
    ) -> MapOutputParser<ChoiceParser<Self, V>, Self::Output>
    where
        Self: Sized,
    {
        self.otherwise(other).map_output(|either| match either {
            Either::Left(left) => left,
            Either::Right(right) => right,
        })
    }

    /// Parse this parser, then the other parser.
    fn then<V: Parser>(self, other: V) -> SequenceParser<Self, V>
    where
        Self: Sized,
    {
        SequenceParser::new(self, other)
    }

    /// Parse this parser, then the other parser that is created base on the output of this parser.
    fn then_lazy<V, F>(self, other: F) -> ThenLazy<Self, F>
    where
        Self: Sized,
        V: CreateParserState,
        F: FnOnce(&Self::Output) -> V + Copy,
    {
        ThenLazy::new(self, other)
    }

    /// Parse this parser, then the other parser while ignoring the current parser's output.
    fn ignore_output_then<V: CreateParserState>(
        self,
        other: V,
    ) -> MapOutputParser<SequenceParser<Self, V>, <V as Parser>::Output>
    where
        Self: Sized,
    {
        SequenceParser::new(self, other).map_output(|(_, second)| second)
    }

    /// Parse this parser, then the other parser while ignoring the output of the other parser.
    fn then_ignore_output<V: CreateParserState>(
        self,
        other: V,
    ) -> MapOutputParser<SequenceParser<Self, V>, <Self as Parser>::Output>
    where
        Self: Sized,
    {
        SequenceParser::new(self, other).map_output(|(first, _)| first)
    }

    /// Parse this parser, then a literal. This is equivalent to `.then_ignore_output(LiteralParser::new(literal))`.
    fn then_literal(
        self,
        literal: impl Into<Cow<'static, str>>,
    ) -> MapOutputParser<SequenceParser<Self, LiteralParser>, <Self as Parser>::Output>
    where
        Self: Sized,
    {
        self.then_ignore_output(LiteralParser::new(literal))
    }

    /// Repeat this parser a number of times.
    fn repeat(self, length_range: std::ops::RangeInclusive<usize>) -> RepeatParser<Self>
    where
        Self: Sized,
    {
        RepeatParser::new(self, length_range)
    }

    /// Map the output of this parser.
    fn map_output<F, O>(self, f: F) -> MapOutputParser<Self, O, F>
    where
        Self: Sized,
        F: Fn(Self::Output) -> O,
    {
        MapOutputParser {
            parser: self,
            map: f,
            _output: std::marker::PhantomData,
        }
    }

    /// Get a boxed version of this parser.
    fn boxed(self) -> ArcParser<Self::Output>
    where
        Self: CreateParserState + Sized + Send + Sync + 'static,
        Self::Output: Send + Sync + 'static,
        Self::PartialState: Send + Sync + 'static,
    {
        ArcParser::new(AnyParser(self))
    }
}

impl<P: Parser> ParserExt for P {}

/// A parser that is lazily initialized.
pub struct LazyParser<P, F> {
    parser: OnceLock<P>,
    parser_fn: F,
}

impl<P: Parser, F: FnOnce() -> P + Copy> LazyParser<P, F> {
    /// Create a new parser that is lazily initialized.
    pub fn new(parser_fn: F) -> Self {
        Self {
            parser: OnceLock::new(),
            parser_fn,
        }
    }

    fn get_parser(&self) -> &P {
        self.parser.get_or_init(self.parser_fn)
    }
}

impl<P: CreateParserState, F: FnOnce() -> P + Copy> CreateParserState for LazyParser<P, F> {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        self.get_parser().create_parser_state()
    }
}

impl<P: CreateParserState, F: FnOnce() -> P + Copy> From<F> for LazyParser<P, F> {
    fn from(parser_fn: F) -> Self {
        Self::new(parser_fn)
    }
}

impl<P: Parser, F: FnOnce() -> P + Copy> Parser for LazyParser<P, F> {
    type Output = P::Output;
    type PartialState = P::PartialState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        self.get_parser().parse(state, input)
    }
}

/// A parser for a choice between two parsers.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum OwnedParseResult<P, R> {
    /// The parser is incomplete.
    Incomplete {
        /// The new state of the parser.
        new_state: P,
        /// The text that is required next.
        required_next: Cow<'static, str>,
    },
    /// The parser is finished.
    Finished {
        /// The result of the parser.
        result: R,
        /// The remaining input.
        remaining: Vec<u8>,
    },
}

impl<P, R> From<ParseStatus<'_, P, R>> for OwnedParseResult<P, R> {
    fn from(result: ParseStatus<P, R>) -> Self {
        match result {
            ParseStatus::Incomplete {
                new_state,
                required_next,
            } => OwnedParseResult::Incomplete {
                new_state,
                required_next,
            },
            ParseStatus::Finished { result, remaining } => OwnedParseResult::Finished {
                result,
                remaining: remaining.to_vec(),
            },
        }
    }
}

/// The state of a parser.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ParseStatus<'a, P, R> {
    /// The parser is incomplete.
    Incomplete {
        /// The new state of the parser.
        new_state: P,
        /// The text that is required next.
        required_next: Cow<'static, str>,
    },
    /// The parser is finished.
    Finished {
        /// The result of the parser.
        result: R,
        /// The remaining input.
        remaining: &'a [u8],
    },
}

impl<'a, P, R> ParseStatus<'a, P, R> {
    /// Take the remaining bytes from the parser.
    pub fn without_remaining(self) -> ParseStatus<'static, P, R> {
        match self {
            ParseStatus::Finished { result, .. } => ParseStatus::Finished {
                result,
                remaining: &[],
            },
            ParseStatus::Incomplete {
                new_state,
                required_next,
            } => ParseStatus::Incomplete {
                new_state,
                required_next,
            },
        }
    }

    /// Unwrap the parser to a finished result.
    pub fn unwrap_finished(self) -> R {
        match self {
            ParseStatus::Finished { result, .. } => result,
            ParseStatus::Incomplete { .. } => {
                panic!("called `ParseStatus::unwrap_finished()` on an `Incomplete` value")
            }
        }
    }

    /// Unwrap the parser to an incomplete result.
    pub fn unwrap_incomplete(self) -> (P, Cow<'static, str>) {
        match self {
            ParseStatus::Finished { .. } => {
                panic!("called `ParseStatus::unwrap_incomplete()` on a `Finished` value")
            }
            ParseStatus::Incomplete {
                new_state,
                required_next,
            } => (new_state, required_next),
        }
    }

    /// Map the result of the parser.
    pub fn map<F, O>(self, f: F) -> ParseStatus<'a, P, O>
    where
        F: FnOnce(R) -> O,
    {
        match self {
            ParseStatus::Finished { result, remaining } => ParseStatus::Finished {
                result: f(result),
                remaining,
            },
            ParseStatus::Incomplete {
                new_state,
                required_next,
            } => ParseStatus::Incomplete {
                new_state,
                required_next,
            },
        }
    }

    /// Map the state of the parser.
    pub fn map_state<F, O>(self, f: F) -> ParseStatus<'a, O, R>
    where
        F: FnOnce(P) -> O,
    {
        match self {
            ParseStatus::Finished { result, remaining } => {
                ParseStatus::Finished { result, remaining }
            }
            ParseStatus::Incomplete {
                new_state,
                required_next,
            } => ParseStatus::Incomplete {
                new_state: f(new_state),
                required_next,
            },
        }
    }
}

/// A validator for a string
#[derive(Debug, Clone)]
pub enum StructureParser {
    /// A literal string
    Literal(Cow<'static, str>),
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
    Either(ChoiceParserState<Box<StructureParserState>, Box<StructureParserState>>),
    Then(SequenceParserState<Box<StructureParserState>, Box<StructureParserState>, ()>),
}

impl CreateParserState for StructureParser {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        match self {
            StructureParser::Literal(literal) => StructureParserState::Literal(
                LiteralParser::from(literal.clone()).create_parser_state(),
            ),
            StructureParser::Num { min, max, integer } => {
                if *integer {
                    StructureParserState::NumInt(
                        IntegerParser::new(*min as i128..=*max as i128).create_parser_state(),
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
    type Output = ();
    type PartialState = StructureParserState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        match (self, state) {
            (StructureParser::Literal(lit_parser), StructureParserState::Literal(state)) => {
                LiteralParser::from(lit_parser.clone())
                    .parse(state, input)
                    .map(|result| result.map(|_| ()).map_state(StructureParserState::Literal))
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
                .map(|result| result.map(|_| ()).map_state(StructureParserState::Num)),
            (
                StructureParser::Num {
                    min,
                    max,
                    integer: true,
                },
                StructureParserState::NumInt(int),
            ) => IntegerParser::new(*min as i128..=*max as i128)
                .parse(int, input)
                .map(|result| result.map(|_| ()).map_state(StructureParserState::NumInt)),
            (StructureParser::Either { first, second }, StructureParserState::Either(state)) => {
                let state = ChoiceParserState {
                    state1: state
                        .state1
                        .as_ref()
                        .map(|state| (**state).clone())
                        .map_err(Clone::clone),
                    state2: state
                        .state2
                        .as_ref()
                        .map(|state| (**state).clone())
                        .map_err(Clone::clone),
                };
                let parser = ChoiceParser::new(first.clone(), second.clone());
                parser.parse(&state, input).map(|result| match result {
                    ParseStatus::Incomplete { required_next, .. } => ParseStatus::Incomplete {
                        new_state: StructureParserState::Either(ChoiceParserState {
                            state1: state.state1.map(Box::new),
                            state2: state.state2.map(Box::new),
                        }),
                        required_next,
                    },
                    ParseStatus::Finished { remaining, .. } => ParseStatus::Finished {
                        result: (),
                        remaining,
                    },
                })
            }
            (StructureParser::Then { first, second }, StructureParserState::Then(state)) => {
                let state = SequenceParserState::FirstParser(match &state {
                    SequenceParserState::FirstParser(state) => (**state).clone(),
                    SequenceParserState::SecondParser(state, _) => (**state).clone(),
                });
                let parser = SequenceParser::new(first.clone(), second.clone());
                parser.parse(&state, input).map(|result| match result {
                    ParseStatus::Incomplete { required_next, .. } => ParseStatus::Incomplete {
                        new_state: StructureParserState::Then(match state {
                            SequenceParserState::FirstParser(state) => {
                                SequenceParserState::FirstParser(Box::new(state))
                            }
                            SequenceParserState::SecondParser(state, _) => {
                                SequenceParserState::SecondParser(Box::new(state), ())
                            }
                        }),
                        required_next,
                    },
                    ParseStatus::Finished { remaining, .. } => ParseStatus::Finished {
                        result: (),
                        remaining,
                    },
                })
            }
            _ => unreachable!(),
        }
    }
}
