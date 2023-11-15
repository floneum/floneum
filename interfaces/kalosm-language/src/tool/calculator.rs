use kalosm_sample::{
    ChoiceParser, ChoiceParserState, CreateParserState, Either, FloatParser, FloatParserState,
    LiteralParser, LiteralParserOffset, ParseResult, Parser, ParserExt, SequenceParser,
    SequenceParserState,
};
use once_cell::sync::{Lazy, OnceCell};
use std::ops::Deref;

use crate::tool::Tool;

#[derive(Debug)]
struct LazyParser<T>(Box<Lazy<T>>);

impl<T: CreateParserState> CreateParserState for LazyParser<T> {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        let _self: &T = self.0.deref();
        _self.create_parser_state()
    }
}

impl<T: Parser + Default> Default for LazyParser<T> {
    fn default() -> Self {
        Self(Box::new(Lazy::new(Default::default)))
    }
}

impl<T: Parser> Parser for LazyParser<T> {
    type Error = T::Error;

    type Output = T::Output;

    type PartialState = T::PartialState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        self.0.parse(state, input)
    }
}

type InnerParser = ChoiceParser<ChoiceParser<FloatParser, SequenceParser<SequenceParser<SequenceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<ChoiceParser<LiteralParser<&'static str>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LazyParser<EquationParser>>, LiteralParser<&'static str>>>, SequenceParser<SequenceParser<SequenceParser<SequenceParser<LiteralParser<&'static str>, LazyParser<EquationParser>>, ChoiceParser<ChoiceParser<ChoiceParser<LiteralParser<&'static str>, LiteralParser<&'static str>>, LiteralParser<&'static str>>, LiteralParser<&'static str>>>, LazyParser<EquationParser>>, LiteralParser<&'static str>>> ;
type InnerParserState =ChoiceParserState<ChoiceParserState<FloatParserState, SequenceParserState<SequenceParserState<SequenceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<ChoiceParserState<LiteralParserOffset, LiteralParserOffset, (), ()>, LiteralParserOffset, Either<(), ()>, ()>, LiteralParserOffset, Either<Either<(), ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<(), ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, LiteralParserOffset, Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>>, EquationParserState, (Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ())>, LiteralParserOffset, ((Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()), String)>, (), Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, EquationParserParseError>, ()>>, SequenceParserState<SequenceParserState<SequenceParserState<SequenceParserState<LiteralParserOffset, EquationParserState, ()>, ChoiceParserState<ChoiceParserState<ChoiceParserState<LiteralParserOffset, LiteralParserOffset, (), ()>, LiteralParserOffset, Either<(), ()>, ()>, LiteralParserOffset, Either<Either<(), ()>, ()>, ()>, ((), String)>, EquationParserState, (((), String), Either<Either<Either<(), ()>, ()>, ()>)>, LiteralParserOffset, ((((), String), Either<Either<Either<(), ()>, ()>, ()>), String)>, Either<(), Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<Either<(), ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, ()>, EquationParserParseError>, ()>>, Either<Either<Either<Either<(), EquationParserParseError>, Either<Either<Either<(), ()>, ()>, ()>>, EquationParserParseError>, ()>> ;

/// A parser for mathematical equations
pub struct EquationParser {
    parser: InnerParser,
}

impl CreateParserState for EquationParser {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        Default::default()
    }
}

impl Default for EquationParser {
    fn default() -> Self {
        let number = FloatParser::new(f64::MIN..=f64::MAX);
        let function = LiteralParser::new("sqrt")
            .or(LiteralParser::new("abs"))
            .or(LiteralParser::new("exp"))
            .or(LiteralParser::new("ln"))
            .or(LiteralParser::new("sin"))
            .or(LiteralParser::new("cos"))
            .or(LiteralParser::new("tan"))
            .or(LiteralParser::new("asin"))
            .or(LiteralParser::new("acos"))
            .or(LiteralParser::new("atan"))
            .or(LiteralParser::new("atan2"))
            .or(LiteralParser::new("sinh"))
            .or(LiteralParser::new("cosh"))
            .or(LiteralParser::new("tanh"))
            .or(LiteralParser::new("asinh"))
            .or(LiteralParser::new("acosh"))
            .or(LiteralParser::new("atanh"))
            .or(LiteralParser::new("floor"))
            .or(LiteralParser::new("ceil"))
            .or(LiteralParser::new("round"))
            .or(LiteralParser::new("signum"))
            .or(LiteralParser::new("pi"))
            .or(LiteralParser::new("e"));

        let addition = LiteralParser::new(" + ");
        let subtraction = LiteralParser::new(" - ");
        let multiplication = LiteralParser::new(" * ");
        let division = LiteralParser::new(" / ");

        let operation = addition.or(subtraction).or(multiplication).or(division);
        let binary_operation = LiteralParser::new("(")
            .then(LazyParser::<Self>::default())
            .then(operation)
            .then(LazyParser::<Self>::default())
            .then(LiteralParser::new(")"));

        let function_call = function
            .then(LiteralParser::new("("))
            .then(LazyParser::<Self>::default())
            .then(LiteralParser::new(")"));

        let expression = number.or(function_call).or(binary_operation);

        Self { parser: expression }
    }
}

/// An error that can occur while parsing an equation
#[derive(Debug, Clone, Copy)]
pub struct EquationParserParseError;

impl std::fmt::Display for EquationParserParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("EquationParserParseError")
    }
}

impl std::error::Error for EquationParserParseError {}

impl Parser for EquationParser {
    type Error = EquationParserParseError;

    type Output = String;

    type PartialState = EquationParserState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        self.parser
            .parse(&*state.state, input)
            .map(|result| {
                let new_text = state.current_text.clone() + std::str::from_utf8(input).unwrap();
                match result {
                    ParseResult::Incomplete {
                        new_state,
                        required_next,
                    } => ParseResult::Incomplete {
                        new_state: EquationParserState {
                            state: LazyState(Box::new(OnceCell::from(new_state))),
                            current_text: new_text,
                        },
                        required_next,
                    },
                    ParseResult::Finished { remaining, .. } => ParseResult::Finished {
                        remaining,
                        result: new_text,
                    },
                }
            })
            .map_err(|_| EquationParserParseError)
    }
}

/// The state of an equation parser.
#[derive(Default, Debug, Clone)]
pub struct EquationParserState {
    state: LazyState<InnerParserState>,
    current_text: String,
}

#[derive(Debug, Clone, Default)]
struct LazyState<T>(Box<OnceCell<T>>);

impl<T: Default> Deref for LazyState<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self.0.get() {
            Some(state) => state,
            None => {
                let _ = self.0.set(Default::default());
                self.0.get().unwrap()
            }
        }
    }
}

#[test]
fn literal_equation() {
    let parser = EquationParser::default();

    let state = parser.create_parser_state();

    let result = parser.parse(&state, "2".as_bytes());
    println!("{:?}", result);
    assert!(result.is_ok());
}

#[test]
fn function_equation() {
    let parser = EquationParser::default();

    let state = parser.create_parser_state();

    let result = parser.parse(&state, "sqrt(2)".as_bytes());
    println!("{:?}", result);
    assert!(result.is_ok());

    let result = parser.parse(&state, "sqrt(sqrt(2))".as_bytes());
    println!("{:?}", result);
    assert!(result.is_ok());
}

#[test]
fn additon_equation() {
    let parser = EquationParser::default();

    let state = parser.create_parser_state();

    let result = parser.parse(&state, "(2+1)".as_bytes());
    println!("{:?}", result);
    assert!(result.is_ok());

    let result = parser.parse(&state, "((2+1)+1)".as_bytes());
    println!("{:?}", result);
    assert!(result.is_ok());
}

/// A tool that can search the web
pub struct CalculatorTool;

#[async_trait::async_trait]
impl Tool for CalculatorTool {
    type Constraint = EquationParser;

    fn constraints(&self) -> Self::Constraint {
        EquationParser::default()
    }

    fn name(&self) -> String {
        "Calculator".to_string()
    }

    fn input_prompt(&self) -> String {
        "Numerical expression to calculate: ".to_string()
    }

    fn description(&self) -> String {
        let input_prompt = self.input_prompt();
        format!("Evaluate a mathematical expression (made only of numbers and one of the prebuilt math functions). Available functions: sqrt, abs, exp, ln, sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, asinh, acosh, atanh, floor, ceil, round, signum, pi, e\nUse tool with:\nAction: Calculator\nAction Input: the expression\nExample:\nQuestion: What is 2 + 2?\nThought: I should calculate 2 + 2.\nAction: Calculator\n{input_prompt}2 + 2\nObservation: 4\nThought: I now know that 2 + 2 is 4.\nFinal Answer: 4")
    }

    async fn run(&mut self, expr: String) -> String {
        match meval::eval_str(expr){
            Ok(result) => result.to_string(),
            Err(e) => format!("Input was invalid, try again making sure to only use numbers and one of the prebuilt math functions. {e}"),
        }
    }
}
