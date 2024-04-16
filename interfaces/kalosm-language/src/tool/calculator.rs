use kalosm_sample::{ArcParser, ParseResult};
use kalosm_sample::{
    CreateParserState, FloatParser, LiteralParser, ParseStatus, Parser, ParserExt,
};
use once_cell::sync::Lazy;
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

use crate::tool::Tool;

use super::IndexParser;

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
    type Output = T::Output;

    type PartialState = T::PartialState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        self.0.parse(state, input)
    }
}

/// A parser for mathematical equations
pub struct EquationParser {
    parser: ArcParser,
}

impl CreateParserState for EquationParser {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        EquationParserState {
            state: self.parser.create_parser_state(),
            current_text: String::new(),
        }
    }
}

impl Default for EquationParser {
    fn default() -> Self {
        let number = FloatParser::new(f64::MIN..=f64::MAX);
        let function = IndexParser::new(vec![
            LiteralParser::new("sqrt"),
            LiteralParser::new("abs"),
            LiteralParser::new("exp"),
            LiteralParser::new("ln"),
            LiteralParser::new("sin"),
            LiteralParser::new("cos"),
            LiteralParser::new("tan"),
            LiteralParser::new("asin"),
            LiteralParser::new("acos"),
            LiteralParser::new("atan"),
            LiteralParser::new("atan2"),
            LiteralParser::new("sinh"),
            LiteralParser::new("cosh"),
            LiteralParser::new("tanh"),
            LiteralParser::new("asinh"),
            LiteralParser::new("acosh"),
            LiteralParser::new("atanh"),
            LiteralParser::new("floor"),
            LiteralParser::new("ceil"),
            LiteralParser::new("round"),
            LiteralParser::new("signum"),
            LiteralParser::new("pi"),
            LiteralParser::new("e"),
        ]);

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

        Self {
            parser: expression.map_output(|_| ()).boxed(),
        }
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
    type Output = String;

    type PartialState = EquationParserState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        self.parser.parse(&state.state, input).map(|result| {
            let new_text = state.current_text.clone() + std::str::from_utf8(input).unwrap();
            match result {
                ParseStatus::Incomplete {
                    new_state,
                    required_next,
                } => ParseStatus::Incomplete {
                    new_state: EquationParserState {
                        state: new_state,
                        current_text: new_text,
                    },
                    required_next,
                },
                ParseStatus::Finished { remaining, .. } => ParseStatus::Finished {
                    remaining,
                    result: new_text,
                },
            }
        })
    }
}

/// The state of an equation parser.
#[derive(Clone)]
pub struct EquationParserState {
    state: Arc<dyn std::any::Any + Send + Sync>,
    current_text: String,
}

impl Debug for EquationParserState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EquationParserState")
            .field("current_text", &self.current_text)
            .finish()
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
        println!("{expr}");
        match meval::eval_str(expr){
            Ok(result) => result.to_string(),
            Err(e) => format!("Input was invalid, try again making sure to only use numbers and one of the prebuilt math functions. {e}"),
        }
    }
}
