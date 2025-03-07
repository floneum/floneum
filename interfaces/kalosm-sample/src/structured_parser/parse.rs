use crate::{CreateParserState, SendCreateParserState, SeparatedParser};
use crate::{
    IntegerParser, LiteralParser, ParseStatus, Parser, ParserExt, SequenceParser, StringParser,
};

use super::Either;

/// Data that can be parsed incrementally.
///
/// You can derive this trait for unit values, unit enums or structs or implement it manually for custom types.
///
/// # Example
/// ```rust
/// use kalosm_sample::*;
///
/// // You can derive parse for structs with named fields that implement Parse
/// #[derive(Parse, Clone)]
/// struct Person {
///     name: String,
///     favorite_color: Color,
/// }
///
/// // You can derive parse for enums with only unit variants
/// #[derive(Parse, Clone)]
/// enum Color {
///     Red,
///     Blue,
///     Green,
///     Yellow,
///     Orange,
///     Purple,
///     Pink,
///     Black,
/// }
///
/// // Or you can implement parse manually for custom types
/// #[derive(Clone)]
/// struct MyStruct(i64, String);
///
/// impl Parse for MyStruct {
///     // The only method on parse is new_parser, which returns a parser that outputs the current type
///     fn new_parser() -> impl kalosm_sample::SendCreateParserState<Output = Self> {
///         let number_parser = i64::new_parser();
///         let string_parser = StringParser::new(0..=usize::MAX);
///         kalosm_sample::LiteralParser::new("MyStruct(")
///             .ignore_output_then(number_parser)
///             .then_literal(", ")
///             .then(string_parser)
///             .then_literal(")")
///             .map_output(|(a, b)| Self(a, b))
///     }
/// }
/// ```
pub trait Parse: Clone + Send + Sync {
    /// Create a new parser that parses the current type and can be sent between threads.
    fn new_parser() -> impl SendCreateParserState<Output = Self>;
}

impl<T: Parse> Parse for Box<T> {
    fn new_parser() -> impl SendCreateParserState<Output = Self> {
        T::new_parser().map_output(Box::new)
    }
}

macro_rules! int_parser {
    ($ty:ident, $num:ty, $test:ident) => {
        #[doc = "A parser for `"]
        #[doc = stringify!($num)]
        #[doc = "`."]
        #[derive(Clone, Debug)]
        pub struct $ty {
            parser: IntegerParser,
        }

        impl $ty {
            /// Create a new parser.
            pub fn new() -> Self {
                Self::default()
            }

            /// Set the range of the integers that this parser can parse.
            pub fn with_range(mut self, range: std::ops::RangeInclusive<$num>) -> Self {
                let start = range.start();
                let end = range.end();
                self.parser = IntegerParser::new(*start as i128..=*end as i128);
                self
            }
        }

        impl Default for $ty {
            fn default() -> Self {
                Self {
                    parser: IntegerParser::new((<$num>::MIN as i128)..=(<$num>::MAX as i128)),
                }
            }
        }

        impl CreateParserState for $ty {
            fn create_parser_state(&self) -> <Self as Parser>::PartialState {
                self.parser.create_parser_state()
            }
        }

        impl Parser for $ty {
            type Output = $num;
            type PartialState = <IntegerParser as Parser>::PartialState;

            fn parse<'a>(
                &self,
                state: &Self::PartialState,
                input: &'a [u8],
            ) -> crate::ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
                self.parser
                    .parse(state, input)
                    .map(|result| result.map(|output| output as $num))
            }
        }

        impl Parse for $num {
            fn new_parser() -> impl SendCreateParserState<Output = Self> {
                $ty::default()
            }
        }

        #[test]
        fn $test() {
            let parser = <$num as Parse>::new_parser();
            let state = parser.create_parser_state();
            for _ in 0..100 {
                let input = rand::random::<$num>();
                let input_str = input.to_string() + "\n";
                println!("input: {:?}", input_str);
                let result = parser.parse(&state, input_str.as_bytes());
                if let ParseStatus::Finished {
                    result: input,
                    remaining: b"\n",
                } = result.unwrap()
                {
                    assert_eq!(input, input);
                } else {
                    panic!("Parser did not finish");
                }
            }
        }
    };
}

int_parser!(U8Parser, u8, test_u8);
int_parser!(U16Parser, u16, test_u16);
int_parser!(U32Parser, u32, test_u32);
int_parser!(U64Parser, u64, test_u64);
int_parser!(I8Parser, i8, test_i8);
int_parser!(I16Parser, i16, test_i16);
int_parser!(I32Parser, i32, test_i32);
int_parser!(I64Parser, i64, test_i64);

impl Parse for String {
    fn new_parser() -> impl SendCreateParserState<Output = Self> {
        StringParser::new(0..=usize::MAX)
    }
}

impl<T: Parse + Clone + Send + Sync> Parse for std::vec::Vec<T> {
    fn new_parser() -> impl SendCreateParserState<Output = Self> {
        SequenceParser::new(
            LiteralParser::new("["),
            SequenceParser::new(
                SeparatedParser::new(T::new_parser(), LiteralParser::new(", "), 0..=usize::MAX),
                LiteralParser::new("]"),
            ),
        )
        .map_output(|((), (outputs, ()))| outputs)
    }
}

impl<const N: usize, T: Parse + Clone + Send + Sync> Parse for [T; N] {
    fn new_parser() -> impl SendCreateParserState<Output = Self> {
        SequenceParser::new(
            LiteralParser::new("["),
            SequenceParser::new(
                SeparatedParser::new(T::new_parser(), LiteralParser::new(", "), N..=N),
                LiteralParser::new("]"),
            ),
        )
        .map_output(|((), (outputs, ()))| {
            outputs
                .try_into()
                .unwrap_or_else(|_| panic!("Array is not the correct size"))
        })
    }
}

impl<T: Parse> Parse for Option<T> {
    fn new_parser() -> impl SendCreateParserState<Output = Self> {
        let parser = T::new_parser();
        parser
            .map_output(|output| Some(output))
            .or(LiteralParser::new("null").map_output(|_| None))
    }
}

impl Parse for bool {
    fn new_parser() -> impl SendCreateParserState<Output = Self> {
        LiteralParser::new("true")
            .otherwise(LiteralParser::new("false"))
            .map_output(|output| matches!(output, Either::Left(_)))
    }
}
