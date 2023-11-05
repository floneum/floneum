use crate::CreateParserState;
use crate::{
    IntegerParser, LiteralParser, ParseResult, Parser, RepeatParser,
    SequenceParser, SequenceParserState, StringParser,
};

/// Data that can be parsed incrementally.
pub trait HasParser {
    /// The parser for the data.
    type Parser: Parser<Output = Self>;

    /// Create a new parser.
    fn new_parser() -> Self::Parser;

    /// Create a new parser state.
    fn create_parser_state() -> <Self::Parser as Parser>::PartialState;
}

macro_rules! intparser {
    ($ty:ident, $num:ty) => {
        #[doc = "A parser for `"]
        #[doc = stringify!($num)]
        #[doc = "`."]
        pub struct $ty {
            parser: IntegerParser,
        }

        impl Parser for $ty {
            type Error = <IntegerParser as Parser>::Error;
            type Output = $num;
            type PartialState = <IntegerParser as Parser>::PartialState;

            fn parse<'a>(
                &self,
                state: &Self::PartialState,
                input: &'a [u8],
            ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
                self.parser
                    .parse(state, input)
                    .map(|result| result.map(|output| output as $num))
            }
        }

        impl HasParser for $num {
            type Parser = $ty;

            fn new_parser() -> Self::Parser {
                $ty {
                    parser: IntegerParser::new((<$num>::MIN as i64)..=(<$num>::MAX as i64)),
                }
            }

            fn create_parser_state() -> <Self::Parser as Parser>::PartialState {
                Default::default()
            }
        }
    };
}

intparser!(U8Parser, u8);
intparser!(U16Parser, u16);
intparser!(U32Parser, u32);
intparser!(U64Parser, u64);
intparser!(U128Parser, u128);
intparser!(I8Parser, i8);
intparser!(I16Parser, i16);
intparser!(I32Parser, i32);
intparser!(I64Parser, i64);
intparser!(I128Parser, i128);

impl HasParser for String {
    type Parser = StringParser;

    fn new_parser() -> Self::Parser {
        StringParser::new(0..=usize::MAX)
    }

    fn create_parser_state() -> <Self::Parser as Parser>::PartialState {
        Default::default()
    }
}

/// A parser for a vector of a type.
pub struct VecParser<T: HasParser> {
    parser: SequenceParser<
        LiteralParser<&'static str>,
        SequenceParser<
            RepeatParser<SequenceParser<T::Parser, LiteralParser<&'static str>>>,
            LiteralParser<&'static str>,
        >,
    >,
}

impl<T: HasParser> Parser for VecParser<T>
where
    <T::Parser as Parser>::PartialState: Clone,
    <T::Parser as Parser>::Output: Clone,
    <T as HasParser>::Parser: CreateParserState,
{
    type Error = <SequenceParser<
        LiteralParser<&'static str>,
        SequenceParser<
            RepeatParser<SequenceParser<T::Parser, LiteralParser<&'static str>>>,
            LiteralParser<&'static str>,
        >,
    > as Parser>::Error;
    type Output = Vec<<T::Parser as Parser>::Output>;
    type PartialState = <SequenceParser<
        LiteralParser<&'static str>,
        SequenceParser<
            RepeatParser<SequenceParser<T::Parser, LiteralParser<&'static str>>>,
            LiteralParser<&'static str>,
        >,
    > as Parser>::PartialState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        self.parser.parse(state, input).map(|result| {
            result.map(|((), (outputs, ()))| outputs.into_iter().map(|(output, _)| output).collect())
        })
    }
}

impl<T: HasParser> HasParser for Vec<T>
where
    <T::Parser as Parser>::PartialState: Clone,
    <T::Parser as Parser>::Output: Clone,
    <T as HasParser>::Parser: CreateParserState,
{
    type Parser = VecParser<T>;

    fn new_parser() -> Self::Parser {
        VecParser{
            parser:SequenceParser::new(
            LiteralParser::new("["),
            SequenceParser::new(
                RepeatParser::new(
                    SequenceParser::new(T::new_parser(), LiteralParser::new(",")),
                    0..=usize::MAX,
                ),
                LiteralParser::new("]"),
            ),
        )}
    }

    fn create_parser_state() -> <Self::Parser as Parser>::PartialState {
        SequenceParserState::default()
    }
}
