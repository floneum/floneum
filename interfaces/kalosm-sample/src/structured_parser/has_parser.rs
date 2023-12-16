use crate::{CreateParserState, SeparatedParser};
use crate::{
    IntegerParser, LiteralParser, ParseResult, Parser, RepeatParser, SequenceParser,
    SequenceParserState, StringParser,
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

macro_rules! int_parser {
    ($ty:ident, $num:ty, $test:ident) => {
        #[doc = "A parser for `"]
        #[doc = stringify!($num)]
        #[doc = "`."]
        #[derive(Clone, Debug)]
        pub struct $ty {
            parser: IntegerParser,
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
                $ty::default()
            }

            fn create_parser_state() -> <Self::Parser as Parser>::PartialState {
                Default::default()
            }
        }

        #[test]
        fn $test() {
            let parser = <$num as HasParser>::new_parser();
            let state = <$num as HasParser>::create_parser_state();
            for _ in 0..100 {
                let input = rand::random::<$num>();
                let input_str = input.to_string() + "\n";
                println!("input: {:?}", input_str);
                let result = parser.parse(&state, input_str.as_bytes());
                assert_eq!(
                    result,
                    Ok(ParseResult::Finished {
                        result: input,
                        remaining: b"\n",
                    })
                );
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

impl HasParser for String {
    type Parser = StringParser<fn(char) -> bool>;

    fn new_parser() -> Self::Parser {
        StringParser::new(0..=usize::MAX)
    }

    fn create_parser_state() -> <Self::Parser as Parser>::PartialState {
        Default::default()
    }
}

/// A parser for a vector of a type.
#[derive(Clone, Debug)]
pub struct VecParser<T: HasParser> {
    parser: SequenceParser<
        LiteralParser<&'static str>,
        SequenceParser<
            SeparatedParser<T::Parser, LiteralParser<&'static str>>,
            LiteralParser<&'static str>,
        >,
    >,
}

impl<T: HasParser> Default for VecParser<T>
where
    <T::Parser as Parser>::PartialState: Clone,
    <T::Parser as Parser>::Output: Clone,
    <T as HasParser>::Parser: CreateParserState,
{
    fn default() -> Self {
        Self {
            parser: SequenceParser::new(
                LiteralParser::new("["),
                SequenceParser::new(
                    SeparatedParser::new(T::new_parser(), LiteralParser::new(", "), 0..=usize::MAX),
                    LiteralParser::new("]"),
                ),
            ),
        }
    }
}

impl<T: HasParser> CreateParserState for VecParser<T>
where
    <T::Parser as Parser>::PartialState: Clone,
    <T::Parser as Parser>::Output: Clone,
    <T as HasParser>::Parser: CreateParserState,
{
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        self.parser.create_parser_state()
    }
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
            SeparatedParser<T::Parser, LiteralParser<&'static str>>,
            LiteralParser<&'static str>,
        >,
    > as Parser>::PartialState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        self.parser
            .parse(state, input)
            .map(|result| result.map(|((), (outputs, ()))| outputs))
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
        VecParser::default()
    }

    fn create_parser_state() -> <Self::Parser as Parser>::PartialState {
        SequenceParserState::default()
    }
}

/// A parser for a fixed size array of a type.
pub struct ArrayParser<const N: usize, T: HasParser> {
    parser: SequenceParser<
        LiteralParser<&'static str>,
        SequenceParser<
            SeparatedParser<T::Parser, LiteralParser<&'static str>>,
            LiteralParser<&'static str>,
        >,
    >,
}

impl<const N: usize, T: HasParser> Default for ArrayParser<N, T>
where
    <T::Parser as Parser>::PartialState: Clone,
    <T::Parser as Parser>::Output: Clone,
    <T as HasParser>::Parser: CreateParserState,
{
    fn default() -> Self {
        Self {
            parser: SequenceParser::new(
                LiteralParser::new("["),
                SequenceParser::new(
                    SeparatedParser::new(T::new_parser(), LiteralParser::new(", "), N..=N),
                    LiteralParser::new("]"),
                ),
            ),
        }
    }
}

impl<const N: usize, T: HasParser> CreateParserState for ArrayParser<N, T>
where
    <T::Parser as Parser>::PartialState: Clone,
    <T::Parser as Parser>::Output: Clone,
    <T as HasParser>::Parser: CreateParserState,
{
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        self.parser.create_parser_state()
    }
}

impl<const N: usize, T: HasParser> Parser for ArrayParser<N, T>
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
    type Output = [<T::Parser as Parser>::Output; N];
    type PartialState = <SequenceParser<
        LiteralParser<&'static str>,
        SequenceParser<
            SeparatedParser<T::Parser, LiteralParser<&'static str>>,
            LiteralParser<&'static str>,
        >,
    > as Parser>::PartialState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        self.parser.parse(state, input).map(|result| {
            result.map(|((), (outputs, ()))| {
                outputs
                    .try_into()
                    .unwrap_or_else(|_| panic!("ArrayParser: wrong number of elements"))
            })
        })
    }
}

impl<const N: usize, T: HasParser> HasParser for [T; N]
where
    <T::Parser as Parser>::PartialState: Clone,
    <T::Parser as Parser>::Output: Clone,
    <T as HasParser>::Parser: CreateParserState,
{
    type Parser = ArrayParser<N, T>;

    fn new_parser() -> Self::Parser {
        ArrayParser::default()
    }

    fn create_parser_state() -> <Self::Parser as Parser>::PartialState {
        SequenceParserState::default()
    }
}
