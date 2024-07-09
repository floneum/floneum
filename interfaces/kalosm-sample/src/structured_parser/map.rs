use crate::{CreateParserState, ParseStatus, Parser};

/// A parser that maps the output of another parser.
pub struct MapOutputParser<P: Parser, O, F = fn(<P as Parser>::Output) -> O> {
    pub(crate) parser: P,
    pub(crate) map: F,
    pub(crate) _output: std::marker::PhantomData<O>,
}

impl<P: CreateParserState, O: Clone, F: Fn(P::Output) -> O> CreateParserState
    for MapOutputParser<P, O, F>
{
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        self.parser.create_parser_state()
    }
}

impl<P: Parser, O: Clone, F: Fn(P::Output) -> O> Parser for MapOutputParser<P, O, F> {
    type Output = O;
    type PartialState = P::PartialState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> crate::ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        let result = self.parser.parse(state, input)?;
        match result {
            ParseStatus::Finished { result, remaining } => Ok(ParseStatus::Finished {
                result: (self.map)(result),
                remaining,
            }),
            ParseStatus::Incomplete {
                new_state,
                required_next,
            } => Ok(ParseStatus::Incomplete {
                new_state,
                required_next,
            }),
        }
    }
}
