use crate::{CreateParserState, ParseStatus, Parser};

/// A parser that maps the output of another parser.
pub struct MapOutputParser<P, F, O> {
    pub(crate) parser: P,
    pub(crate) map: F,
    pub(crate) _output: std::marker::PhantomData<O>,
}

impl<P: CreateParserState, F: Fn(P::Output) -> O, O> CreateParserState
    for MapOutputParser<P, F, O>
{
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        self.parser.create_parser_state()
    }
}

impl<P: Parser, F: Fn(P::Output) -> O, O> Parser for MapOutputParser<P, F, O> {
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
