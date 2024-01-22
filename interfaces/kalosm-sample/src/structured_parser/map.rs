use crate::{ParseResult, Parser};

/// A parser that maps the output of another parser.
pub struct MapOutputParser<P: Parser, F: Fn(P::Output) -> O, O> {
    pub(crate) parser: P,
    pub(crate) map: F,
    pub(crate) _output: std::marker::PhantomData<O>,
}

impl<P: Parser, F: Fn(P::Output) -> O, O> Parser for MapOutputParser<P, F, O> {
    type Error = P::Error;
    type Output = O;
    type PartialState = P::PartialState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        let result = self.parser.parse(state, input)?;
        match result {
            ParseResult::Finished { result, remaining } => Ok(ParseResult::Finished {
                result: (self.map)(result),
                remaining,
            }),
            ParseResult::Incomplete {
                new_state,
                required_next,
            } => Ok(ParseResult::Incomplete {
                new_state,
                required_next,
            }),
        }
    }
}
