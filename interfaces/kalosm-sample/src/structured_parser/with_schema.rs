use super::{CreateParserState, Parser, SchemaType};

/// A parser that overrides the schema the parser validates
#[derive(Default, Debug, PartialEq, Clone)]
pub struct WithSchema<P> {
    parser: P,
    schema: Option<SchemaType>,
}

impl<P> WithSchema<P> {
    /// Create a new with schema parser.
    pub fn new(parser: P, schema: Option<SchemaType>) -> Self {
        Self { parser, schema }
    }
}

impl<P: Parser> Parser for WithSchema<P> {
    type Output = P::Output;
    type PartialState = P::PartialState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> crate::ParseResult<crate::ParseStatus<'a, Self::PartialState, Self::Output>> {
        self.parser.parse(state, input)
    }

    fn maybe_schema(&self) -> Option<SchemaType> {
        self.schema.clone()
    }
}

impl<P: CreateParserState> CreateParserState for WithSchema<P> {
    fn create_parser_state(&self) -> Self::PartialState {
        self.parser.create_parser_state()
    }
}
