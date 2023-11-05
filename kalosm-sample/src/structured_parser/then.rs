use crate::{CreateParserState, Either, ParseResult, Parser};

/// State of a sequence parser.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum SequenceParserState<P1, P2, O1> {
    /// The first parser is incomplete.
    FirstParser(P1),
    /// The first parser is finished, and the second parser is incomplete.
    SecondParser(P2, O1),
}

impl<P1, P2, O1> SequenceParserState<P1, P2, O1>{
    /// Create a new sequence parser state.
    pub fn new(state1: P1) -> Self {
        Self::FirstParser(state1)
    }
}

impl<P1: Default, P2, O1> Default for SequenceParserState<P1, P2, O1> {
    fn default() -> Self {
        SequenceParserState::FirstParser(Default::default())
    }
}

impl<
        E1,
        E2,
        O1: Clone,
        O2,
        PA1,
        PA2,
        P1: Parser<Error = E1, Output = O1, PartialState = PA1> + CreateParserState,
        P2: Parser<Error = E2, Output = O2, PartialState = PA2> + CreateParserState,
    > CreateParserState for SequenceParser<P1, P2>
{
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        SequenceParserState::FirstParser(self.parser1.create_parser_state())
    }
}

/// A parser for a sequence of two parsers.
#[derive(Default, Debug, PartialEq, Eq, Copy, Clone)]
pub struct SequenceParser<P1, P2> {
    parser1: P1,
    parser2: P2,
}

impl<P1, P2> SequenceParser<P1, P2> {
    /// Create a new sequence parser.
    pub fn new(parser1: P1, parser2: P2) -> Self {
        Self { parser1, parser2 }
    }
}

impl<
        E1,
        E2,
        O1: Clone,
        O2,
        PA1,
        PA2,
        P1: Parser<Error = E1, Output = O1, PartialState = PA1>,
        P2: Parser<Error = E2, Output = O2, PartialState = PA2> + CreateParserState,
    > Parser for SequenceParser<P1, P2>
{
    type Error = Either<E1, E2>;
    type Output = (O1, O2);
    type PartialState = SequenceParserState<PA1, PA2, O1>;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        match state {
            SequenceParserState::FirstParser(p1) => {
                let result = self.parser1.parse(p1, input).map_err(Either::Left)?;
                match result {
                    ParseResult::Finished {
                        result: o1,
                        remaining,
                    } => {
                        let second_parser_state = self.parser2.create_parser_state();
                        let result = self.parser2.parse(&second_parser_state, remaining).map_err(Either::Right)?;
                        match result {
                            ParseResult::Finished { result, remaining } => {
                                Ok(ParseResult::Finished {
                                    result: (o1, result),
                                    remaining,
                                })
                            }
                            ParseResult::Incomplete(p2) => {
                                let new_state = SequenceParserState::SecondParser(p2, o1);
                                Ok(ParseResult::Incomplete(new_state))
                            }
                        }
                    }
                    ParseResult::Incomplete(p1) => {
                        let new_state = SequenceParserState::FirstParser(p1);
                        Ok(ParseResult::Incomplete(new_state))
                    }
                }
            }
            SequenceParserState::SecondParser(p2, o1) => {
                let result = self.parser2.parse(p2, input).map_err(Either::Right)?;
                match result {
                    ParseResult::Finished { result, remaining } => Ok(ParseResult::Finished {
                        result: (o1.clone(), result),
                        remaining,
                    }),
                    ParseResult::Incomplete(p2) => {
                        let new_state = SequenceParserState::SecondParser(p2, o1.clone());
                        Ok(ParseResult::Incomplete(new_state))
                    }
                }
            }
        }
    }
}

#[test]
fn sequence_parser() {
    use crate::{LiteralParser, LiteralParserOffset};
    let parser = SequenceParser {
        parser1: LiteralParser::new("Hello, "),
        parser2: LiteralParser::new("world!"),
    };
    let state = SequenceParserState::FirstParser(LiteralParserOffset::default());
    assert_eq!(
        parser.parse(&state, b"Hello, world!"),
        Ok(ParseResult::Finished {
            result: ((), ()),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"Hello, "),
        Ok(ParseResult::Incomplete(SequenceParserState::SecondParser(
            LiteralParserOffset::new(0),
            ()
        )))
    );
    assert_eq!(
        parser.parse(
            &parser
                .parse(&state, b"Hello, ")
                .unwrap()
                .unwrap_incomplete(),
            b"world!"
        ),
        Ok(ParseResult::Finished {
            result: ((), ()),
            remaining: &[]
        })
    );
    assert_eq!(parser.parse(&state, b"Goodbye, world!"), Err(Either::Left(())));
}
