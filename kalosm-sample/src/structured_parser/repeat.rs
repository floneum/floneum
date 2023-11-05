use crate::{CreateParserState, ParseResult, Parser};

/// State of a repeat parser.
#[derive(Debug, PartialEq, Eq)]
pub struct RepeatParserState<P: Parser> {
    pub(crate) new_state_in_progress: bool,
    pub(crate) last_state: P::PartialState,
    pub(crate) outputs: Vec<P::Output>,
}

impl<P: Parser> Clone for RepeatParserState<P>
where
    P::PartialState: Clone,
    P::Output: Clone,
{
    fn clone(&self) -> Self {
        Self {
            new_state_in_progress: self.new_state_in_progress,
            last_state: self.last_state.clone(),
            outputs: self.outputs.clone(),
        }
    }
}

impl<P: Parser> RepeatParserState<P> {
    /// Create a new repeat parser state.
    pub fn new(state: P::PartialState, outputs: Vec<P::Output>) -> Self {
        Self {
            new_state_in_progress: false,
            last_state: state,
            outputs,
        }
    }
}

impl<P: Parser> Default for RepeatParserState<P>
where
    P::PartialState: Default,
{
    fn default() -> Self {
        RepeatParserState {
            new_state_in_progress: false,
            last_state: Default::default(),
            outputs: Default::default(),
        }
    }
}

/// A parser for a repeat of two parsers.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct RepeatParser<P> {
    pub(crate) parser: P,
    length_range: std::ops::RangeInclusive<usize>,
}

impl<P> Default for RepeatParser<P>
where
    P: Default,
{
    fn default() -> Self {
        RepeatParser {
            parser: Default::default(),
            length_range: 0..=usize::MAX,
        }
    }
}

impl<P> RepeatParser<P> {
    /// Create a new repeat parser.
    pub fn new(parser: P, length_range: std::ops::RangeInclusive<usize>) -> Self {
        Self {
            parser,
            length_range,
        }
    }
}

impl<E, O, PA, P: Parser<Error = E, Output = O, PartialState = PA> + CreateParserState>
    CreateParserState for RepeatParser<P>
where
    P::PartialState: Clone,
    P::Output: Clone,
{
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        RepeatParserState {
            new_state_in_progress: false,
            last_state: self.parser.create_parser_state(),
            outputs: Vec::new(),
        }
    }
}

impl<E, O, PA, P: Parser<Error = E, Output = O, PartialState = PA> + CreateParserState> Parser
    for RepeatParser<P>
where
    P::PartialState: Clone,
    P::Output: Clone,
{
    type Error = E;
    type Output = Vec<O>;
    type PartialState = RepeatParserState<P>;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        let mut state = state.clone();
        let mut remaining = input;
        loop {
            let result = self.parser.parse(&state.last_state, remaining);
            match result {
                Ok(ParseResult::Finished {
                    result,
                    remaining: new_remaining,
                }) => {
                    state.outputs.push(result);
                    state.last_state = self.parser.create_parser_state();
                    state.new_state_in_progress = false;
                    remaining = new_remaining;
                    if self.length_range.end() == &state.outputs.len() {
                        return Ok(ParseResult::Finished {
                            result: state.outputs,
                            remaining,
                        });
                    }
                    if remaining.is_empty() {
                        break;
                    }
                }
                Ok(ParseResult::Incomplete(new_state)) => {
                    state.last_state = new_state;
                    state.new_state_in_progress = true;
                    break;
                }
                Err(e) => {
                    if !state.new_state_in_progress
                        && self.length_range.contains(&state.outputs.len())
                    {
                        return Ok(ParseResult::Finished {
                            result: state.outputs,
                            remaining,
                        });
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Ok(ParseResult::Incomplete(state))
    }
}

#[test]
fn repeat_parser() {
    use crate::{IntegerParser, LiteralParser};
    let parser = RepeatParser::new(LiteralParser::from("a"), 1..=3);
    let state = parser.create_parser_state();
    let result = parser.parse(&state, b"aaa");
    assert_eq!(
        result,
        Ok(ParseResult::Finished {
            result: vec![(); 3],
            remaining: b"",
        })
    );

    let parser = RepeatParser::new(IntegerParser::new(1..=3), 1..=3);
    let state = parser.create_parser_state();
    let result = parser.parse(&state, b"123");
    assert_eq!(
        result,
        Ok(ParseResult::Finished {
            result: vec![1, 2, 3],
            remaining: b"",
        })
    );

    let parser = RepeatParser::new(IntegerParser::new(1..=3), 1..=3);
    let state = parser.create_parser_state();
    let result = parser.parse(&state, b"12");
    assert_eq!(
        result,
        Ok(ParseResult::Incomplete(RepeatParserState {
            new_state_in_progress: false,
            last_state: IntegerParser::new(1..=3).create_parser_state(),
            outputs: vec![1, 2],
        }))
    );
}
