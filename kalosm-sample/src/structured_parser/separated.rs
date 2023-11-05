use crate::{CreateParserState, Either, ParseResult, Parser};

/// The state of the item in the separated parser.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SeparatedItemState<Item, Separator> {
    /// The item is in progress.
    Item(Item),
    /// The separator is in progress.
    Separator(Separator),
}

/// State of a repeat parser.
#[derive(Debug, PartialEq, Eq)]
pub struct SeparatedParserState<P: Parser, S: Parser> {
    pub(crate) new_state_in_progress: bool,
    pub(crate) last_state: SeparatedItemState<P::PartialState, S::PartialState>,
    pub(crate) outputs: Vec<P::Output>,
}

impl<P: Parser, S: Parser> Clone for SeparatedParserState<P, S>
where
    P::PartialState: Clone,
    P::Output: Clone,
    S::PartialState: Clone,
{
    fn clone(&self) -> Self {
        Self {
            new_state_in_progress: self.new_state_in_progress,
            last_state: self.last_state.clone(),
            outputs: self.outputs.clone(),
        }
    }
}

impl<P: Parser, S: Parser> SeparatedParserState<P, S> {
    /// Create a new repeat parser state.
    pub fn new(
        state: SeparatedItemState<P::PartialState, S::PartialState>,
        outputs: Vec<P::Output>,
    ) -> Self {
        Self {
            new_state_in_progress: false,
            last_state: state,
            outputs,
        }
    }
}

impl<P: Parser, S: Parser> Default for SeparatedParserState<P, S>
where
    P::PartialState: Default,
{
    fn default() -> Self {
        SeparatedParserState {
            new_state_in_progress: false,
            last_state: SeparatedItemState::Item(Default::default()),
            outputs: Default::default(),
        }
    }
}

/// A parser for a repeat of two parsers.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SeparatedParser<P, S> {
    pub(crate) parser: P,
    pub(crate) separator: S,
    length_range: std::ops::RangeInclusive<usize>,
}

impl<P, S> Default for SeparatedParser<P, S>
where
    P: Default,
    S: Default,
{
    fn default() -> Self {
        SeparatedParser {
            parser: Default::default(),
            separator: Default::default(),
            length_range: 0..=usize::MAX,
        }
    }
}

impl<P, S> SeparatedParser<P, S> {
    /// Create a new repeat parser.
    pub fn new(parser: P, separator: S, length_range: std::ops::RangeInclusive<usize>) -> Self {
        Self {
            parser,
            separator,
            length_range,
        }
    }
}

impl<
        E1,
        E2,
        O1,
        O2,
        PA1,
        PA2,
        P: Parser<Error = E1, Output = O1, PartialState = PA1> + CreateParserState,
        S: Parser<Error = E2, Output = O2, PartialState = PA2> + CreateParserState,
    > CreateParserState for SeparatedParser<P, S>
where
    P::PartialState: Clone,
    P::Output: Clone,
    S::PartialState: Clone,
{
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        SeparatedParserState {
            new_state_in_progress: false,
            last_state: SeparatedItemState::Item(self.parser.create_parser_state()),
            outputs: Vec::new(),
        }
    }
}

impl<
        E1,
        E2,
        O1,
        O2,
        PA1,
        PA2,
        P: Parser<Error = E1, Output = O1, PartialState = PA1> + CreateParserState,
        S: Parser<Error = E2, Output = O2, PartialState = PA2> + CreateParserState,
    > Parser for SeparatedParser<P, S>
where
    P::PartialState: Clone,
    P::Output: Clone,
    S::PartialState: Clone,
{
    type Error = Either<E1, E2>;
    type Output = Vec<O1>;
    type PartialState = SeparatedParserState<P, S>;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        let mut state = state.clone();
        let mut remaining = input;
        loop {
            match state.last_state {
                SeparatedItemState::Item(item_state) => {
                    let result = self.parser.parse(&item_state, remaining);
                    match result {
                        Ok(ParseResult::Finished {
                            result,
                            remaining: new_remaining,
                        }) => {
                            state.outputs.push(result);
                            state.last_state =
                                SeparatedItemState::Separator(self.separator.create_parser_state());
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
                            state.last_state = SeparatedItemState::Item(new_state);
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
                                return Err(Either::Left(e));
                            }
                        }
                    }
                }
                SeparatedItemState::Separator(separator_state) => {
                    let result = self.separator.parse(&separator_state, remaining);
                    match result {
                        Ok(ParseResult::Finished {
                            remaining: new_remaining,
                            ..
                        }) => {
                            state.last_state =
                                SeparatedItemState::Item(self.parser.create_parser_state());
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
                            state.last_state = SeparatedItemState::Separator(new_state);
                            state.new_state_in_progress = true;
                            break;
                        }
                        Err(e) => {
                            if self.length_range.contains(&state.outputs.len()) {
                                return Ok(ParseResult::Finished {
                                    result: state.outputs,
                                    remaining,
                                });
                            } else {
                                return Err(Either::Right(e));
                            }
                        }
                    }
                }
            }
        }

        Ok(ParseResult::Incomplete(state))
    }
}

#[test]
fn repeat_parser() {
    use crate::{CreateParserState, IntegerParser, LiteralParser};
    let parser = SeparatedParser::new(LiteralParser::from("a"), LiteralParser::from("b"), 1..=3);
    let state = parser.create_parser_state();
    let result = parser.parse(&state, b"ababa");
    assert_eq!(
        result,
        Ok(ParseResult::Finished {
            result: vec![(); 3],
            remaining: b"",
        })
    );

    let parser = SeparatedParser::new(IntegerParser::new(1..=3), LiteralParser::from("b"), 1..=3);
    let state = parser.create_parser_state();
    let result = parser.parse(&state, b"1b2b3");
    assert_eq!(
        result,
        Ok(ParseResult::Finished {
            result: vec![1, 2, 3],
            remaining: b"",
        })
    );

    let parser = SeparatedParser::new(IntegerParser::new(1..=3), LiteralParser::from("b"), 1..=3);
    let state = parser.create_parser_state();
    let result = parser.parse(&state, b"1b2b");
    assert_eq!(
        result,
        Ok(ParseResult::Incomplete(SeparatedParserState {
            new_state_in_progress: false,
            last_state: SeparatedItemState::Item(IntegerParser::new(1..=3).create_parser_state()),
            outputs: vec![1, 2],
        }))
    );
}
