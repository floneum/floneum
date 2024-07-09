use crate::{CreateParserState, ParseStatus, Parser};

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

impl<P: CreateParserState, S: CreateParserState> CreateParserState for SeparatedParser<P, S> {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        SeparatedParserState {
            new_state_in_progress: false,
            last_state: SeparatedItemState::Item(self.parser.create_parser_state()),
            outputs: Vec::new(),
        }
    }
}

impl<P: CreateParserState, S: CreateParserState> Parser for SeparatedParser<P, S> {
    type Output = Vec<P::Output>;
    type PartialState = SeparatedParserState<P, S>;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> crate::ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        let mut state = state.clone();
        let mut remaining = input;
        let required_next;
        loop {
            match &state.last_state {
                SeparatedItemState::Item(item_state) => {
                    let result = self.parser.parse(item_state, remaining);
                    match result {
                        Ok(ParseStatus::Finished {
                            result,
                            remaining: new_remaining,
                        }) => {
                            state.outputs.push(result);
                            let separator_state = self.separator.create_parser_state();
                            state.new_state_in_progress = false;
                            remaining = new_remaining;
                            if self.length_range.end() == &state.outputs.len() {
                                return Ok(ParseStatus::Finished {
                                    result: state.outputs,
                                    remaining,
                                });
                            }
                            if remaining.is_empty() {
                                match self.separator.parse(&separator_state, remaining) {
                                    Ok(ParseStatus::Incomplete {
                                        required_next: new_required_next,
                                        ..
                                    }) => required_next = Some(new_required_next),
                                    _ => required_next = None,
                                }
                                state.last_state = SeparatedItemState::Separator(separator_state);
                                break;
                            }
                            state.last_state = SeparatedItemState::Separator(separator_state);
                        }
                        Ok(ParseStatus::Incomplete {
                            new_state,
                            required_next: new_required_next,
                        }) => {
                            state.last_state = SeparatedItemState::Item(new_state);
                            state.new_state_in_progress = true;
                            required_next = Some(new_required_next);
                            break;
                        }
                        Err(e) => {
                            if !state.new_state_in_progress
                                && self.length_range.contains(&state.outputs.len())
                            {
                                return Ok(ParseStatus::Finished {
                                    result: state.outputs,
                                    remaining,
                                });
                            } else {
                                crate::bail!(e);
                            }
                        }
                    }
                }
                SeparatedItemState::Separator(separator_state) => {
                    let result = self.separator.parse(separator_state, remaining);
                    match result {
                        Ok(ParseStatus::Finished {
                            remaining: new_remaining,
                            ..
                        }) => {
                            let item_state = self.parser.create_parser_state();
                            state.new_state_in_progress = false;
                            remaining = new_remaining;
                            if self.length_range.end() == &state.outputs.len() {
                                return Ok(ParseStatus::Finished {
                                    result: state.outputs,
                                    remaining,
                                });
                            }
                            if remaining.is_empty() {
                                match self.parser.parse(&item_state, remaining) {
                                    Ok(ParseStatus::Incomplete {
                                        required_next: new_required_next,
                                        ..
                                    }) => required_next = Some(new_required_next),
                                    _ => required_next = None,
                                }
                                break;
                            }
                            state.last_state = SeparatedItemState::Item(item_state);
                        }
                        Ok(ParseStatus::Incomplete {
                            new_state,
                            required_next: new_required_next,
                        }) => {
                            state.last_state = SeparatedItemState::Separator(new_state);
                            state.new_state_in_progress = true;
                            required_next = Some(new_required_next);
                            break;
                        }
                        Err(e) => {
                            if self.length_range.contains(&state.outputs.len()) {
                                return Ok(ParseStatus::Finished {
                                    result: state.outputs,
                                    remaining,
                                });
                            } else {
                                crate::bail!(e);
                            }
                        }
                    }
                }
            }
        }

        Ok(ParseStatus::Incomplete {
            new_state: state,
            required_next: required_next.unwrap_or_default(),
        })
    }
}

#[test]
fn repeat_parser() {
    use crate::{CreateParserState, IntegerParser, LiteralParser, LiteralParserOffset};
    let parser = SeparatedParser::new(LiteralParser::from("a"), LiteralParser::from("b"), 1..=3);
    let state = parser.create_parser_state();
    let result = parser.parse(&state, b"ababa");
    assert_eq!(
        result,
        Ok(ParseStatus::Finished {
            result: vec![(); 3],
            remaining: b"",
        })
    );

    let parser = SeparatedParser::new(IntegerParser::new(1..=3), LiteralParser::from("b"), 1..=3);
    let state = parser.create_parser_state();
    let result = parser.parse(&state, b"1b2b3");
    assert_eq!(
        result,
        Ok(ParseStatus::Finished {
            result: vec![1, 2, 3],
            remaining: b"",
        })
    );

    let parser = SeparatedParser::new(IntegerParser::new(1..=3), LiteralParser::from("bb"), 1..=3);
    let state = parser.create_parser_state();
    let result = parser.parse(&state, b"1bb2b");
    assert_eq!(
        result,
        Ok(ParseStatus::Incomplete {
            new_state: SeparatedParserState {
                new_state_in_progress: true,
                last_state: SeparatedItemState::Separator(LiteralParserOffset::new(1)),
                outputs: vec![1, 2],
            },
            required_next: "b".into()
        })
    );
}
