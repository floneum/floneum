use std::{collections::HashMap, sync::RwLock};

use crate::{CreateParserState, Parser};
use regex_automata::{
    dfa::{dense, Automaton},
    util::primitives::StateID,
};

/// A parser that uses a regex pattern to parse input.
pub struct RegexParser {
    dfa: dense::DFA<Vec<u32>>,
    config: regex_automata::util::start::Config,
    // A cache for the required next bytes for each state
    jump_table: RwLock<HashMap<StateID, String>>,
}

impl RegexParser {
    /// Create a new `RegexParser` from a regex pattern.
    #[allow(clippy::result_large_err)]
    pub fn new(regex: &str) -> std::result::Result<Self, regex_automata::dfa::dense::BuildError> {
        let dfa = dense::DFA::new(regex)?;

        let config =
            regex_automata::util::start::Config::new().anchored(regex_automata::Anchored::Yes);

        Ok(Self {
            dfa,
            config,
            jump_table: Default::default(),
        })
    }
}

impl CreateParserState for RegexParser {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        let start_state = self.dfa.start_state(&self.config).unwrap();
        RegexParserState {
            state: start_state,
            value: Vec::new(),
        }
    }
}

impl Parser for RegexParser {
    type Output = String;
    type PartialState = RegexParserState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> crate::ParseResult<crate::ParseStatus<'a, Self::PartialState, Self::Output>> {
        let mut state = state.clone();
        let mut iter = input.iter();
        while let Some(&b) = iter.next() {
            state.state = self.dfa.next_state(state.state, b);
            state.value.push(b);
            // If this is a match state, accept it only if it matches the whole regex
            let finish_state = self.dfa.next_eoi_state(state.state);
            if self.dfa.is_match_state(finish_state) {
                return Ok(crate::ParseStatus::Finished {
                    result: String::from_utf8_lossy(&state.value).to_string(),
                    remaining: iter.as_slice(),
                });
            } else if self.dfa.is_dead_state(state.state) || self.dfa.is_quit_state(state.state) {
                crate::bail!(regex_automata::MatchError::quit(b, 0));
            }
        }

        let mut required_next = String::new();
        let mut required_next_state = state.state;
        let jump_table_read = self.jump_table.read().unwrap();

        if let Some(string) = jump_table_read.get(&required_next_state) {
            required_next.push_str(string);
        } else {
            'required_next: loop {
                let mut one_valid_byte = None;

                if let Some(string) = jump_table_read.get(&required_next_state) {
                    required_next.push_str(string);
                    break;
                }

                for byte in 0..=255 {
                    let next_state = self.dfa.next_state(required_next_state, byte);
                    if self.dfa.is_dead_state(next_state) || self.dfa.is_quit_state(next_state) {
                        continue;
                    }
                    if one_valid_byte.is_some() {
                        break 'required_next;
                    }
                    one_valid_byte = Some((byte, next_state));
                }

                if let Some((byte, new_state)) = one_valid_byte {
                    required_next.push(byte.into());
                    required_next_state = new_state;
                } else {
                    break;
                }
            }

            if !required_next.is_empty() {
                drop(jump_table_read);
                self.jump_table
                    .write()
                    .unwrap()
                    .insert(state.state, required_next.clone());
            }
        }

        Ok(crate::ParseStatus::Incomplete {
            new_state: state,
            required_next: required_next.into(),
        })
    }
}

/// The state of a regex parser.
#[derive(Default, Debug, PartialEq, Eq, Clone)]
pub struct RegexParserState {
    state: StateID,
    value: Vec<u8>,
}

#[test]
fn parse_regex() {
    use crate::ParseStatus;

    let regex = r#"\"\w+\""#;
    let parser = RegexParser::new(regex).unwrap();
    let state = parser.create_parser_state();
    let result = parser.parse(&state, b"\"hello\"world").unwrap();
    assert_eq!(
        result,
        ParseStatus::Finished {
            result: "\"hello\"".to_string(),
            remaining: b"world"
        }
    );

    let result = parser.parse(&state, b"\"hello world\"");
    assert!(result.is_err(),);

    let result = parser.parse(&state, b"\"hel").unwrap();
    match result {
        ParseStatus::Incomplete {
            new_state,
            required_next,
        } => {
            assert_eq!(new_state.value, b"\"hel");
            assert!(required_next.is_empty());
        }
        _ => panic!("unexpected result to be incomplete: {result:?}"),
    }
}

#[test]
fn required_next_regex() {
    use crate::ParseStatus;

    let regex = r#"\{ name: "\w+", description: "[\w ]+" \}"#;
    let parser = RegexParser::new(regex).unwrap();
    let start_state = parser
        .dfa
        .start_state(
            &regex_automata::util::start::Config::new().anchored(regex_automata::Anchored::Yes),
        )
        .unwrap();
    let state = parser.create_parser_state();
    let result = parser.parse(&state, b"").unwrap();
    assert_eq!(
        result,
        ParseStatus::Incomplete {
            new_state: RegexParserState {
                state: start_state,
                value: b"".to_vec(),
            },
            required_next: "{ name: \"".into(),
        }
    );

    let result = parser.parse(&state, b"{ name: \"hello\"").unwrap();

    match result {
        ParseStatus::Incomplete {
            new_state,
            required_next,
        } => {
            assert_eq!(new_state.value, b"{ name: \"hello\"");
            assert_eq!(required_next, ", description: \"");
        }
        _ => panic!("unexpected result to be incomplete: {result:?}"),
    }
}
