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
    pub fn new(regex: &str) -> anyhow::Result<Self> {
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
        self.dfa.start_state(&self.config).unwrap()
    }
}

impl Parser for RegexParser {
    type Error = regex_automata::MatchError;
    type Output = ();
    type PartialState = StateID;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<crate::ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        let mut state = *state;
        for (idx, &b) in input.iter().enumerate() {
            state = self.dfa.next_state(state, b);
            if self.dfa.is_match_state(state) {
                // If this is a match state, accept it only if it's the last byte
                return if idx == input.len() - 1 {
                    Ok(crate::ParseResult::Finished {
                        result: (),
                        remaining: Default::default(),
                    })
                } else {
                    Err(regex_automata::MatchError::quit(b, 0))
                };
            } else if self.dfa.is_dead_state(state) || self.dfa.is_quit_state(state) {
                return Err(regex_automata::MatchError::quit(b, 0));
            }
        }

        let mut required_next = String::new();
        let mut required_next_state = state;
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
                    .insert(state, required_next.clone());
            }
        }

        Ok(crate::ParseResult::Incomplete {
            new_state: state,
            required_next: required_next.into(),
        })
    }
}
