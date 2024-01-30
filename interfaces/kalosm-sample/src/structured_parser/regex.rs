use crate::{CreateParserState, Parser};
use regex_automata::{
    dfa::{sparse, Automaton},
    util::primitives::StateID,
};

/// A parser that uses a regex pattern to parse input.
pub struct RegexParser {
    dfa: sparse::DFA<Vec<u8>>,
    config: regex_automata::util::start::Config,
}

impl RegexParser {
    /// Create a new `RegexParser` from a regex pattern.
    pub fn new(regex: &str) -> anyhow::Result<Self> {
        let dfa = sparse::DFA::new(regex)?;

        let config =
            regex_automata::util::start::Config::new().anchored(regex_automata::Anchored::Yes);

        Ok(Self { dfa, config })
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

        Ok(crate::ParseResult::Incomplete {
            new_state: state,
            required_next: "".into(),
        })
    }
}
