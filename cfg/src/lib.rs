use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    hash::Hash,
    ops::ControlFlow,
};

use rustc_hash::{FxHashMap, FxHashSet};
use slab::Slab;

use crate::{
    parse::Grammar,
    slab_grammar::SlabGrammar,
    tokenizer::{Merge, Tokenizer},
};

pub mod cnf;
pub mod parse;
pub mod slab_grammar;
pub mod tokenizer;

pub struct DenseGrammar<'bump> {
    rules: &'bump [DenseRule<'bump>],
    start: usize,
}

impl<'bump> DenseGrammar<'bump> {
    pub fn recognizes(&self, input: &[u8], tokenizer: &Tokenizer) -> bool {
        self.recognizes_tokens(input.iter().map(|&byte| tokenizer.bytes[byte as usize]))
    }

    pub fn recognizes_tokens(&self, input: impl IntoIterator<Item = u32>) -> bool {
        let mut recognizer = Recognizer::new(self);

        input
            .into_iter()
            .try_fold((), |_, token| match recognizer.push(token) {
                RecognizerState::Valid => ControlFlow::Break(true),
                RecognizerState::Invalid => ControlFlow::Break(false),
                RecognizerState::Incomplete => ControlFlow::Continue(()),
            })
            .break_value()
            .unwrap_or(false)
    }

    pub fn replace_bytes(&self, bytes: &[u8]) -> String {
        let mut result = String::new();
        for byte in bytes {
            if let Some(token) = self.rules.get(*byte as usize) {
                result.push_str(token.to_string().as_str());
            } else {
                result.push(*byte as char);
            }
        }
        result
    }
}

impl<'bump> Display for DenseGrammar<'bump> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "start: s{}", self.start)?;
        for (i, rule) in self.rules.iter().enumerate() {
            writeln!(f, "s{} -> {rule}", i)?;
        }
        Ok(())
    }
}

struct DenseRule<'bump> {
    rhs: &'bump [&'bump [DenseSymbol]],
}

impl<'bump> Display for DenseRule<'bump> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.rhs
                .iter()
                .map(|rhs| rhs
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" "))
                .collect::<Vec<_>>()
                .join("\n\t| ")
        )
    }
}

#[derive(Debug, Clone, Copy, Eq, Hash)]
enum DenseSymbol {
    NonTerminal(usize),
    Terminal(u32),
    Epsilon,
}

impl PartialEq for DenseSymbol {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (DenseSymbol::NonTerminal(id1), DenseSymbol::NonTerminal(id2)) => id1 == id2,
            (DenseSymbol::Terminal(lit1), DenseSymbol::Terminal(lit2)) => lit1 == lit2,
            (DenseSymbol::Epsilon, DenseSymbol::Epsilon) => true,
            _ => false,
        }
    }
}

impl Display for DenseSymbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DenseSymbol::NonTerminal(id) => write!(f, "s{}", id),
            DenseSymbol::Terminal(lit) => write!(f, "'{}'", lit),
            DenseSymbol::Epsilon => write!(f, "ε"),
        }
    }
}

impl Grammar {
    pub fn replace_tokenizer_terminals(&self, tokenizer: &Tokenizer) -> Grammar<u32> {
        let mut lhs_to_u32_map = HashMap::new();
        let mut lhs_to_u32 = |lhs: &str| {
            let len = lhs_to_u32_map.len() as u32;
            *lhs_to_u32_map.entry(lhs.to_string()).or_insert_with(|| len)
        };

        let mut myself = Grammar {
            start: lhs_to_u32(&self.start),
            rules: Vec::new(),
        };

        for rule in &*self.rules {
            let mut new_rhs = Vec::new();
            for rhs in &*rule.rhs {
                let mut new_sequence = Vec::new();
                for symbol in &**rhs {
                    match symbol {
                        parse::Symbol::Terminal(lit) => {
                            let lit = lit.as_bytes()[0] as usize;
                            if let Some(token) = tokenizer.bytes.get(lit) {
                                // Replace the terminal with the corresponding token from the tokenizer
                                new_sequence.push(parse::Symbol::Terminal(*token));
                            } else {
                                panic!("Token {:?} not found in tokenizer vocabulary", lit);
                            }
                        }
                        parse::Symbol::NonTerminal(nt) => {
                            new_sequence.push(parse::Symbol::NonTerminal(lhs_to_u32(nt)));
                        }
                        parse::Symbol::Epsilon => {
                            new_sequence.push(parse::Symbol::Epsilon);
                        }
                    }
                }
                new_rhs.push(new_sequence.into());
            }
            myself.rules.push(parse::Rule {
                lhs: lhs_to_u32(&rule.lhs),
                rhs: new_rhs,
            });
        }

        myself
    }
}

impl Grammar<u32> {
    pub fn reallocate<'bump>(&self, bump: &'bump bumpalo::Bump) -> DenseGrammar<'bump> {
        // Create a mapping from non-terminal names to indices.
        let non_terminal_indices = self
            .rules
            .iter()
            .enumerate()
            .map(|(i, rule)| (rule.lhs.clone(), i))
            .collect::<std::collections::HashMap<_, _>>();

        // Allocate rules in the bump allocator.
        let rules: Vec<DenseRule> = self
            .rules
            .iter()
            .map(|rule| {
                let rhs: Vec<&'bump [DenseSymbol]> = rule
                    .rhs
                    .iter()
                    .map(|seq| {
                        let possibility = seq
                            .iter()
                            .map(|symbol| match symbol {
                                parse::Symbol::NonTerminal(nt) => {
                                    DenseSymbol::NonTerminal(non_terminal_indices[nt])
                                }
                                parse::Symbol::Terminal(lit) => DenseSymbol::Terminal(*lit),
                                parse::Symbol::Epsilon => DenseSymbol::Epsilon,
                            })
                            .collect::<Vec<_>>();

                        bump.alloc(possibility).as_slice()
                    })
                    .collect();

                DenseRule {
                    rhs: bump.alloc(rhs).as_slice(),
                }
            })
            .collect();

        // Find the start rule index.
        let start_index = non_terminal_indices[&self.start];

        DenseGrammar {
            rules: &*bump.alloc(rules),
            start: start_index,
        }
    }
}

#[derive(Clone)]
pub struct Recognizer<'a> {
    grammar: &'a DenseGrammar<'a>,
    chart: Vec<Vec<u32>>,
    positions: Slab<Position<'a>>,
}

impl<'bump> Recognizer<'bump> {
    pub fn new(grammar: &'bump DenseGrammar<'bump>) -> Self {
        let chart: Vec<Vec<u32>> = vec![Vec::new()];

        let start = grammar.start;
        let mut myself = Self {
            grammar,
            chart,
            positions: Slab::new(),
        };

        for rhs in &*grammar.rules[start].rhs {
            let pos = myself.new_position(None, start, *rhs);
            myself.chart[0].push(pos);
        }

        myself.prep_states();

        myself
    }

    pub fn could_become_valid(&self) -> bool {
        // The position could be valid if the last position is non-empty
        self.chart.last().map_or(false, |last| !last.is_empty())
    }

    pub fn push_tokens(&mut self, tokens: impl IntoIterator<Item = u32>) -> RecognizerState {
        let result =
            tokens
                .into_iter()
                .try_fold(RecognizerState::Incomplete, |_, token| {
                    match self.push(token) {
                        RecognizerState::Valid => ControlFlow::Break(RecognizerState::Valid),
                        RecognizerState::Invalid => ControlFlow::Break(RecognizerState::Invalid),
                        RecognizerState::Incomplete => {
                            ControlFlow::Continue(RecognizerState::Incomplete)
                        }
                    }
                });
        match result {
            ControlFlow::Break(state) => state,
            ControlFlow::Continue(state) => state,
        }
    }

    pub fn prep_states(&mut self) -> RecognizerState {
        let k = self.chart.len() - 1;

        // Process each state in the current position
        let mut index = 0;
        while index < self.chart[k].len() {
            let current = self.chart[k][index];
            let Position {
                parent,
                non_terminal,
                rhs,
            } = &self.positions[current as usize];
            index += 1;

            // If the dot is not at the end of the rule, we can either predict or scan
            if let [symbol, remaining @ ..] = rhs {
                match symbol {
                    DenseSymbol::NonTerminal(next_non_terminal) => {
                        // Predictor: Add new states for the non-terminal
                        if self.grammar.rules[*next_non_terminal].rhs.len() > 0 {
                            for next_rhs in self.grammar.rules[*next_non_terminal].rhs {
                                let new =
                                    self.new_position(Some(current), *next_non_terminal, *next_rhs);
                                self.chart[k].push(new);
                            }
                        }
                    }
                    DenseSymbol::Terminal(_) => {}
                    DenseSymbol::Epsilon => {
                        // Epsilon transition, just move the dot forward
                        let pos = self.new_position(*parent, *non_terminal, remaining);
                        self.chart[k].push(pos);
                    }
                }
            } else {
                // Pop this state and move forward in the parent chain
                if let Some(parent_state) = parent {
                    let parent_state = &self.positions[*parent_state as usize];
                    // Completer: If we reach the end of a rule, we can complete it
                    let pos = self.new_position(
                        parent_state.parent,
                        parent_state.non_terminal,
                        &parent_state.rhs[1..],
                    );
                    self.chart[k].push(pos);
                } else {
                    // If there's no parent, this is a completed state
                    if *non_terminal == self.grammar.start && rhs.is_empty() {
                        // If we reached the start rule and the dot is at the end
                        return RecognizerState::Valid;
                    }
                }
            }
        }

        // If we reach here, the input was not recognized
        self.chart.last().map_or(RecognizerState::Invalid, |last| {
            if last.is_empty() {
                RecognizerState::Invalid
            } else {
                RecognizerState::Incomplete
            }
        })
    }

    pub fn push(&mut self, byte: u32) -> RecognizerState {
        let k = self.chart.len() - 1;
        let mut new_positions = Vec::new();

        // Process each state in the current position
        let mut index = 0;
        while index < self.chart[k].len() {
            let current = self.chart[k][index];
            let Position {
                parent,
                non_terminal,
                rhs,
            } = &self.positions[current as usize];
            index += 1;

            // If the dot is not at the end of the rule, we can either predict or scan
            if let [symbol, remaining @ ..] = rhs {
                match symbol {
                    DenseSymbol::Terminal(lit) => {
                        // Scanner: Check if we can match the terminal
                        if byte == *lit {
                            // Add the new state with the terminal matched
                            let pos = self.new_position(*parent, *non_terminal, remaining);
                            new_positions.push(pos);
                        }
                    }
                    _ => {}
                }
            }
        }

        self.chart.push(new_positions);

        self.prep_states()
    }

    pub fn possible_next_terminals(&self) -> FxHashSet<u32> {
        let k = self.chart.len() - 1;
        let mut possible_terminals = FxHashSet::default();

        // Process each state in the current position
        let mut index = 0;
        while index < self.chart[k].len() {
            let current = self.chart[k][index];
            let Position { rhs, .. } = &self.positions[current as usize];
            index += 1;

            // If the dot is not at the end of the rule, we can either predict or scan
            if let [symbol, ..] = rhs {
                match symbol {
                    DenseSymbol::Terminal(lit) => {
                        possible_terminals.insert(*lit);
                    }
                    _ => {}
                }
            }
        }

        possible_terminals
    }

    pub fn pop(&mut self) {
        // Pop the last position from the chart
        if let Some(last) = self.chart.pop() {
            // Remove all positions that were created in this step
            for pos in last {
                self.positions.remove(pos as usize);
            }
        }
    }

    fn new_position(
        &mut self,
        parent: Option<u32>,
        non_terminal: usize,
        rhs: &'bump [DenseSymbol],
    ) -> u32 {
        let pos = Position {
            parent,
            non_terminal,
            rhs,
        };
        self.positions.insert(pos) as u32
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum RecognizerState {
    Incomplete,
    Invalid,
    Valid,
}

impl RecognizerState {
    pub fn is_valid(&self) -> bool {
        matches!(self, RecognizerState::Valid)
    }

    pub fn is_invalid(&self) -> bool {
        matches!(self, RecognizerState::Invalid)
    }

    pub fn is_incomplete(&self) -> bool {
        matches!(self, RecognizerState::Incomplete)
    }

    pub fn could_become_valid(&self) -> bool {
        matches!(self, RecognizerState::Incomplete | RecognizerState::Valid)
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Position<'a> {
    parent: Option<u32>,
    non_terminal: usize,
    rhs: &'a [DenseSymbol],
}

#[test]
fn test_cyk_recognizes() {
    let grammar = parse::Grammar::parse(
        r#"Start -> ntString
ntString -> 'name' | '" "' | '(' 'str.++' ' ' ntString ' ' ntString ')' | '(' 'str.replace' ' ' ntString ' ' ntString ' ' ntString ')' | '(' 'str.at' ' ' ntString ' ' ntInt ')' | '(' 'int.to.str' ' ' ntInt ')' | '(' 'str.substr' ' ' ntString ' ' ntInt ' ' ntInt ')'
ntInt -> '0' | '1' | '2' | '(' '+' ' ' ntInt ' ' ntInt ')' | '(' '-' ' ' ntInt ' ' ntInt ')' | '(' 'str.len' ' ' ntString ')' | '(' 'str.to.int' ' ' ntString ')' | '(' 'str.indexof' ' ' ntString ' ' ntString ' ' ntInt ')'
ntBool -> 'true' | 'false' | '(' 'str.prefixof' ' ' ntString ' ' ntString ')' | '(' 'str.suffixof' ' ' ntString ' ' ntString ')' | '(' 'str.contains' ' ' ntString ' ' ntString ')'
"#,
    )
    .unwrap();

    let grammar = grammar.split_terminals();
    let cnf_grammar = grammar.to_cnf().unwrap();
    let bump = bumpalo::Bump::new();
    let tokenizer = Tokenizer::load_tokenizer("tokenizer.json");
    let cnf_grammar = cnf_grammar.replace_tokenizer_terminals(&tokenizer);
    let dense_grammar = cnf_grammar.reallocate(&bump);
    println!("Dense grammar:\n{}", dense_grammar);

    assert!(dense_grammar.recognizes(br#"name"#, &tokenizer));
    assert!(dense_grammar.recognizes(br#"(str.++ name name)"#, &tokenizer));
    assert!(dense_grammar.recognizes(br#"(str.replace name name name)"#, &tokenizer));
    assert!(dense_grammar.recognizes(br#"(str.at name 0)"#, &tokenizer));
    assert!(dense_grammar.recognizes(br#"(int.to.str 0)"#, &tokenizer));
    assert!(dense_grammar.recognizes(br#"(str.substr name 0 1)"#, &tokenizer));

    assert!(!dense_grammar.recognizes(br#"(str.substr name name 2)"#, &tokenizer));
    assert!(!dense_grammar.recognizes(br#"invalid_input"#, &tokenizer));
}

#[test]
fn test_cyk_pop() {
    let grammar = parse::Grammar::parse(
        r#"Start -> ntString
ntString -> 'name' | '" "' | '(' 'str.++' ' ' ntString ' ' ntString ')' | '(' 'str.replace' ' ' ntString ' ' ntString ' ' ntString ')' | '(' 'str.at' ' ' ntString ' ' ntInt ')' | '(' 'int.to.str' ' ' ntInt ')' | '(' 'str.substr' ' ' ntString ' ' ntInt ' ' ntInt ')'
ntInt -> '0' | '1' | '2' | '(' '+' ' ' ntInt ' ' ntInt ')' | '(' '-' ' ' ntInt ' ' ntInt ')' | '(' 'str.len' ' ' ntString ')' | '(' 'str.to.int' ' ' ntString ')' | '(' 'str.indexof' ' ' ntString ' ' ntString ' ' ntInt ')'
ntBool -> 'true' | 'false' | '(' 'str.prefixof' ' ' ntString ' ' ntString ')' | '(' 'str.suffixof' ' ' ntString ' ' ntString ')' | '(' 'str.contains' ' ' ntString ' ' ntString ')'
"#,
    )
    .unwrap();

    let grammar = grammar.split_terminals();
    let cnf_grammar = grammar.to_cnf().unwrap();
    let bump = bumpalo::Bump::new();
    let tokenizer = Tokenizer::load_tokenizer("tokenizer.json");
    let cnf_grammar = cnf_grammar.replace_tokenizer_terminals(&tokenizer);
    let dense_grammar = cnf_grammar.reallocate(&bump);
    println!("Dense grammar:\n{}", dense_grammar);

    let sequences: [(bool, &'static [u8]); 8] = [
        (true, br#"name"#),
        (true, br#"(str.++ name name)"#),
        (true, br#"(str.replace name name name)"#),
        (true, br#"(str.at name 0)"#),
        (true, br#"(int.to.str 0)"#),
        (true, br#"(str.substr name 0 1)"#),
        (false, br#"(str.substr name name 2)"#),
        (false, br#"invalid_input"#),
    ];

    for _ in 0..100 {
        for sequence in &sequences {
            let (expected, input) = *sequence;
            let mut recognizer = Recognizer::new(&dense_grammar);

            let mut position = 0;
            let mut state = RecognizerState::Incomplete;
            while position < input.len() {
                let len = rand::random::<u32>() % 10;
                for _ in 0..len {
                    let byte = rand::random::<u8>();
                    let token = tokenizer.bytes.get(byte as usize).cloned().unwrap_or(0);
                    recognizer.push(token);
                }
                for _ in 0..len {
                    recognizer.pop();
                }
                let token = input[position] as u32;
                let token = tokenizer.bytes[token as usize];
                state = recognizer.push(token);
                position += 1;
            }

            match state {
                RecognizerState::Valid => {
                    assert!(expected, "Expected valid for input: {:?}", input)
                }
                RecognizerState::Invalid => {
                    assert!(!expected, "Expected invalid for input: {:?}", input)
                }
                RecognizerState::Incomplete => {
                    panic!("Unexpected incomplete state for input: {:?}", input)
                }
            }
        }
    }
}
