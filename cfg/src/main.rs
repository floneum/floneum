use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    hash::Hash,
};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    parse::Grammar,
    slab_grammar::SlabGrammar,
    tokenizer::{Merge, Tokenizer},
};

mod cnf;
mod parse;
mod slab_grammar;
mod tokenizer;

fn main() {
    let tokenizer = Tokenizer::load_tokenizer("tokenizer.json");

    let grammar = parse::Grammar::parse(
        r#"Start -> ntInt
ntString -> 'name' | '" "' | '(' 'str.++' ' ' ntString ' ' ntString ')' | '(' 'str.replace' ' ' ntString ' ' ntString ' ' ntString ')' | '(' 'str.at' ' ' ntString ' ' ntInt ')' | '(' 'int.to.str' ' ' ntInt ')' | '(' 'str.substr' ' ' ntString ' ' ntInt ' ' ntInt ')'
ntInt -> '0' | '1' | '2' | '(' '+' ' ' ntInt ' ' ntInt ')' | '(' '-' ' ' ntInt ' ' ntInt ')' | '(' 'str.len' ' ' ntString ')' | '(' 'str.to.int' ' ' ntString ')' | '(' 'str.indexof' ' ' ntString ' ' ntString ' ' ntInt ')'
ntBool -> 'true' | 'false' | '(' 'str.prefixof' ' ' ntString ' ' ntString ')' | '(' 'str.suffixof' ' ' ntString ' ' ntString ')' | '(' 'str.contains' ' ' ntString ' ' ntString ')'
"#,
    )
    .unwrap();

    let grammar = grammar.split_terminals();
    let bump = bumpalo::Bump::new();
    let grammar = grammar.replace_tokenizer_terminals(&tokenizer);
    let mut grammar = SlabGrammar::new(&grammar);
    grammar.garbage_collect_non_terminals();
    println!("start rule count: {}", grammar.rules.len());
    let merges = &tokenizer.merges;
    let mut last_size = grammar.rules.len();
    for (i, merge) in merges.iter().enumerate() {
        println!(
            "Applying merge {i}: {:?} + {:?} = {:?}",
            String::from_utf8_lossy(&tokenizer.inverse_vocab[&merge.pair[0]]),
            String::from_utf8_lossy(&tokenizer.inverse_vocab[&merge.pair[1]]),
            String::from_utf8_lossy(&tokenizer.inverse_vocab[&merge.new_token])
        );
        let start = std::time::Instant::now();
        grammar.shortcut_merge(merge);
        println!("Time to merge: {:?}", start.elapsed());
        println!("size: {}", grammar.rules.len());
        println!(
            "grew by a factor of {:.10}",
            grammar.rules.len() as f64 / last_size as f64
        );
        last_size = grammar.rules.len();
        {
            let grammar = grammar.to_grammar();
            println!(
                "grammar:\n{}",
                grammar.clone().map(
                    |r| String::from_utf8_lossy(&tokenizer.inverse_vocab[&r]).to_string(),
                    |r| r.to_string()
                )
            );
            let dense_grammar = grammar.reallocate(&bump);
            println!("dense size: {}", bump.allocated_bytes());
            println!("after shortcut merge rule count: {}", grammar.rules.len());

            assert!(dense_grammar.recognizes(b"0", &tokenizer));
            assert!(dense_grammar.recognizes(b"1", &tokenizer));
            assert!(dense_grammar.recognizes(b"2", &tokenizer));
            assert!(dense_grammar.recognizes(b"(+ 1 2)", &tokenizer));
            assert!(dense_grammar.recognizes(b"(- 2 1)", &tokenizer));
            assert!(dense_grammar.recognizes(b"(str.len name)", &tokenizer));
        }
    }
    let grammar = grammar.to_grammar();
    let dense_grammar = grammar.reallocate(&bump);
    println!("dense size: {}", bump.allocated_bytes());
    println!("after shortcut merge rule count: {}", grammar.rules.len());

    assert!(dense_grammar.recognizes(b"0", &tokenizer));
    assert!(dense_grammar.recognizes(b"1", &tokenizer));
    assert!(dense_grammar.recognizes(b"2", &tokenizer));
    assert!(dense_grammar.recognizes(b"(+ 1 2)", &tokenizer));
    assert!(dense_grammar.recognizes(b"(- 2 1)", &tokenizer));
    assert!(dense_grammar.recognizes(b"(str.len name)", &tokenizer));
    // The unmerged text should no longer be recognized
    assert!(!dense_grammar.recognizes(b"(str.to.int name)", &tokenizer));
    // But if you merge the `t` and `o` tokens, it should be recognized
    let mut tokens = b"(str."
        .iter()
        .map(|b| tokenizer.bytes[*b as usize])
        .collect::<Vec<_>>();
    tokens.push(10_000); // The merged token
    tokens.extend(b".int name)".iter().map(|b| tokenizer.bytes[*b as usize]));
    assert!(dense_grammar.recognizes_tokens(&tokens));
}

struct DenseGrammar<'bump> {
    rules: &'bump [DenseRule<'bump>],
    start: usize,
}

impl<'bump> DenseGrammar<'bump> {
    pub fn recognizes(&self, input: &[u8], tokenizer: &Tokenizer) -> bool {
        let bump = bumpalo::Bump::new();
        let mut recognizer = Recognizer::new(self, &bump);

        for byte in input {
            recognizer.push_byte(tokenizer.bytes[*byte as usize]);
        }
        recognizer.finish()
    }

    pub fn recognizes_tokens(&self, input: &[u32]) -> bool {
        let bump = bumpalo::Bump::new();
        let mut recognizer = Recognizer::new(self, &bump);

        for token in input {
            recognizer.push_byte(*token);
        }
        recognizer.finish()
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

#[derive(Debug)]
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
    fn reallocate<'bump>(&self, bump: &'bump bumpalo::Bump) -> DenseGrammar<'bump> {
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

struct Recognizer<'bump> {
    grammar: &'bump DenseGrammar<'bump>,
    chart: Vec<Vec<&'bump Position<'bump>>>,
    bump: &'bump bumpalo::Bump,
}

impl<'bump> Recognizer<'bump> {
    pub fn new(grammar: &'bump DenseGrammar<'bump>, bump: &'bump bumpalo::Bump) -> Self {
        let mut chart: Vec<Vec<&'bump Position<'bump>>> = vec![Vec::new()];

        let start = grammar.start;
        for rhs in &*grammar.rules[start].rhs {
            chart[0].push(bump.alloc(Position {
                parent: None, // No parent for the initial state
                non_terminal: start,
                position: 0,
                rhs,
            }));
        }

        Recognizer {
            grammar,
            chart,
            bump,
        }
    }

    fn could_become_valid(&self) -> bool {
        // The position could be valid if the last position is non-empty
        self.chart.last().map_or(false, |last| !last.is_empty())
    }

    fn push_byte(&mut self, byte: u32) -> bool {
        // Push the byte into the chart and process it
        self.push(Some(byte))
    }

    fn finish(&mut self) -> bool {
        // Process the final state with no byte
        self.push(None)
    }

    fn push(&mut self, byte: Option<u32>) -> bool {
        let k = self.chart.len() - 1;
        if byte.is_some() {
            self.chart.push(Vec::new());
        }

        // Process each state in the current position
        let mut index = 0;
        while index < self.chart[k].len() {
            let current = self.chart[k][index];
            let Position {
                parent,
                non_terminal,
                position,
                rhs,
            } = current;
            index += 1;

            // If the dot is not at the end of the rule, we can either predict or scan
            if let Some(symbol) = rhs.get(*position) {
                match symbol {
                    DenseSymbol::NonTerminal(next_non_terminal) => {
                        // Predictor: Add new states for the non-terminal
                        if self.grammar.rules[*next_non_terminal].rhs.len() > 0 {
                            for next_rhs in self.grammar.rules[*next_non_terminal].rhs {
                                let new = self.bump.alloc(Position {
                                    parent: Some(current),
                                    non_terminal: *next_non_terminal,
                                    position: 0,
                                    rhs: next_rhs,
                                });
                                self.chart[k].push(new);
                            }
                        }
                    }
                    DenseSymbol::Terminal(lit) => {
                        // Scanner: Check if we can match the terminal
                        if byte == Some(*lit as _) {
                            // Add the new state with the terminal matched
                            self.chart[k + 1].push(self.bump.alloc(Position {
                                parent: *parent,
                                non_terminal: *non_terminal,
                                position: position + 1,
                                rhs,
                            }));
                        }
                    }
                    DenseSymbol::Epsilon => {
                        // Epsilon transition, just move the dot forward
                        self.chart[k].push(self.bump.alloc(Position {
                            parent: *parent,
                            non_terminal: *non_terminal,
                            position: position + 1,
                            rhs,
                        }));
                    }
                }
            } else {
                // Pop this state and move forward in the parent chain
                if let Some(parent_state) = *parent {
                    // Completer: If we reach the end of a rule, we can complete it
                    self.chart[k].push(self.bump.alloc(Position {
                        parent: parent_state.parent,
                        non_terminal: parent_state.non_terminal,
                        position: parent_state.position + 1,
                        rhs: parent_state.rhs,
                    }));
                } else {
                    // If there's no parent, this is a completed state
                    if *non_terminal == self.grammar.start && *position == rhs.len() {
                        // If we reached the start rule and the dot is at the end
                        return true; // Input recognized
                    }
                }
            }
        }

        // If we reach here, the input was not recognized
        false
    }
}

#[derive(Debug, Clone, Copy)]
struct Position<'bump> {
    parent: Option<&'bump Position<'bump>>,
    non_terminal: usize,
    position: usize,
    rhs: &'bump [DenseSymbol],
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
