use std::fmt::Display;

use crate::parse::Grammar;

mod cnf;
mod parse;

fn main() {
    let grammar = parse::Grammar::parse(
        r#"Start -> ntInt
ntString -> 'name' | '" "' | '(' 'str.++' ' ' ntString ' ' ntString ')' | '(' 'str.replace' ' ' ntString ' ' ntString ' ' ntString ')' | '(' 'str.at' ' ' ntString ' ' ntInt ')' | '(' 'int.to.str' ' ' ntInt ')' | '(' 'str.substr' ' ' ntString ' ' ntInt ' ' ntInt ')'
ntInt -> '0' | '1' | '2' | '(' '+' ' ' ntInt ' ' ntInt ')' | '(' '-' ' ' ntInt ' ' ntInt ')' | '(' 'str.len' ' ' ntString ' ' ')' | '(' 'str.to.int' ' ' ntString ' ' ')' | '(' 'str.indexof' ' ' ntString ' ' ntString ' ' ntInt ')'
ntBool -> 'true' | 'false' | '(' 'str.prefixof' ' ' ntString ' ' ntString ')' | '(' 'str.suffixof' ' ' ntString ' ' ntString ')' | '(' 'str.contains' ' ' ntString ' ' ntString ')'
"#,
    )
    .unwrap();

    let cnf_grammar = grammar.to_cnf().unwrap();
    let bump = bumpalo::Bump::new();
    let dense_grammar = cnf_grammar.reallocate(&bump);
    println!("dense size: {}", bump.allocated_bytes());

    let mut recognizer = Recognizer::new(&dense_grammar, &bump);
    let mut text = String::new();
    loop {
        let mut new_text = String::new();
        std::io::stdin().read_line(&mut new_text).unwrap();
        let new_text = new_text.trim_end_matches('\n');
        for byte in new_text.as_bytes() {
            text.push(*byte as char);
            if recognizer.push_byte(*byte) || recognizer.finish() {
                println!("Input {:?} is accepted", text);
                break;
            }
            if !recognizer.could_become_valid() {
                println!("Input {:?} could never become valid", text);
                break;
            }
        }
    }
}

struct DenseGrammar<'bump> {
    rules: &'bump [DenseRule<'bump>],
    start: usize,
}

impl<'bump> DenseGrammar<'bump> {
    pub fn recognizes(&self, input: &[u8]) -> bool {
        let bump = bumpalo::Bump::new();
        let mut recognizer = Recognizer::new(self, &bump);

        for byte in input {
            recognizer.push_byte(*byte);
        }
        recognizer.finish()
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
    rhs: &'bump [&'bump [DenseSymbol<'bump>]],
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
enum DenseSymbol<'bump> {
    NonTerminal(usize),
    Terminal(&'bump str),
    Epsilon,
}

impl PartialEq for DenseSymbol<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (DenseSymbol::NonTerminal(id1), DenseSymbol::NonTerminal(id2)) => id1 == id2,
            (DenseSymbol::Terminal(lit1), DenseSymbol::Terminal(lit2)) => lit1 == lit2,
            (DenseSymbol::Epsilon, DenseSymbol::Epsilon) => true,
            _ => false,
        }
    }
}

impl<'bump> Display for DenseSymbol<'bump> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DenseSymbol::NonTerminal(id) => write!(f, "s{}", id),
            DenseSymbol::Terminal(lit) => write!(f, "'{}'", lit),
            DenseSymbol::Epsilon => write!(f, "ε"),
        }
    }
}

impl Grammar {
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
                                parse::Symbol::Terminal(lit) => {
                                    DenseSymbol::Terminal(bump.alloc_str(lit.as_str()))
                                }
                                parse::Symbol::Epsilon => DenseSymbol::Epsilon,
                            })
                            .collect::<Vec<_>>();

                        bump.alloc(possibility).as_slice()
                    })
                    .collect();

                DenseRule {
                    rhs: bump.alloc(rhs),
                }
            })
            .collect();

        // Find the start rule index.
        let start_index = non_terminal_indices[&self.start];

        DenseGrammar {
            rules: bump.alloc(rules),
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
        for rhs in grammar.rules[start].rhs {
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

    fn push_byte(&mut self, byte: u8) -> bool {
        // Push the byte into the chart and process it
        self.push(Some(byte))
    }

    fn finish(&mut self) -> bool {
        // Process the final state with no byte
        self.push(None)
    }

    fn push(&mut self, byte: Option<u8>) -> bool {
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
                        if byte == lit.as_bytes().first().copied() {
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
                    println!(
                        "Reached a completed state without a parent, non-terminal: {}, position: {}, rhs: {:?}",
                        non_terminal, position, rhs
                    );
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
    rhs: &'bump [DenseSymbol<'bump>],
}

#[test]
fn test_cyk_recognizes() {
    let grammar = parse::Grammar::parse(
        r#"Start -> ntString
ntString -> 'name' | '" "' | '(' 'str.++' ' ' ntString ' ' ntString ')' | '(' 'str.replace' ' ' ntString ' ' ntString ' ' ntString ')' | '(' 'str.at' ' ' ntString ' ' ntInt ')' | '(' 'int.to.str' ' ' ntInt ')' | '(' 'str.substr' ' ' ntString ' ' ntInt ' ' ntInt ')'
ntInt -> '0' | '1' | '2' | '(' '+' ' ' ntInt ' ' ntInt ')' | '(' '-' ' ' ntInt ' ' ntInt ')' | '(' 'str.len' ' ' ntString ' ' ')' | '(' 'str.to.int' ' ' ntString ' ' ')' | '(' 'str.indexof' ' ' ntString ' ' ntString ' ' ntInt ')'
ntBool -> 'true' | 'false' | '(' 'str.prefixof' ' ' ntString ' ' ntString ')' | '(' 'str.suffixof' ' ' ntString ' ' ntString ')' | '(' 'str.contains' ' ' ntString ' ' ntString ')'
"#,
    )
    .unwrap();

    let cnf_grammar = grammar.to_cnf().unwrap();
    let bump = bumpalo::Bump::new();
    let dense_grammar = cnf_grammar.reallocate(&bump);
    println!("Dense grammar:\n{}", dense_grammar);

    assert!(dense_grammar.recognizes(br#"name"#));
    assert!(dense_grammar.recognizes(br#"(str.++ name name)"#));
    assert!(dense_grammar.recognizes(br#"(str.replace name name name)"#));
    assert!(dense_grammar.recognizes(br#"(str.at name 0)"#));
    assert!(dense_grammar.recognizes(br#"(int.to.str 0)"#));
    assert!(dense_grammar.recognizes(br#"(str.substr name 0 1)"#));

    assert!(!dense_grammar.recognizes(br#"(str.substr name name 2)"#));
    assert!(!dense_grammar.recognizes(br#"invalid_input"#));
}
