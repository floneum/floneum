use std::fmt::Display;

use comfy_table::Table;

use crate::parse::Grammar;

mod cnf;
mod parse;

fn main() {
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
    println!("dense size: {}", bump.allocated_bytes());

    loop {
        let mut input = String::new();
        println!("Enter a string to test (or 'exit' to quit):");
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input == "exit" {
            break;
        }

        if dense_grammar.recognizes(input.as_bytes()) {
            println!("The grammar recognizes the input: '{}'", input);
        } else {
            println!("The grammar does NOT recognize the input: '{}'", input);
        }
    }
}

struct DenseGrammar<'bump> {
    rules: &'bump [DenseRule<'bump>],
    start: usize,
}

use std::collections::HashMap;

impl<'bump> DenseGrammar<'bump> {
    pub fn recognizes(&self, input: &[u8]) -> bool {
        let bump = bumpalo::Bump::new();
        let n = input.len();

        let mut terminal_map: HashMap<u8, Vec<usize>> = HashMap::new();
        let mut binary_map: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

        for (lhs, rule) in self.rules.iter().enumerate() {
            for alt in rule.rhs.iter() {
                match *alt {
                    [DenseSymbol::Terminal(lit)] if lit.len() == 1 => {
                        terminal_map.entry(lit.as_bytes()[0]).or_default().push(lhs);
                    }
                    [DenseSymbol::NonTerminal(b), DenseSymbol::NonTerminal(c)] => {
                        binary_map.entry((*b, *c)).or_default().push(lhs);
                    }
                    [DenseSymbol::Epsilon] => {}
                    _ => unreachable!("{:?}", alt),
                }
            }
        }

        #[cfg(debug_assertions)]
        for value in terminal_map.values_mut() {
            assert!(value.is_sorted());
        }

        let table = bump.alloc_slice_fill_with(n, |_| bump.alloc_slice_fill_with(n, |_| bumpalo::collections::Vec::new_in(&bump)));

        for (idx, &byte) in input.iter().enumerate() {
            if let Some(vars) = terminal_map.get(&byte) {
                table[idx][idx].extend(vars);
            }
        }

        for span in 2..=n {
            for i in 0..=n - span {
                let j = i + span - 1;
                for k in i..j {
                    for b in bump.alloc_slice_copy(table[i][k].as_slice()).iter().copied() {
                        for c in bump.alloc_slice_copy(table[k + 1][j].as_slice()).iter().copied() {
                            if let Some(vars) = binary_map.get(&(b, c)) {
                                let original = &mut table[i][j];
                                // Join sorted lists
                                let mut start_index = 0;
                                for new in vars {
                                    match original[start_index..]
                                        .binary_search(&new)
                                    {
                                        Ok(pos) => start_index += pos + 1, // Move past the found element
                                        Err(pos) => {
                                            original.insert(pos + start_index, *new);
                                            start_index += pos + 1; // Move past the inserted element
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        println!("CYK table:");
        let mut pretty_table = Table::new();

        for row in &*table {
            let row_str = row
                .iter()
                .map(|set| {
                    if set.is_empty() {
                        "∅".to_string()
                    } else {
                        set.iter()
                            .map(|&x| format!("s{}", x))
                            .collect::<Vec<_>>()
                            .join(", ")
                    }
                })
                .collect::<Vec<_>>();
            pretty_table.add_row(row_str);
        }
        println!("{pretty_table}");

        table[0][n - 1].contains(&self.start)
    }
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
