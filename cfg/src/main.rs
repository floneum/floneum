use std::fmt::Display;

use crate::parse::Grammar;

mod cnf;
mod parse;

struct CykSolutionParser {}

fn main() {
    let input = r#"Start -> ntString
ntString -> name | '"' ' ' '"' | '(' 'str.++'      ntString ntString ')' | '(' 'str.replace' ntString ntString ntString ')' | '(' 'str.at'      ntString ntInt    ')' | '(' 'int.to.str'  ntInt             ')' | '(' 'str.substr'  ntString ntInt ntInt ')'
ntInt -> '0' | '1' | '2' | '(' '+'            ntInt ntInt ')' | '(' '-'            ntInt ntInt ')' | '(' 'str.len'      ntString    ')' | '(' 'str.to.int'   ntString    ')' | '(' 'str.indexof'  ntString ntString ntInt ')'
ntBool -> 'true' | 'false' | '(' 'str.prefixof' ntString ntString ')' | '(' 'str.suffixof' ntString ntString ')' | '(' 'str.contains' ntString ntString ')'
"#;

    let grammar = parse::Grammar::parse(input).unwrap();
    println!("Parsed grammar:\n{}", grammar);

    let cnf_grammar = grammar.to_cnf().unwrap();
    println!("Converted to CNF:\n{}", cnf_grammar);

    let bump = bumpalo::Bump::new();
    let dense_grammar = cnf_grammar.reallocate(&bump);
    println!("Reallocated grammar:\n{}", dense_grammar);
    println!("dense size: {}", bump.allocated_bytes());
}

struct DenseGrammar<'bump> {
    rules: &'bump [DenseRule<'bump>],
    start: usize,
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
                        let possibility = seq.iter().map(|symbol| match symbol {
                            parse::Symbol::NonTerminal(nt) => {
                                DenseSymbol::NonTerminal(non_terminal_indices[nt])
                            }
                            parse::Symbol::Terminal(lit) => {
                                DenseSymbol::Terminal(bump.alloc_str(lit.as_str()))
                            }
                            parse::Symbol::Epsilon => DenseSymbol::Epsilon,
                        }).collect::<Vec<_>>();

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
