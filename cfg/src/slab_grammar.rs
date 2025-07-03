use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    parse::{self, Grammar, Rule, Symbol},
    tokenizer::{Merge, Tokenizer},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
enum State {
    // Before seeing the first token or the merged token
    Start = 0,
    // After seeing the first token
    AfterFirstToken = 1,
    // After seeing the merged token
    AfterMergedToken = 2,
}

impl State {
    const ALL: [State; 3] = [
        State::Start,
        State::AfterFirstToken,
        State::AfterMergedToken,
    ];

    fn is_start(&self) -> bool {
        matches!(self, State::Start)
    }
}

struct ReachabilityCache {
    non_terminal_to_possible_first_tokens: FxHashMap<u32, FxHashSet<u32>>,
    non_terminal_to_possible_last_tokens: FxHashMap<u32, FxHashSet<u32>>,
}

#[derive(Debug, Clone)]
pub struct SlabGrammar {
    pub start: u32,
    pub rules: slab::Slab<Rule<u32>>,
}

impl SlabGrammar {
    pub fn new(grammar: &Grammar<u32>) -> Self {
        let mut map = FxHashMap::default();
        for (i, rule) in grammar.rules.iter().enumerate() {
            map.insert(rule.lhs, i as u32);
        }
        let mut rules = slab::Slab::new();

        let mapped_non_terminals = grammar.clone().map(|x| map[&x], |x| x);

        let start = mapped_non_terminals.start;

        for rule in mapped_non_terminals.rules {
            rules.insert(rule);
        }

        Self { start, rules }
    }

    fn remove(&mut self, id: u32) {
        self.rules.remove(id as usize);
    }

    fn insert(&mut self, rhs: Vec<Vec<Symbol<u32>>>) -> u32 {
        let entry = self.rules.vacant_entry();
        let lhs = entry.key() as u32;
        entry.insert(Rule { lhs, rhs });
        lhs
    }

    pub fn shortcut_merge(&mut self, merge: &Merge) {
        let mut state_mapping = FxHashMap::default();

        let mut stack = vec![(self.start, State::Start, None)];

        while let Some((nt, state, parent)) = stack.pop() {
            
        }
    }

    pub fn garbage_collect_non_terminals(&mut self) {
        // Remove any non-terminals that are not used in any rules
        let mut used_non_terminals = FxHashSet::default();
        let mut queue = vec![self.start.clone()];
        while let Some(nt) = queue.pop() {
            if used_non_terminals.insert(nt.clone()) {
                for (_, rule) in &self.rules {
                    if rule.lhs == nt {
                        for rhs in &rule.rhs {
                            for symbol in rhs {
                                if let parse::Symbol::NonTerminal(non_terminal) = symbol {
                                    if !used_non_terminals.contains(non_terminal) {
                                        queue.push(non_terminal.clone());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        self.rules
            .retain(|_, rule| used_non_terminals.contains(&rule.lhs));
    }

    pub fn to_grammar(&self) -> Grammar<u32> {
        let mut rules = Vec::new();
        for (_, rule) in &self.rules {
            rules.push(rule.clone());
        }
        Grammar {
            start: self.start,
            rules,
        }
    }
}

#[test]
fn test_enforce_merge() {
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
    let cnf_grammar = grammar.to_cnf().unwrap();
    let bump = bumpalo::Bump::new();
    let cnf_grammar = cnf_grammar.replace_tokenizer_terminals(&tokenizer);
    let mut cnf_grammar = SlabGrammar::new(&cnf_grammar);
    cnf_grammar.shortcut_merge(&Merge {
        rank: 0,
        pair: [
            tokenizer.bytes[b't' as usize],
            tokenizer.bytes[b'o' as usize],
        ],
        new_token: 10_000,
    });
    cnf_grammar.garbage_collect_non_terminals();
    let cnf_grammar = cnf_grammar.to_grammar();
    println!("CNF grammar:\n{}", cnf_grammar);
    let dense_grammar = cnf_grammar.reallocate(&bump);
    println!("dense size: {}", bump.allocated_bytes());

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
