use std::collections::VecDeque;

use beef::Cow;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

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

    const ALL_BITSET: u8 = 0b111;

    fn is_start(&self) -> bool {
        matches!(self, State::Start)
    }

    fn reachable_states(&self, merge: &Merge, token: u32, mut f: impl FnMut(State, Option<u32>)) {
        match self {
            State::Start => {
                if token == merge.pair[0] {
                    f(State::AfterFirstToken, Some(token));
                    f(State::AfterMergedToken, Some(merge.new_token));
                } else {
                    f(State::Start, Some(token));
                }
            }
            State::AfterFirstToken => {
                if token != merge.pair[1] {
                    f(State::Start, Some(token));
                }
            }
            State::AfterMergedToken => {
                if token == merge.pair[1] {
                    f(State::Start, None);
                }
            }
        }
    }

    const fn to_bitset(&self) -> u8 {
        1 << *self as u8
    }

    fn from_bitset(bitset: u8) -> smallvec::SmallVec<[State; 3]> {
        State::ALL
            .iter()
            .filter(|&&state| (bitset & state.to_bitset()) != 0)
            .cloned()
            .collect()
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

    fn insert(&mut self, rhs: Vec<Cow<'static, [Symbol<u32>]>>) -> u32 {
        let entry = self.rules.vacant_entry();
        let lhs = entry.key() as u32;
        entry.insert(Rule { lhs, rhs });
        lhs
    }

    // a -> b
    // b -> b1 | c | 2
    // c -> b
    pub fn shortcut_merge(&mut self, merge: &Merge) {
        let mut queued = VecDeque::new();
        queued.push_back((self.start, 0));
        let mut depth = FxHashMap::default();
        while let Some((nt, current_depth)) = queued.pop_front() {
            if depth.contains_key(&nt) {
                continue;
            }
            depth.insert(nt, current_depth);
            queued.extend(self.rules[nt as usize].rhs.iter().flat_map(|rules| {
                rules.iter().filter_map(|rule| {
                    if let Symbol::NonTerminal(lhs) = rule {
                        Some((*lhs, current_depth + 1))
                    } else {
                        None
                    }
                })
            }));
        }

        let mut rules_sorted = depth.keys().cloned().collect::<Vec<_>>();
        rules_sorted.sort_unstable_by_key(|&nt| depth[&nt]);

        let mut reachable_states: FxHashMap<(State, u32), u8> = FxHashMap::default();

        fn count_reachable_states(reachable_states: &FxHashMap<(State, u32), u8>) -> u32 {
            reachable_states
                .values()
                .copied()
                .map(|s| s.count_ones())
                .sum()
        }

        let mut prev_reachable_states = count_reachable_states(&reachable_states);

        loop {
            for nt in rules_sorted.iter().rev().copied() {
                let options = &self.rules[nt as usize].rhs;
                for rules in options {
                    for possible_start_state in State::ALL {
                        // The start state only ever starts from the start state
                        if nt == self.start && !possible_start_state.is_start() {
                            continue;
                        }
                        let mut current_states = possible_start_state.to_bitset();
                        for symbol in rules.iter() {
                            let mut new_states = 0;
                            match symbol {
                                Symbol::NonTerminal(next_nt) => {
                                    for current_state in State::from_bitset(current_states) {
                                        let bitset = reachable_states
                                            .get(&(current_state, *next_nt))
                                            .copied()
                                            .unwrap_or(State::ALL_BITSET);
                                        new_states |= bitset;
                                    }
                                    current_states = new_states;
                                }
                                Symbol::Terminal(token) => {
                                    for current_state in State::from_bitset(current_states) {
                                        current_state.reachable_states(
                                            merge,
                                            *token,
                                            |next_state, _| {
                                                new_states |= next_state.to_bitset();
                                            },
                                        );
                                    }
                                    current_states = new_states;
                                }
                                Symbol::Epsilon => {}
                            }
                        }
                        reachable_states.insert((possible_start_state, nt), current_states);
                    }
                }
            }
            let new_reachable_states = count_reachable_states(&reachable_states);
            println!(
                "reachable states: {}->{}",
                prev_reachable_states, new_reachable_states
            );
            if new_reachable_states == prev_reachable_states {
                break;
            }
            prev_reachable_states = new_reachable_states;
        }

        // First transpose the map into nt -> Vec<(State, u8)>
        let non_terminal_to_states: FxHashMap<u32, SmallVec<[(State, u8); 3]>> = reachable_states
            .into_iter()
            .fold(FxHashMap::default(), |mut acc, ((state, nt), bitset)| {
                acc.entry(nt).or_default().push((state, bitset));
                acc
            });

        let mut transition_map: FxHashMap<(u32, State, State), u32> = FxHashMap::default();

        // First insert empty entries for all non-terminals
        for (nt, new_states) in &non_terminal_to_states {
            let transitions: SmallVec<[(State, State); 9]> = new_states
                .iter()
                .flat_map(|(state, bitset)| {
                    State::from_bitset(*bitset)
                        .into_iter()
                        .map(move |s| (s, *state))
                })
                .collect();

            let last_index = transitions.len() - 1;
            for (i, (start, end)) in transitions.iter().enumerate() {
                let key = (*nt, *start, *end);
                if i == last_index {
                    transition_map.insert(key, *nt);
                } else {
                    transition_map.insert(key, self.insert(Vec::new()));
                }
            }
        }

        // Then fill in the transitions
        for (nt, new_states) in &non_terminal_to_states {
            let transitions: SmallVec<[(State, State); 9]> = new_states
                .iter()
                .flat_map(|(state, bitset)| {
                    State::from_bitset(*bitset)
                        .into_iter()
                        .map(move |s| (s, *state))
                })
                .collect();

            for (start, end) in transitions {
                let options = &self.rules[*nt as usize].rhs;
                let mut new_options: Vec<Cow<[Symbol<u32>]>> = vec![];
                for rules in options {
                    let mut possible_rules = vec![];
                    for symbol in rules.iter() {
                        match symbol {
                            Symbol::NonTerminal(next_nt) => {
                                if possible_rules.is_empty() {
                                    for next in State::ALL {
                                        if let Some(nt) =
                                            transition_map.get(&(*next_nt, start, next))
                                        {
                                            possible_rules
                                                .push((start, vec![Symbol::NonTerminal(*nt)]));
                                        }
                                    }
                                } else {
                                    possible_rules = possible_rules
                                        .into_iter()
                                        .flat_map(|(state, symbols)| {
                                            let mut new = Vec::new();
                                            for next in State::ALL {
                                                if let Some(nt) =
                                                    transition_map.get(&(*next_nt, state, next))
                                                {
                                                    let mut new_symbols = symbols.clone();
                                                    new_symbols.push(Symbol::NonTerminal(*nt));
                                                    new.push((next, new_symbols));
                                                }
                                            }
                                            new
                                        })
                                        .collect();
                                }
                            }
                            Symbol::Terminal(token) => {
                                if possible_rules.is_empty() {
                                    possible_rules.push((start, vec![Symbol::Terminal(*token)]));
                                } else {
                                    possible_rules = possible_rules
                                        .into_iter()
                                        .flat_map(|(state, symbols)| {
                                            let mut new = Vec::new();
                                            state.reachable_states(
                                                merge,
                                                *token,
                                                |next_state, token| {
                                                    let mut new_symbols = symbols.clone();
                                                    if let Some(token) = token {
                                                        new_symbols.push(Symbol::Terminal(token));
                                                    }
                                                    new.push((next_state, new_symbols));
                                                },
                                            );
                                            new
                                        })
                                        .collect();
                                }
                            }
                            Symbol::Epsilon => {}
                        }
                    }
                    new_options.extend(
                        possible_rules
                            .into_iter()
                            .filter_map(|(state, symbols)| (state == end).then(|| symbols.into())),
                    );
                }
                let id = transition_map[&(*nt, start, end)];
                self.rules[id as usize].rhs = new_options;
            }
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
                            for symbol in &**rhs {
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
