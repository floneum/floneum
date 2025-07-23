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
    const NONE_BITSET: u8 = 0b000;

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
    pub terminals_present: FxHashSet<u32>,
}

impl SlabGrammar {
    pub fn new(grammar: &Grammar<u32>) -> Self {
        let mut map = FxHashMap::default();
        for (i, rule) in grammar.rules.iter().enumerate() {
            map.insert(rule.lhs, i as u32);
        }
        let mut rules = slab::Slab::new();

        let mapped_non_terminals = grammar.clone().map(|x| x, |x| map[&x]);

        let start = mapped_non_terminals.start;

        for rule in mapped_non_terminals.rules {
            rules.insert(rule);
        }

        let mut terminals_present = FxHashSet::default();

        // Collect all terminals present in the grammar
        for (_, rule) in rules.iter() {
            for rhs in &rule.rhs {
                for symbol in &**rhs {
                    if let Symbol::Terminal(token) = symbol {
                        terminals_present.insert(*token);
                    }
                }
            }
        }

        Self {
            start,
            rules,
            terminals_present,
        }
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
    pub fn shortcut_merge(&mut self, merge: &Merge) -> bool {
        // If neither of the pair tokens are present in the grammar, we can skip the merge
        if !self.terminals_present.contains(&merge.pair[0])
            || !self.terminals_present.contains(&merge.pair[1])
        {
            println!("No tokens to merge, skipping merge");
            return false;
        }

        let start_time = std::time::Instant::now();
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
        let mut start_states: FxHashMap<u32, u8> = FxHashMap::default();
        // Insert all start states for each non-terminal
        for nt in rules_sorted.iter() {
            start_states.insert(*nt, 0);
        }
        start_states.insert(self.start, State::Start.to_bitset());

        fn count_reachable_states(reachable_states: &FxHashMap<(State, u32), u8>) -> u32 {
            reachable_states
                .values()
                .copied()
                .map(|s| s.count_ones())
                .sum()
        }
        fn count_start_states(start_states: &FxHashMap<u32, u8>) -> u32 {
            start_states.values().copied().map(|s| s.count_ones()).sum()
        }

        let mut prev_reachable_states = count_reachable_states(&reachable_states);
        let mut prev_start_states = count_start_states(&start_states);

        loop {
            for nt in rules_sorted.iter().copied() {
                let options = &self.rules[nt as usize].rhs;
                let possible_start_states = State::from_bitset(start_states[&nt]);
                for possible_start_state in possible_start_states {
                    let mut final_possible_states = 0;
                    for rules in options {
                        let mut current_states = possible_start_state.to_bitset();
                        for symbol in rules.iter() {
                            let mut new_states = 0;
                            match symbol {
                                Symbol::NonTerminal(next_nt) => {
                                    let states = State::from_bitset(current_states);
                                    *start_states.get_mut(next_nt).unwrap() |= current_states;
                                    for current_state in states {
                                        match reachable_states
                                            .get(&(current_state, *next_nt))
                                            .copied()
                                        {
                                            Some(bitset) => {
                                                new_states |= bitset;
                                            }
                                            None => {
                                                continue;
                                            }
                                        }
                                    }
                                    current_states = new_states;
                                }
                                Symbol::Terminal(token) => {
                                    let states = State::from_bitset(current_states);
                                    for current_state in states {
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
                            if current_states == 0 {
                                break;
                            }
                        }
                        final_possible_states |= current_states;
                    }
                    let key = (possible_start_state, nt);
                    // The start state cannot end after the merged token
                    if nt == self.start {
                        final_possible_states &= !State::AfterMergedToken.to_bitset();
                    }
                    reachable_states.insert(key, final_possible_states);
                }
            }
            let new_reachable_states = count_reachable_states(&reachable_states);
            let new_start_states = count_start_states(&start_states);
            if new_reachable_states == prev_reachable_states
                && new_start_states == prev_start_states
            {
                break;
            }
            prev_reachable_states = new_reachable_states;
            prev_start_states = new_start_states;
        }

        println!(
            "Time to compute reachable states: {:?}",
            start_time.elapsed()
        );

        let start_time = std::time::Instant::now();

        // First transpose the map into nt -> Vec<(State, u8)>
        let non_terminal_to_states: FxHashMap<u32, SmallVec<[(State, u8); 3]>> =
            reachable_states.iter().fold(
                FxHashMap::default(),
                |mut acc, ((start, nt), end_bitset)| {
                    if *end_bitset != 0 {
                        acc.entry(*nt).or_default().push((*start, *end_bitset));
                    }
                    acc
                },
            );

        let mut transition_map: FxHashMap<(u32, State, State), u32> = FxHashMap::default();

        // First insert empty entries for all non-terminals
        for (nt, new_states) in &non_terminal_to_states {
            let transitions: SmallVec<[(State, State); 9]> = new_states
                .iter()
                .flat_map(|(start, end_bitset)| {
                    State::from_bitset(*end_bitset)
                        .into_iter()
                        .map(move |s| (*start, s))
                })
                .collect();

            let last_index = transitions.len() - 1;
            let add_new_root = *nt == self.start && transitions.len() > 1;
            for (i, (start, end)) in transitions.iter().enumerate() {
                let key = (*nt, *start, *end);
                if i == last_index && (!add_new_root) {
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
                .flat_map(|(start, end_bitset)| {
                    State::from_bitset(*end_bitset)
                        .into_iter()
                        .map(move |s| (*start, s))
                })
                .collect();

            for (start, end) in transitions.iter().copied() {
                let options = &self.rules[*nt as usize].rhs;
                let mut new_options: Vec<Cow<[Symbol<u32>]>> = vec![];
                for rules in options {
                    let mut possible_rules = vec![];
                    for symbol in &**rules {
                        match symbol {
                            Symbol::NonTerminal(next_nt) => {
                                if possible_rules.is_empty() {
                                    let bitset = reachable_states
                                        .get(&(start, *next_nt))
                                        .copied()
                                        .unwrap_or_else(|| {
                                            // panic!("No reachable states for ({start:?}, {next_nt})")
                                            State::NONE_BITSET
                                        });
                                    let transition_map = &transition_map;
                                    possible_rules = State::from_bitset(bitset)
                                        .into_iter()
                                        .map(|next| {
                                            let next_id = transition_map[&(*next_nt, start, next)];
                                            (next, vec![Symbol::NonTerminal(next_id)])
                                        })
                                        .collect::<Vec<_>>();
                                } else {
                                    let reachable_states = &reachable_states;
                                    let transition_map = &transition_map;
                                    possible_rules = possible_rules
                                        .into_iter()
                                        .flat_map(|(start, symbols)| {
                                            let bitset = reachable_states
                                                .get(&(start, *next_nt))
                                                .copied()
                                                .unwrap_or_else(|| {
                                                    // panic!("No reachable states for ({start:?}, {next_nt})")
                                                    State::NONE_BITSET
                                                });
                                            State::from_bitset(bitset)
                                                .into_iter()
                                                .map(|next| {
                                                    let mut new_symbols = symbols.clone();
                                                    let next_id =
                                                        transition_map[&(*next_nt, start, next)];
                                                    new_symbols.push(Symbol::NonTerminal(next_id));
                                                    (next, new_symbols)
                                                })
                                                .collect::<Vec<_>>()
                                        })
                                        .collect();
                                }
                            }
                            Symbol::Terminal(token) => {
                                if possible_rules.is_empty() {
                                    start.reachable_states(merge, *token, |next_state, token| {
                                        let new_symbols = if let Some(token) = token {
                                            vec![Symbol::Terminal(token)]
                                        } else {
                                            vec![Symbol::Epsilon]
                                        };
                                        possible_rules.push((next_state, new_symbols));
                                    });
                                } else {
                                    possible_rules = possible_rules
                                        .into_iter()
                                        .flat_map(|(start, symbols)| {
                                            let mut new = Vec::new();
                                            start.reachable_states(
                                                merge,
                                                *token,
                                                |next_state, token| {
                                                    let mut new_symbols = symbols.clone();
                                                    if let Some(token) = token {
                                                        new_symbols.push(Symbol::Terminal(token));
                                                    } else {
                                                        new_symbols.push(Symbol::Epsilon);
                                                    }
                                                    new.push((next_state, new_symbols));
                                                },
                                            );
                                            new
                                        })
                                        .collect();
                                }
                            }
                            Symbol::Epsilon => {
                                if possible_rules.is_empty() {
                                    possible_rules.push((start, vec![Symbol::Epsilon]));
                                }
                            }
                        }
                    }
                    new_options.extend(
                        possible_rules
                            .into_iter()
                            .filter_map(|(state, symbols)| (state == end).then(|| symbols.into())),
                    );
                }
                let id = transition_map[&(*nt, start, end)];
                if new_options.is_empty() {
                    eprintln!("transition {nt} -> {start:?} -> {end:?} {options:?} is empty!");
                }
                self.rules[id as usize].rhs = new_options;
            }

            let add_new_root = *nt == self.start && transitions.len() > 1;
            if add_new_root {
                // If this is the start non-terminal, we need to add a new root rule
                let rhs = transitions
                    .into_iter()
                    .map(|(_, end)| {
                        vec![Symbol::NonTerminal(
                            transition_map[&(*nt, State::Start, end)],
                        )]
                        .into()
                    })
                    .collect();
                self.rules[self.start as usize].rhs = rhs;
            }
        }

        println!("Time to compute transitions: {:?}", start_time.elapsed());

        true
    }

    pub fn garbage_collect_non_terminals(&mut self) -> bool {
        // Remove any non-terminals that are not used in any rules
        let mut used_non_terminals = FxHashSet::default();
        let mut queue = vec![self.start.clone()];
        while let Some(nt) = queue.pop() {
            if used_non_terminals.insert(nt) {
                let rule = &self.rules[nt as usize];
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

        let mut changed = false;
        self.rules.retain(|_, rule| {
            let used = used_non_terminals.contains(&rule.lhs);
            changed |= !used;
            used
        });
        changed
    }

    pub fn deduplicate_non_terminals(&mut self) -> bool {
        // Remove any duplicate non-terminals
        let mut rhs_to_lhs: FxHashMap<Vec<Cow<[Symbol<u32>]>>, Vec<u32>> = FxHashMap::default();
        for (_, rule) in &self.rules {
            rhs_to_lhs
                .entry(rule.rhs.clone())
                .or_default()
                .push(rule.lhs);
        }

        // Keep only the first occurrence of each RHS
        let canonical_nt_map: FxHashMap<u32, u32> = rhs_to_lhs
            .iter()
            .flat_map(|(_, lhs_list)| {
                let first_lhs = lhs_list[0];
                lhs_list.iter().map(move |&lhs| (lhs, first_lhs))
            })
            .collect();

        // As we update also recalculate terminals_present
        self.terminals_present.clear();

        // Map all rules to their canonical non-terminal
        for (_, rule) in self.rules.iter_mut() {
            for rhs in &mut rule.rhs {
                let mut new_rhs = Vec::with_capacity(rhs.len());
                for symbol in &**rhs {
                    match symbol {
                        Symbol::NonTerminal(nt) => {
                            new_rhs.push(Symbol::NonTerminal(canonical_nt_map[nt]));
                        }
                        Symbol::Terminal(token) => {
                            self.terminals_present.insert(*token);
                            new_rhs.push(Symbol::Terminal(*token));
                        }
                        Symbol::Epsilon => {
                            new_rhs.push(Symbol::Epsilon);
                        }
                    }
                }
                *rhs = new_rhs.into();
            }
        }

        let mut changed = false;
        // Remove any non-canonical non-terminals
        self.rules.retain(|id, _| {
            let id = id as u32;
            let canonical_lhs = canonical_nt_map[&id];
            let retain = canonical_lhs == id;
            changed |= !retain;
            retain
        });
        changed
    }

    fn inline_single_use_non_terminals(&mut self) -> bool {
        let mut changed = true;
        let mut changed_overall = false;
        while changed {
            changed = false;
            let mut non_terminal_uses: FxHashMap<u32, Vec<(u32, usize)>> = FxHashMap::default();
            for (_, rule) in &self.rules {
                if rule.lhs == self.start {
                    continue;
                }
                for (i, rhs) in rule.rhs.iter().enumerate() {
                    for symbol in rhs.iter() {
                        if let Symbol::NonTerminal(nt) = symbol {
                            non_terminal_uses
                                .entry(*nt)
                                .or_default()
                                .push((rule.lhs, i));
                        }
                    }
                }
            }

            // Inline any non-terminals that are only used once
            for (nt, uses) in &non_terminal_uses {
                let &[(parent, index)] = uses.as_slice() else {
                    continue;
                };
                if self.rules[*nt as usize].rhs.len() != 1 {
                    continue;
                }
                let rule = self.rules.remove(*nt as usize);
                let rhs = &rule.rhs[0];
                let mut new_rhs = Vec::new();
                let old_rhs = &mut self.rules[parent as usize].rhs[index];
                for symbol in &**old_rhs {
                    match symbol {
                        Symbol::NonTerminal(n) if *n == *nt => {
                            new_rhs.extend(rhs.iter().cloned());
                        }
                        _ => {
                            new_rhs.push(symbol.clone());
                        }
                    }
                }
                *old_rhs = new_rhs.into();
                changed = true;
                changed_overall |= true;
                break;
            }
        }
        changed_overall
    }

    fn inline_simple(&mut self) -> bool {
        let mut changed = false;
        let mut nt_to_token: FxHashMap<u32, Symbol<u32>> = FxHashMap::default();
        self.rules.retain(|_, rule| {
            if let [rhs] = rule.rhs.as_slice() {
                match rhs.as_ref() {
                    [symbol] => {
                        nt_to_token.insert(rule.lhs, symbol.clone());
                        return false;
                    }
                    _ => {}
                }
            }
            true
        });

        // Update all rules to replace the non-terminal with the terminal
        for (_, rule) in self.rules.iter_mut() {
            for rhs in &mut rule.rhs {
                let mut new_rhs = Vec::new();
                for symbol in rhs.iter() {
                    if let Symbol::NonTerminal(n) = symbol {
                        if let Some(symbol) = nt_to_token.get(n) {
                            let symbol = if let Symbol::NonTerminal(nt) = symbol {
                                nt_to_token.get(nt).cloned().unwrap_or(symbol.clone())
                            } else {
                                symbol.clone()
                            };
                            if symbol != Symbol::Epsilon {
                                new_rhs.push(symbol.clone());
                            }
                            changed = true;
                            continue;
                        }
                    }
                    new_rhs.push(symbol.clone());
                }
                *rhs = new_rhs.into();
            }
        }

        changed
    }

    pub fn inline_optimize(&mut self) {
        loop {
            let mut changed = self.inline_single_use_non_terminals();
            changed |= self.inline_simple();
            changed |= self.garbage_collect_non_terminals();
            changed |= self.deduplicate_non_terminals();
            if !changed {
                break;
            }
        }
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
    let bump = bumpalo::Bump::new();
    let grammar = grammar.replace_tokenizer_terminals(&tokenizer);
    println!("start rule count: {}", grammar.rules.len());
    let mut grammar = SlabGrammar::new(&grammar);
    grammar.shortcut_merge(&Merge {
        rank: 0,
        pair: [
            tokenizer.bytes[b't' as usize],
            tokenizer.bytes[b'o' as usize],
        ],
        new_token: 10_000,
    });
    grammar.garbage_collect_non_terminals();
    let grammar = grammar.to_grammar();
    println!("CNF grammar:\n{}", grammar);
    let dense_grammar = grammar.reallocate(&bump);
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
    assert!(dense_grammar.recognizes_tokens(tokens));
}

#[test]
fn test_slab_grammar() {
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
    let cnf_grammar = SlabGrammar::new(&cnf_grammar);
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
    assert!(dense_grammar.recognizes(b"(str.to.int name)", &tokenizer));
}
