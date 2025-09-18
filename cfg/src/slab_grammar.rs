use std::collections::{HashSet, VecDeque};

use beef::Cow;
use rustc_hash::{FxHashMap, FxHashSet};
use slab::Slab;
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

    fn reachable_states(
        &self,
        merge: &Merge,
        token: u32,
        mut f: impl FnMut(State, Option<u32>),
        allow_incorrect: bool,
    ) {
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
                if token != merge.pair[1] || allow_incorrect {
                    if token == merge.pair[0] {
                        f(State::AfterFirstToken, Some(token));
                        f(State::AfterMergedToken, Some(merge.new_token));
                    } else {
                        f(State::Start, Some(token));
                    }
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

/// One production rule: *lhs → rhs1 | rhs2 | …*
#[derive(Debug, Clone)]
pub struct SlabRule {
    /// The left‑hand non‑terminal.
    pub lhs: u32,
    /// Alternative right‑hand sides, each a vector of symbols composing a *sequence*.
    pub rhs: Slab<Cow<'static, [Symbol<u32>]>>,
    /// Where this rule is used
    pub used_in: FxHashSet<UsedLocation>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct UsedLocation {
    rule: u32,
    index: u32,
}

#[derive(Debug, Clone)]
pub struct SlabGrammar {
    pub start: u32,
    pub rules: slab::Slab<SlabRule>,
    pub terminal_locations: FxHashMap<u32, FxHashSet<UsedLocation>>,
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
            let mut rhs = Slab::new();
            for option in rule.rhs {
                rhs.insert(option);
            }
            rules.insert(SlabRule {
                lhs: rule.lhs,
                rhs,
                used_in: FxHashSet::default(),
            });
        }

        let mut terminals_locations: FxHashMap<u32, FxHashSet<UsedLocation>> = FxHashMap::default();

        // Collect all terminals present in the grammar
        let rule_ids = rules.iter().map(|(id, _)| id as u32).collect::<Vec<_>>();
        for rule_id in rule_ids {
            let rule = &rules[rule_id as usize];
            let rule_variant_ids = rule.rhs.iter().map(|(id, _)| id).collect::<Vec<_>>();
            for i in rule_variant_ids {
                let rule = &mut rules[rule_id as usize];
                let rhs = rule.rhs[i].clone();

                for symbol in (*rhs).iter() {
                    let location = UsedLocation {
                        rule: rule_id as u32,
                        index: i as u32,
                    };
                    println!("adding {location:?}");
                    match symbol {
                        Symbol::Terminal(token) => {
                            terminals_locations
                                .entry(*token)
                                .or_default()
                                .insert(location);
                        }
                        Symbol::NonTerminal(nt) => {
                            let rule = &mut rules[*nt as usize];
                            rule.used_in.insert(location);
                        }
                        _ => {}
                    }
                }
            }
        }

        println!("rules: {rules:?}");

        Self {
            start,
            rules,
            terminal_locations: terminals_locations,
        }
    }

    fn remove(&mut self, id: u32) {
        let rule_ids = self.rules[id as usize]
            .rhs
            .iter()
            .map(|(id, _)| id as u32)
            .collect::<Vec<_>>();
        for rule_id in rule_ids {
            let len = self.rules[id as usize].rhs[rule_id as usize].len();
            for i in 0..len {
                let rhs = &mut self.rules[id as usize].rhs[rule_id as usize];
                let symbol = rhs[i].clone();
                if let Symbol::Terminal(token) = symbol {
                    if let Some(locations) = self.terminal_locations.get_mut(&token) {
                        locations.retain(|loc| loc.rule != id as u32);
                    }
                } else if let Symbol::NonTerminal(nt) = symbol {
                    if let Some(nt_rule) = self.rules.get_mut(nt as usize) {
                        nt_rule.used_in.retain(|loc| loc.rule != id as u32);
                    }
                }
            }
        }
        self.rules.remove(id as usize);
    }

    fn insert(&mut self, rhs: Slab<Cow<'static, [Symbol<u32>]>>) -> u32 {
        let entry = self.rules.vacant_entry();
        let lhs = entry.key() as u32;

        entry.insert(SlabRule {
            lhs,
            rhs,
            used_in: FxHashSet::default(),
        });

        lhs
    }

    // a -> b
    // b -> b1 | c | 2
    // c -> b
    pub fn shortcut_merge(&mut self, merge: &Merge, allow_incorrect: bool) -> bool {
        // If neither of the pair tokens are present in the grammar, we can skip the merge
        let (Some(pair_0), Some(pair_1)) = (
            self.terminal_locations.get(&merge.pair[0]).cloned(),
            self.terminal_locations.get(&merge.pair[1]).cloned(),
        ) else {
            return false;
        };

        // First identify locations where the two pairs are adjacent in the same rule
        let mut possible_set = pair_1.clone();
        let mut starts_with_pair_1 = [Symbol::Terminal(merge.pair[1])]
            .into_iter()
            .collect::<FxHashSet<_>>();
        let mut last_starts_with_pair_1 = starts_with_pair_1.len() + 1;
        while starts_with_pair_1.len() != last_starts_with_pair_1 {
            last_starts_with_pair_1 = starts_with_pair_1.len();
            let mut new_possible = Vec::new();
            for loc in &possible_set {
                println!("{:?}", self.rules);
                let rule = &self.rules[dbg!(loc.rule) as usize];
                let rhs = &rule.rhs[loc.index as usize];
                let this_nt = Symbol::NonTerminal(loc.rule);
                if let Some(first) = rhs.first() {
                    if starts_with_pair_1.contains(first) {
                        starts_with_pair_1.insert(this_nt);
                        new_possible.extend(rule.used_in.iter().copied());
                    }
                }
            }
            possible_set.extend(new_possible);
        }

        let mut possible_set = pair_1.clone();
        let mut ends_with_pair_0 = [Symbol::Terminal(merge.pair[0])]
            .into_iter()
            .collect::<FxHashSet<_>>();
        let mut last_ends_with_pair_0 = ends_with_pair_0.len() + 1;
        while ends_with_pair_0.len() != last_ends_with_pair_0 {
            last_ends_with_pair_0 = ends_with_pair_0.len();
            let mut new_possible = Vec::new();
            for loc in &possible_set {
                let rule = &self.rules[loc.rule as usize];
                let rhs = &rule.rhs[loc.index as usize];
                let this_nt = Symbol::NonTerminal(loc.rule);
                if let Some(first) = rhs.last() {
                    if ends_with_pair_0.contains(first) {
                        ends_with_pair_0.insert(this_nt);
                        new_possible.extend(rule.used_in.iter().copied());
                    }
                }
            }
            possible_set.extend(new_possible);
        }

        // Now we need to find any locations where every starts_with_pair_1 is used
        let ends_with_pair_0_used = ends_with_pair_0
            .iter()
            .filter_map(|loc| {
                if let Symbol::NonTerminal(nt) = loc {
                    Some(*nt)
                } else {
                    None
                }
            })
            .flat_map(|loc| self.rules[loc as usize].used_in.iter())
            .chain(pair_0.iter())
            .cloned()
            .collect::<FxHashSet<_>>();
        let starts_with_pair_1_used = starts_with_pair_1
            .iter()
            .filter_map(|loc| {
                if let Symbol::NonTerminal(nt) = loc {
                    Some(*nt)
                } else {
                    None
                }
            })
            .flat_map(|loc| self.rules[loc as usize].used_in.iter())
            .chain(pair_1.iter())
            .cloned()
            .collect::<FxHashSet<_>>();

        // Find all adjacent pairs of ends_with_pair_0_used and starts_with_pair_1_used
        let both_appear = ends_with_pair_0_used
            .union(&starts_with_pair_1_used)
            .collect::<FxHashSet<_>>();
        let mut appear_adjacent = FxHashSet::default();
        let mut appear_at_start = FxHashSet::default();
        let mut appear_at_end = FxHashSet::default();
        for loc in both_appear {
            let rule = &self.rules[loc.rule as usize];
            let rhs = &rule.rhs[loc.index as usize];
            for i in 0..rhs.len() - 1 {
                let first = &rhs[i];
                let second = &rhs[i + 1];
                if ends_with_pair_0.contains(&*first) && starts_with_pair_1.contains(&*second) {
                    appear_adjacent.insert(loc.clone());
                    if let Symbol::NonTerminal(token) = first {
                        appear_at_end.insert(*token);
                    }
                    if let Symbol::NonTerminal(token) = second {
                        appear_at_start.insert(*token);
                    }
                }
            }
        }

        println!("appear_adjacent: {appear_adjacent:?}");

        // Create a variant of appear_at_end with the last symbol swapped for the new token
        let mut merged_appear_at_end = FxHashMap::default();
        // First insert an empty value for each non-terminal
        for loc in &appear_at_end {
            let empty = self.insert(Default::default());
            merged_appear_at_end.insert(*loc, empty);
        }
        // Then fill it in
        for loc in &appear_at_end {
            let mut unchanged = Slab::new();
            self.rules[*loc as usize].rhs.retain(|_, rhs| {
                if rhs.last().is_some_and(|rhs| ends_with_pair_0.contains(rhs)) {
                    true
                } else {
                    unchanged.insert(rhs.clone());
                    false
                }
            });
            let common = self.insert(unchanged);
            let old_rule = &mut self.rules[*loc as usize];
            _ = old_rule
                .rhs
                .insert(vec![Symbol::NonTerminal(common)].into());
            let mut merged = old_rule.clone();
            for (_, rhs) in &mut merged.rhs {
                if rhs.last().is_some_and(|rhs| ends_with_pair_0.contains(rhs)) {
                    let mut new_rhs = rhs.to_vec();
                    let tok = new_rhs.pop().unwrap();
                    new_rhs.push(match tok {
                        Symbol::Terminal(_) => Symbol::Terminal(merge.new_token),
                        Symbol::NonTerminal(nt) => Symbol::NonTerminal(merged_appear_at_end[&nt]),
                        other => other,
                    });
                    *rhs = new_rhs.into();
                }
            }
        }

        // Create a variant of appear_at_start with the first symbol removed
        let mut merged_appear_at_start = FxHashMap::default();
        // First insert an empty value for each non-terminal
        for loc in &appear_at_start {
            let empty = self.insert(Default::default());
            merged_appear_at_start.insert(*loc, empty);
        }
        // Then fill it in
        for loc in &appear_at_start {
            let mut unchanged = Slab::new();
            self.rules[*loc as usize].rhs.retain(|_, rhs| {
                if rhs
                    .first()
                    .is_some_and(|rhs| starts_with_pair_1.contains(rhs))
                {
                    true
                } else {
                    unchanged.insert(rhs.clone());
                    false
                }
            });
            let common = self.insert(unchanged);
            let old_rule = &mut self.rules[*loc as usize];
            _ = old_rule
                .rhs
                .insert(vec![Symbol::NonTerminal(common)].into());
            let mut merged = old_rule.clone();
            for (_, rhs) in &mut merged.rhs {
                if rhs
                    .first()
                    .is_some_and(|rhs| starts_with_pair_1.contains(rhs))
                {
                    let mut new_rhs = rhs.to_vec();
                    let tok = new_rhs[0].clone();
                    new_rhs[0] = match tok {
                        Symbol::Terminal(_) => Symbol::Terminal(merge.new_token),
                        Symbol::NonTerminal(nt) => Symbol::NonTerminal(merged_appear_at_start[&nt]),
                        other => other,
                    };
                    *rhs = new_rhs.into();
                }
            }
        }

        // Then substitute the new token in all locations where the pair appears
        for loc in appear_adjacent {
            let rule = &mut self.rules[loc.rule as usize];
            let rhs = &mut rule.rhs[loc.index as usize];
            let mut new_rhs = Vec::new();
            let mut i = 0;
            while i < rhs.len() - 1 {
                let first = &rhs[i];
                let second = &rhs[i + 1];
                if ends_with_pair_0.contains(first) && starts_with_pair_1.contains(second) {
                    let new_first = if let Symbol::NonTerminal(nt) = first {
                        Symbol::NonTerminal(merged_appear_at_end[&nt])
                    } else {
                        Symbol::Terminal(merge.new_token)
                    };
                    new_rhs.push(new_first);
                    if let Symbol::NonTerminal(nt) = second {
                        new_rhs.push(Symbol::NonTerminal(merged_appear_at_start[&nt]));
                    }
                    i += 2;
                } else {
                    new_rhs.push(first.clone());
                    i += 1;
                }
            }
            *rhs = new_rhs.into();
        }

        true
    }

    pub fn garbage_collect_non_terminals(&mut self) -> bool {
        // Remove any non-terminals that are not used in any rules
        let mut used_non_terminals = FxHashSet::default();
        let mut queue = vec![self.start.clone()];
        while let Some(nt) = queue.pop() {
            if used_non_terminals.insert(nt) {
                let rule = &self.rules[nt as usize];
                for (_, rhs) in &rule.rhs {
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
        self.retain(|_, rule| {
            let used = used_non_terminals.contains(&rule.lhs);
            changed |= !used;
            used
        });
        changed
    }

    pub fn retain(&mut self, mut f: impl FnMut(u32, &SlabRule) -> bool) {
        let mut rule_ids = self
            .rules
            .iter()
            .map(|(id, _)| id as u32)
            .collect::<Vec<_>>();
        for id in rule_ids {
            let rule = &self.rules[id as usize];
            let keep = f(id, rule);
            if !keep {
                self.remove(id);
            }
        }
    }

    pub fn deduplicate_non_terminals(&mut self) -> bool {
        // // Remove any duplicate non-terminals
        // let mut rhs_to_lhs: FxHashMap<Slab<Cow<[Symbol<u32>]>>, Vec<u32>> = FxHashMap::default();
        // for (_, rule) in &self.rules {
        //     rhs_to_lhs
        //         .entry(rule.rhs.clone())
        //         .or_default()
        //         .push(rule.lhs);
        // }

        // // Keep only the first occurrence of each RHS
        // let canonical_nt_map: FxHashMap<u32, u32> = rhs_to_lhs
        //     .iter()
        //     .flat_map(|(_, lhs_list)| {
        //         let first_lhs = lhs_list[0];
        //         lhs_list.iter().map(move |&lhs| (lhs, first_lhs))
        //     })
        //     .collect();

        // // As we update also recalculate terminal_locations
        // self.terminal_locations.clear();

        // // Map all rules to their canonical non-terminal
        // for (rule_id, rule) in self.rules.iter_mut() {
        //     for (_, rhs) in &mut rule.rhs {
        //         let mut new_rhs = Vec::with_capacity(rhs.len());
        //         for (i, symbol) in (*rhs).iter().enumerate() {
        //             match symbol {
        //                 Symbol::NonTerminal(nt) => {
        //                     new_rhs.push(Symbol::NonTerminal(canonical_nt_map[nt]));
        //                 }
        //                 Symbol::Terminal(token) => {
        //                     self.terminal_locations
        //                         .entry(*token)
        //                         .or_default()
        //                         .insert((rule_id as u32, i));
        //                     new_rhs.push(Symbol::Terminal(*token));
        //                 }
        //                 Symbol::Epsilon => {
        //                     new_rhs.push(Symbol::Epsilon);
        //                 }
        //             }
        //         }
        //         *rhs = new_rhs.into();
        //     }
        // }

        // let mut changed = false;
        // // Remove any non-canonical non-terminals
        // self.rules.retain(|id, _| {
        //     let id = id as u32;
        //     let canonical_lhs = canonical_nt_map[&id];
        //     let retain = canonical_lhs == id;
        //     changed |= !retain;
        //     retain
        // });
        // changed
        todo!()
    }

    fn inline_single_use_non_terminals(&mut self) -> bool {
        // let mut changed = true;
        // let mut changed_overall = false;
        // while changed {
        //     changed = false;
        //     let mut non_terminal_uses: FxHashMap<u32, Vec<(u32, usize)>> = FxHashMap::default();
        //     for (_, rule) in &self.rules {
        //         if rule.lhs == self.start {
        //             continue;
        //         }
        //         for (i, rhs) in rule.rhs.iter().enumerate() {
        //             for symbol in rhs.iter() {
        //                 if let Symbol::NonTerminal(nt) = symbol {
        //                     non_terminal_uses
        //                         .entry(*nt)
        //                         .or_default()
        //                         .push((rule.lhs, i));
        //                 }
        //             }
        //         }
        //     }

        //     // Inline any non-terminals that are only used once
        //     for (nt, uses) in &non_terminal_uses {
        //         let &[(parent, index)] = uses.as_slice() else {
        //             continue;
        //         };
        //         let rhs = &self.rules[*nt as usize].rhs;
        //         let [rhs] = rhs.as_slice() else {
        //             continue;
        //         };
        //         if rhs.contains(&Symbol::NonTerminal(*nt)) {
        //             // If the non-terminal contains itself, we cannot inline it
        //             continue;
        //         }
        //         let rule = self.rules.remove(*nt as usize);
        //         let rhs = &rule.rhs[0];
        //         let mut new_rhs = Vec::new();
        //         let old_rhs = &mut self.rules[parent as usize].rhs[index];
        //         for symbol in &**old_rhs {
        //             match symbol {
        //                 Symbol::NonTerminal(n) if *n == *nt => {
        //                     new_rhs.extend(rhs.iter().cloned());
        //                 }
        //                 _ => {
        //                     new_rhs.push(symbol.clone());
        //                 }
        //             }
        //         }
        //         *old_rhs = new_rhs.into();
        //         changed = true;
        //         changed_overall |= true;
        //         break;
        //     }
        // }
        // changed_overall
        todo!()
    }

    fn inline_simple(&mut self) -> bool {
        // let mut changed = false;
        // let mut nt_to_token: FxHashMap<u32, Symbol<u32>> = FxHashMap::default();
        // self.rules.retain(|_, rule| {
        //     if let [rhs] = rule.rhs.as_slice() {
        //         match rhs.as_ref() {
        //             [symbol] => {
        //                 nt_to_token.insert(rule.lhs, symbol.clone());
        //                 return false;
        //             }
        //             _ => {}
        //         }
        //     }
        //     true
        // });

        // // Update all rules to replace the non-terminal with the terminal
        // for (_, rule) in self.rules.iter_mut() {
        //     for (_, rhs) in &mut rule.rhs {
        //         let mut new_rhs = Vec::new();
        //         for symbol in rhs.iter() {
        //             if let Symbol::NonTerminal(n) = symbol {
        //                 if let Some(symbol) = nt_to_token.get(n) {
        //                     let mut symbol = symbol.clone();
        //                     while let Symbol::NonTerminal(nt) = symbol {
        //                         if let Some(next) = nt_to_token.get(&nt) {
        //                             symbol = next.clone();
        //                         } else {
        //                             break;
        //                         }
        //                     }
        //                     if symbol != Symbol::Epsilon {
        //                         new_rhs.push(symbol.clone());
        //                     }
        //                     changed = true;
        //                     continue;
        //                 }
        //             }
        //             new_rhs.push(symbol.clone());
        //         }
        //         *rhs = new_rhs.into();
        //     }
        // }

        // changed
        todo!()
    }

    pub fn inline_optimize(&mut self) {
        // loop {
        //     let mut changed = self.inline_single_use_non_terminals();
        //     changed |= self.inline_simple();
        //     changed |= self.garbage_collect_non_terminals();
        //     changed |= self.deduplicate_non_terminals();
        //     if !changed {
        //         break;
        //     }
        // }
    }

    pub fn to_grammar(&self) -> Grammar<u32> {
        let mut rules = Vec::new();
        for (_, rule) in &self.rules {
            rules.push(Rule {
                lhs: rule.lhs,
                rhs: rule.rhs.iter().map(|(_, r)| r.clone()).collect(),
            });
        }
        Grammar {
            start: self.start,
            rules,
        }
    }

    fn verify_integrity(&self, msg: &str) {
        // Verify that all non-terminals exist in the rules
        for (lhs, rules) in &self.rules {
            assert_eq!(
                lhs, rules.lhs as usize,
                "LHS of rule does not match its key {msg}",
            );
            for (_, rhs) in &rules.rhs {
                for symbol in &**rhs {
                    if let Symbol::NonTerminal(nt) = symbol {
                        assert!(
                            self.rules.contains(*nt as usize),
                            "Non-terminal {} does not exist in the rules {msg}",
                            nt
                        );
                    }
                }
            }
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
    let start = std::time::Instant::now();
    let grammar = grammar.replace_tokenizer_terminals(&tokenizer);
    println!("Time to replace terminals: {:?}", start.elapsed());
    println!("start rule count: {}", grammar.rules.len());
    let mut grammar = SlabGrammar::new(&grammar);
    grammar.shortcut_merge(
        &Merge {
            rank: 0,
            pair: dbg!([
                tokenizer.bytes[b't' as usize],
                tokenizer.bytes[b'o' as usize],
            ]),
            new_token: 10_000,
        },
        false,
    );
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
