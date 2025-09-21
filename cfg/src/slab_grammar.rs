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
    pub used_in: FxHashMap<UsedLocation, usize>,
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
    pub terminal_locations: FxHashMap<u32, FxHashMap<UsedLocation, usize>>,
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
                used_in: Default::default(),
            });
        }

        let mut terminals_locations: FxHashMap<u32, FxHashMap<UsedLocation, usize>> =
            FxHashMap::default();

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
                    match symbol {
                        Symbol::Terminal(token) => {
                            *terminals_locations
                                .entry(*token)
                                .or_default()
                                .entry(location)
                                .or_default() += 1;
                        }
                        Symbol::NonTerminal(nt) => {
                            let rule = &mut rules[*nt as usize];
                            *rule.used_in.entry(location).or_default() += 1;
                        }
                        _ => {}
                    }
                }
            }
        }

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
                let loc = UsedLocation {
                    rule: id,
                    index: rule_id,
                };
                self.remove_reference(&symbol, loc);
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
            used_in: Default::default(),
        });

        lhs
    }

    fn push_rhs(&mut self, lhs: u32, rhs: Cow<'static, [Symbol<u32>]>) {
        let rhs_clone = rhs.clone();
        let rule = &mut self.rules[lhs as usize];
        let index = rule.rhs.insert(rhs);
        let location = UsedLocation {
            rule: lhs,
            index: index as u32,
        };
        for symbol in &*rhs_clone {
            self.add_reference(symbol, location);
        }
    }

    fn add_reference(&mut self, symbol: &Symbol<u32>, location: UsedLocation) {
        match symbol {
            Symbol::Terminal(token) => {
                *self
                    .terminal_locations
                    .entry(*token)
                    .or_default()
                    .entry(location)
                    .or_default() += 1;
            }
            Symbol::NonTerminal(nt) => {
                let rule = &mut self.rules[*nt as usize];
                *rule.used_in.entry(location).or_default() += 1;
            }
            _ => {}
        }
    }

    fn remove_reference(&mut self, symbol: &Symbol<u32>, location: UsedLocation) {
        match symbol {
            Symbol::Terminal(token) => {
                if let Some(locations) = self.terminal_locations.get_mut(&token) {
                    let entry = locations.entry(location).or_default();
                    if *entry > 1 {
                        *entry -= 1;
                    } else {
                        locations.remove(&location);
                    }
                    if locations.is_empty() {
                        self.terminal_locations.remove(&token);
                    }
                }
            }
            Symbol::NonTerminal(nt) => {
                if let Some(rule) = self.rules.get_mut(*nt as usize) {
                    let entry = rule.used_in.entry(location).or_default();
                    if *entry > 1 {
                        *entry -= 1;
                    } else {
                        rule.used_in.remove(&location);
                    }
                }
            }
            _ => {}
        }
    }

    fn split_nt(
        &mut self,
        loc: &u32,
        new_loc: &u32,
        common: &u32,
        mut new_row: impl FnMut(&[Symbol<u32>]) -> bool,
        mut replace_row: impl FnMut(&[Symbol<u32>]) -> Vec<Symbol<u32>>,
    ) {
        let mut id_mapping = FxHashMap::default();
        let mut changed = Slab::new();
        let mut unchanged = Slab::new();
        self.rules[*loc as usize].rhs.retain(|i, rhs| {
            if new_row(&rhs) {
                changed.insert(rhs.clone());
                true
            } else {
                let new_id = unchanged.insert(rhs.clone());
                id_mapping.insert(new_id, i);
                false
            }
        });

        let mut merged = changed;
        for (i, rhs) in &mut merged {
            if new_row(&rhs) {
                *rhs = replace_row(&rhs).into();
            }
            let location = UsedLocation {
                rule: *new_loc,
                index: i as u32,
            };
            for symbol in &**rhs {
                self.add_reference(symbol, location);
            }
        }
        self.rules[*new_loc as usize].rhs = merged;

        if !unchanged.is_empty() {
            for (id, rhs) in &unchanged {
                let old_location = UsedLocation {
                    rule: *loc,
                    index: id_mapping[&id] as u32,
                };
                let common_location = UsedLocation {
                    rule: *common,
                    index: id as u32,
                };
                for symbol in &**rhs {
                    self.remove_reference(symbol, old_location);
                    self.add_reference(symbol, common_location);
                }
            }
            self.rules[*common as usize].rhs = unchanged;
            self.push_rhs(*loc, vec![Symbol::NonTerminal(*common)].into());
        }

        self.verify_integrity("after split_nt");
    }

    fn find_starts_with_token(
        &self,
        token: u32,
        starting_set: FxHashSet<UsedLocation>,
        starting_set_symbols: FxHashSet<Symbol<u32>>,
    ) -> FxHashSet<Symbol<u32>> {
        let mut possible_set = starting_set;
        let mut starts_with_token = starting_set_symbols;
        starts_with_token.insert(Symbol::Terminal(token));
        let mut last_starts_with_token = 0;
        while starts_with_token.len() != last_starts_with_token {
            last_starts_with_token = starts_with_token.len();
            let mut new_possible = Vec::new();
            for loc in possible_set.drain() {
                let rule = &self.rules[loc.rule as usize];
                let rhs = &rule.rhs[loc.index as usize];
                if let Some(first) = rhs.first() {
                    if starts_with_token.contains(first) {
                        starts_with_token.insert(Symbol::NonTerminal(loc.rule));
                        new_possible.extend(rule.used_in.keys().copied());
                    }
                }
            }
            possible_set.extend(new_possible);
        }
        starts_with_token
    }

    fn find_ends_with_token(
        &self,
        token: u32,
        starting_set: FxHashSet<UsedLocation>,
        starting_set_symbols: FxHashSet<Symbol<u32>>,
    ) -> FxHashSet<Symbol<u32>> {
        let mut possible_set = starting_set;
        let mut ends_with_token = starting_set_symbols;
        ends_with_token.insert(Symbol::Terminal(token));
        let mut last_ends_with_token = 0;
        while ends_with_token.len() != last_ends_with_token {
            last_ends_with_token = ends_with_token.len();
            let mut new_possible = Vec::new();
            for loc in possible_set.drain() {
                let rule = &self.rules[loc.rule as usize];
                let rhs = &rule.rhs[loc.index as usize];
                if let Some(first) = rhs.last() {
                    if ends_with_token.contains(first) {
                        ends_with_token.insert(Symbol::NonTerminal(loc.rule));
                        new_possible.extend(rule.used_in.keys().copied());
                    }
                }
            }
            possible_set.extend(new_possible);
        }
        ends_with_token
    }

    fn find_merge_canidates(
        &self,
        merge: &Merge,
        starts_with_pair_1: &FxHashSet<Symbol<u32>>,
        ends_with_pair_0: &FxHashSet<Symbol<u32>>,
    ) -> FxHashSet<UsedLocation> {
        let pair_0 = &self.terminal_locations[&merge.pair[0]];
        let pair_1 = &self.terminal_locations[&merge.pair[1]];

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
            .flat_map(|loc| self.rules[loc as usize].used_in.keys())
            .chain(pair_0.keys())
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
            .flat_map(|loc| self.rules[loc as usize].used_in.keys())
            .chain(pair_1.keys())
            .cloned()
            .collect::<FxHashSet<_>>();

        // Find all adjacent pairs of ends_with_pair_0_used and starts_with_pair_1_used
        let both_appear = ends_with_pair_0_used
            .union(&starts_with_pair_1_used)
            .cloned()
            .collect::<FxHashSet<_>>();
        let mut appear_adjacent = FxHashSet::default();

        for loc in &both_appear {
            let rule = &self.rules[loc.rule as usize];
            let rhs = &rule.rhs[loc.index as usize];
            for window in rhs.windows(2) {
                let first = &window[0];
                let second = &window[1];
                if ends_with_pair_0.contains(first) && starts_with_pair_1.contains(second) {
                    appear_adjacent.insert(loc.clone());
                }
            }
        }

        appear_adjacent
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
        let starts_with_pair_1 = self.find_starts_with_token(
            merge.pair[1],
            pair_1.keys().cloned().collect(),
            Default::default(),
        );
        let ends_with_pair_0 = self.find_ends_with_token(
            merge.pair[0],
            pair_0.keys().cloned().collect(),
            Default::default(),
        );

        // Create a variant of appear_at_end with the last symbol swapped for the new token
        let mut merged_appear_at_end = FxHashMap::default();
        // First insert an empty value for each non-terminal
        for symbol in &ends_with_pair_0 {
            let Symbol::NonTerminal(nt) = symbol else {
                continue;
            };
            merged_appear_at_end.insert(
                *nt,
                [
                    self.insert(Default::default()),
                    self.insert(Default::default()),
                ],
            );
        }
        // Then fill it in
        for (loc, [new_loc, common]) in &merged_appear_at_end {
            self.split_nt(
                loc,
                new_loc,
                common,
                |rhs| rhs.last().is_some_and(|rhs| ends_with_pair_0.contains(rhs)),
                |rhs| {
                    let mut new_rhs = rhs.to_vec();
                    let tok = new_rhs.pop().unwrap();
                    new_rhs.push(match tok {
                        Symbol::Terminal(_) => Symbol::Terminal(merge.new_token),
                        Symbol::NonTerminal(nt) => {
                            Symbol::NonTerminal(merged_appear_at_end[&nt][0])
                        }
                        other => other,
                    });
                    new_rhs
                },
            );
        }

        // Create a variant of appear_at_start with the first symbol removed
        let mut merged_appear_at_start = FxHashMap::default();
        // First insert an empty value for each non-terminal
        for symbol in &starts_with_pair_1 {
            let Symbol::NonTerminal(nt) = symbol else {
                continue;
            };
            merged_appear_at_start.insert(
                *nt,
                [
                    self.insert(Default::default()),
                    self.insert(Default::default()),
                ],
            );
        }
        // Then fill it in
        for (loc, [new_loc, common]) in &merged_appear_at_start {
            self.split_nt(
                loc,
                new_loc,
                common,
                |rhs| {
                    rhs.first()
                        .is_some_and(|rhs| starts_with_pair_1.contains(rhs))
                },
                |rhs| {
                    let mut new_rhs = rhs.to_vec();
                    let tok = new_rhs[0].clone();
                    match tok {
                        Symbol::NonTerminal(nt) => {
                            new_rhs[0] = Symbol::NonTerminal(merged_appear_at_start[&nt][0])
                        }
                        _ => {
                            if new_rhs.len() > 1 {
                                new_rhs.remove(0);
                            } else {
                                new_rhs[0] = Symbol::Epsilon
                            }
                        }
                    };
                    new_rhs
                },
            );
        }

        // If neither of the pair tokens are present in the grammar, we can skip the merge
        let (Some(pair_0), Some(pair_1)) = (
            self.terminal_locations.get(&merge.pair[0]).cloned(),
            self.terminal_locations.get(&merge.pair[1]).cloned(),
        ) else {
            return false;
        };

        // First identify locations where the two pairs are adjacent in the same rule
        let starts_with_pair_1 = self.find_starts_with_token(
            merge.pair[1],
            pair_1.keys().cloned().collect(),
            Default::default(),
        );
        let ends_with_pair_0 = self.find_ends_with_token(
            merge.pair[0],
            pair_0.keys().cloned().collect(),
            Default::default(),
        );

        let appear_adjacent =
            self.find_merge_canidates(merge, &starts_with_pair_1, &ends_with_pair_0);

        // Then substitute the new token in all locations where the pair appears
        for loc in appear_adjacent {
            let mut new_rhs = Vec::new();
            let mut i = 0;
            while i < self.rules[loc.rule as usize].rhs[loc.index as usize].len() {
                let first = self.rules[loc.rule as usize].rhs[loc.index as usize][i].clone();
                if i + 1 < self.rules[loc.rule as usize].rhs[loc.index as usize].len() {
                    let second =
                        self.rules[loc.rule as usize].rhs[loc.index as usize][i + 1].clone();
                    if ends_with_pair_0.contains(&first) && starts_with_pair_1.contains(&second) {
                        self.remove_reference(&first, loc);
                        self.remove_reference(&second, loc);
                        let maybe_merge = self.insert(Default::default());
                        let maybe_merge_symbol = Symbol::NonTerminal(maybe_merge);
                        self.add_reference(&maybe_merge_symbol, loc);
                        new_rhs.push(maybe_merge_symbol);

                        let mut do_merge_rhs = Vec::new();
                        let mut dont_merge_rhs = FxHashSet::default();
                        if let Symbol::NonTerminal(nt) = first {
                            let [first_merged, first_common] = merged_appear_at_end[&nt];
                            do_merge_rhs.push(Symbol::NonTerminal(first_merged));
                            if !self.rules[first_common as usize].rhs.is_empty() && !allow_incorrect
                            {
                                dont_merge_rhs.insert(
                                    vec![Symbol::NonTerminal(first_common), second.clone()].into(),
                                );
                            }
                        } else {
                            do_merge_rhs.push(Symbol::Terminal(merge.new_token));
                        }
                        if let Symbol::NonTerminal(nt) = second {
                            let [second_merged, second_common] = merged_appear_at_start[&nt];
                            do_merge_rhs.push(Symbol::NonTerminal(second_merged));
                            if !self.rules[second_common as usize].rhs.is_empty()
                                && !allow_incorrect
                            {
                                dont_merge_rhs.insert(
                                    vec![first.clone(), Symbol::NonTerminal(second_common)].into(),
                                );
                            }
                        }
                        if allow_incorrect {
                            dont_merge_rhs.insert(vec![first.clone(), second.clone()].into());
                        }
                        self.push_rhs(maybe_merge, do_merge_rhs.into());
                        for dont_merge_rhs in dont_merge_rhs {
                            self.push_rhs(maybe_merge, dont_merge_rhs);
                        }
                        i += 2;
                    } else {
                        new_rhs.push(first.clone());
                        i += 1;
                    }
                } else {
                    new_rhs.push(first.clone());
                    i += 1;
                }
            }
            self.rules[loc.rule as usize].rhs[loc.index as usize] = new_rhs.into();
            self.verify_integrity("after rewriting rule in shortcut_merge");
        }

        true
    }

    pub fn garbage_collect_non_terminals(&mut self) -> bool {
        // Remove any non-terminals that are not used in any rules
        let mut used_non_terminals = FxHashSet::default();
        let mut queue = vec![self.start.clone()];
        used_non_terminals.insert(self.start);
        while let Some(nt) = queue.pop() {
            let rule = &self.rules[nt as usize];
            for (_, rhs) in &rule.rhs {
                for symbol in &**rhs {
                    if let parse::Symbol::NonTerminal(non_terminal) = symbol {
                        if used_non_terminals.insert(*non_terminal) {
                            queue.push(non_terminal.clone());
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
        let rule_ids = self
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
        let mut changed = false;

        // Remove any duplicate non-terminals
        let mut rhs_to_lhs: FxHashMap<Vec<Cow<[Symbol<u32>]>>, Vec<u32>> = FxHashMap::default();
        for (_, rule) in &self.rules {
            let mut rhs: Vec<_> = rule.rhs.iter().map(|(_, r)| r.clone()).collect();
            rhs.sort();
            rhs_to_lhs.entry(rhs).or_default().push(rule.lhs);
        }

        for value in rhs_to_lhs.values_mut() {
            value.sort_unstable_by_key(|a| if *a == self.start { 0 } else { 1 });
        }

        // Keep only the first occurrence of each RHS
        let mut non_canonical_nts = Vec::new();
        let mut canonical_nt_map = FxHashMap::default();
        for (_, lhs_list) in &rhs_to_lhs {
            let [first, rest @ ..] = &lhs_list[..] else {
                continue;
            };
            for lhs in lhs_list {
                canonical_nt_map.insert(lhs, *first);
            }
            non_canonical_nts.extend_from_slice(rest);
        }

        // Map all rules to their canonical non-terminal
        for rule_id in non_canonical_nts {
            let referenced_in = self.rules[rule_id as usize]
                .used_in
                .keys()
                .cloned()
                .collect::<Vec<_>>();
            for loc in referenced_in {
                let used_in_rule = &self.rules[loc.rule as usize];
                let mut rhs = used_in_rule.rhs[loc.index as usize].to_vec();
                for symbol in &mut rhs {
                    if let Symbol::NonTerminal(nt) = symbol {
                        if *nt == rule_id {
                            self.remove_reference(&Symbol::NonTerminal(*nt), loc);
                            *nt = canonical_nt_map[&rule_id];
                            self.add_reference(&Symbol::NonTerminal(*nt), loc);
                        }
                    }
                }
                self.rules[loc.rule as usize].rhs[loc.index as usize] = rhs.into();
            }
            changed = true;
            self.remove(rule_id);
        }

        changed
    }

    pub fn inline_single_use_non_terminals(&mut self) -> bool {
        let mut changed = false;
        let mut to_be_inlined = FxHashSet::default();
        for (_, rule) in &self.rules {
            if rule.used_in.len() == 1 && rule.rhs.len() == 1 {
                to_be_inlined.insert(rule.lhs);
            }
        }
        while let Some(pos) = to_be_inlined
            .iter()
            .find(|pos| {
                !self.rules[**pos as usize].rhs.iter().any(|(_, r)| {
                    !r.iter().any(
                        |s| matches!(s, Symbol::NonTerminal(nt) if to_be_inlined.contains(nt) || *nt != **pos),
                    )
                })
            }) {
            let rule =self.rules[*pos as usize].clone();
            to_be_inlined.remove(&rule.lhs);

            let (loc, _) = rule.used_in.iter().next().unwrap();
            let (_, rule_rhs) = rule.rhs.iter().next().unwrap();
            let used_in_rule = &self.rules[loc.rule as usize];
            let rhs = &used_in_rule.rhs[loc.index as usize];
            let mut new_rhs = Vec::new();
            for symbol in &**rhs {
                if let Symbol::NonTerminal(nt) = symbol {
                    if *nt == rule.lhs {
                        new_rhs.extend(rule_rhs.iter().cloned());
                        continue;
                    }
                }
                new_rhs.push(symbol.clone());
            }
            self.remove_reference(&Symbol::NonTerminal(rule.lhs), *loc);
            for symbol in rule_rhs.iter() {
                self.add_reference(symbol, *loc);
            }
            self.rules[loc.rule as usize].rhs[loc.index as usize] = new_rhs.into();
            changed = true;
            self.remove(rule.lhs);
        }
        changed
    }

    pub fn inline_simple(&mut self) -> bool {
        let mut changed = false;
        let mut nt_to_remove = FxHashSet::default();
        let start = self.start;
        for (_, rule) in &self.rules {
            if rule.rhs.len() == 1 && rule.lhs != start {
                let (_, rhs) = rule.rhs.iter().next().unwrap();
                if rhs.len() == 1 {
                    nt_to_remove.insert(rule.lhs);
                }
            }
        }

        // Update all rules to replace the non-terminal with the terminal
        while let Some(nt) = nt_to_remove.iter().copied().find(|pos| {
            !self.rules[*pos as usize].rhs.iter().any(|(_, r)| {
                !r.iter().any(
                    |s| matches!(s, Symbol::NonTerminal(nt) if nt_to_remove.contains(nt) && *nt != *pos),
                )
            })
        }) {
            let rule = &self.rules[nt as usize];
            let (_, rhs) = rule.rhs.iter().next().unwrap();
            let symbol = if let [symbol] = rhs.as_ref() {
                symbol.clone()
            } else {
                continue;
            };
            let locations = rule.used_in.keys().cloned().collect::<Vec<_>>();
            nt_to_remove.remove(&rule.lhs);
            for loc in locations {
                let used_in_rule = &self.rules[loc.rule as usize];
                let rhs = &used_in_rule.rhs[loc.index as usize];
                let mut new_rhs = Vec::new();
                for s in &**rhs {
                    if let Symbol::NonTerminal(nt2) = s {
                        if *nt2 == nt {
                            new_rhs.push(symbol.clone());
                            continue;
                        }
                    }
                    new_rhs.push(s.clone());
                }
                self.remove_reference(&Symbol::NonTerminal(nt), loc);
                self.add_reference(&symbol, loc);
                self.rules[loc.rule as usize].rhs[loc.index as usize] = new_rhs.into();
            }
            self.remove(nt);
            changed = true;
        }

        changed
    }

    pub fn inline_optimize(&mut self) {
        loop {
            let mut changed = self.inline_single_use_non_terminals();
            self.verify_integrity("after gc");
            changed |= self.inline_simple();
            self.verify_integrity("after inline");
            changed |= self.garbage_collect_non_terminals();
            self.verify_integrity("after inline");
            changed |= self.deduplicate_non_terminals();
            self.verify_integrity("after deduplicate");
            if !changed {
                break;
            }
        }
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

    pub fn verify_integrity(&self, msg: &str) {
        if cfg!(not(debug_assertions)) {
            return;
        }

        // Make sure the start rule exists
        assert!(
            self.rules.contains(self.start as usize),
            "Start rule {} does not exist in the rules {msg}",
            self.start
        );

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

        self.verify_reference_integrity(msg)
    }

    fn verify_reference_integrity(&self, msg: &str) {
        let correct_terminal_locations: FxHashMap<u32, FxHashMap<UsedLocation, usize>> = self
            .rules
            .iter()
            .flat_map(|(_, rule)| {
                rule.rhs.iter().flat_map(move |(i, rhs)| {
                    rhs.iter().filter_map(move |symbol| {
                        if let Symbol::Terminal(token) = symbol {
                            Some((
                                (
                                    *token,
                                    UsedLocation {
                                        rule: rule.lhs,
                                        index: i as u32,
                                    },
                                ),
                                1,
                            ))
                        } else {
                            None
                        }
                    })
                })
            })
            .fold(FxHashMap::default(), |mut acc, (key, count)| {
                *acc.entry(key.0).or_default().entry(key.1).or_default() += count;
                acc
            });
        assert_eq!(
            self.terminal_locations, correct_terminal_locations,
            "Terminal locations do not match the rules {msg}"
        );

        let correct_non_terminal_locations: FxHashMap<u32, FxHashMap<UsedLocation, usize>> = self
            .rules
            .iter()
            .flat_map(|(_, rule)| {
                rule.rhs.iter().flat_map(move |(i, rhs)| {
                    rhs.iter().filter_map(move |symbol| {
                        if let Symbol::NonTerminal(nt) = symbol {
                            Some((
                                (
                                    *nt,
                                    UsedLocation {
                                        rule: rule.lhs,
                                        index: i as u32,
                                    },
                                ),
                                1,
                            ))
                        } else {
                            None
                        }
                    })
                })
            })
            .fold(FxHashMap::default(), |mut acc, (key, count)| {
                *acc.entry(key.0).or_default().entry(key.1).or_default() += count;
                acc
            });

        for (nt, rule) in &self.rules {
            assert_eq!(
                rule.used_in,
                correct_non_terminal_locations
                    .get(&(nt as u32))
                    .cloned()
                    .unwrap_or_default(),
                "Non-terminal locations do not match the rules for non-terminal {} {msg}",
                nt
            );
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
