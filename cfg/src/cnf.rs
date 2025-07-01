use std::{collections::{HashMap, HashSet}, fmt::Display, hash::Hash};

use crate::parse::{Grammar, Rule, Symbol};

impl Grammar {
    /// Step 0: Split up each literal into single character terminals
    pub fn split_terminals(self) -> Self {
        let rules = self
            .rules
            .iter()
            .map(|rule| {
                let new_rhs: Vec<Vec<Symbol>> = rule
                    .rhs
                    .iter()
                    .map(|alt| {
                        alt.iter()
                            .flat_map(|sym| match sym {
                                Symbol::Terminal(t) if t.len() > 1 => t
                                    .chars()
                                    .map(|c| Symbol::Terminal(c.to_string()))
                                    .collect::<Vec<_>>(),
                                _ => vec![sym.clone()],
                            })
                            .collect()
                    })
                    .collect();
                Rule {
                    lhs: rule.lhs.clone(),
                    rhs: new_rhs,
                }
            })
            .collect();

        Grammar {
            start: self.start.clone(),
            rules,
        }
    }
}

impl<T: Clone + Eq + Hash + Display> Grammar<T> {
    /// Converts the grammar to **Chomsky Normal Form (CNF)** by invoking
    /// each conversion step in sequence.
    pub fn to_cnf(&self) -> Result<Grammar<T>, String> {
        let rules = self.rules.clone();

        // Step 1: Add a fresh start symbol.
        let (rules_step1, mut nonterminals, new_start) =
            self.add_new_start(rules, self.start.clone())?;

        // Step 2: Eliminate ε-productions.
        let rules_step2 = self.eliminate_epsilon(rules_step1, &new_start);

        // Step 3: Eliminate unit productions.
        let rules_step3 = self.eliminate_units(rules_step2, &nonterminals);

        // Step 4: Replace terminals in RHS of length > 1.
        let rules_step4 = self.replace_terminals(rules_step3, &mut nonterminals);

        // Step 5: Binarize productions with RHS length > 2.
        let rules_step5 = self.binarize(rules_step4, &mut nonterminals);

        Ok(Grammar {
            start: new_start,
            rules: rules_step5,
        })
    }

    /// Step 1: Introduce a fresh start symbol `S0 → original_start`.
    ///
    /// Returns a tuple `(new_rules, nonterminals, new_start_symbol)`.
    fn add_new_start(
        &self,
        rules: Vec<Rule<T>>,
        start: String,
    ) -> Result<(Vec<Rule<T>>, HashSet<String>, String), String> {
        // Collect all existing non-terminals.
        let mut nonterminals: HashSet<String> = rules.iter().map(|r| r.lhs.clone()).collect();

        // Generate a fresh start symbol not in `nonterminals`.
        let mut new_start = format!("{}0", start);
        while nonterminals.contains(&new_start) {
            new_start.push('0');
            if new_start.len() > 50 {
                return Err("Unable to generate a fresh start symbol".into());
            }
        }
        nonterminals.insert(new_start.clone());

        // Build the new rule list: first the new start rule, then all original rules.
        let mut new_rules = Vec::new();
        new_rules.push(Rule {
            lhs: new_start.clone(),
            rhs: vec![vec![Symbol::NonTerminal(start.clone())]],
        });
        for rule in &rules {
            new_rules.push(rule.clone());
        }

        Ok((new_rules, nonterminals, new_start))
    }

    /// Step 2: Eliminate ε-productions (nullable non-terminals), except possibly for the new start symbol.
    fn eliminate_epsilon(&self, rules: Vec<Rule<T>>, new_start: &String) -> Vec<Rule<T>> {
        // (a) Compute the set of nullable non-terminals.
        let mut nullable: HashSet<String> = HashSet::new();
        for r in &rules {
            for alt in &r.rhs {
                if alt == &[Symbol::Epsilon] {
                    nullable.insert(r.lhs.clone());
                }
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for r in &rules {
                if nullable.contains(&r.lhs) {
                    continue;
                }
                for alt in &r.rhs {
                    if alt.iter().all(|sym| match sym {
                        Symbol::NonTerminal(nt) => nullable.contains(nt),
                        Symbol::Epsilon => true,
                        Symbol::Terminal(_) => false,
                    }) {
                        nullable.insert(r.lhs.clone());
                        changed = true;
                        break;
                    }
                }
            }
        }

        // (b) Rebuild each rule’s RHS by dropping nullable symbols in all combinations.
        let mut after_eps_elim: Vec<Rule<T>> = Vec::new();
        for r in &rules {
            let mut new_alternatives: HashSet<Vec<Symbol<T>>> = HashSet::new();
            for alt in &r.rhs {
                // Collect positions of nullable non-terminals within this alt.
                let nullable_positions: Vec<usize> = alt
                    .iter()
                    .enumerate()
                    .filter_map(|(i, sym)| match sym {
                        Symbol::NonTerminal(nt) if nullable.contains(nt) => Some(i),
                        _ => None,
                    })
                    .collect();
                // There are 2^(#nullable_positions) ways to drop or keep each nullable symbol.
                let subsets = 1 << nullable_positions.len();
                for mask in 0..subsets {
                    let mut new_rhs: Vec<Symbol<T>> = Vec::with_capacity(alt.len());
                    for (i, sym) in alt.iter().enumerate() {
                        if let Some(idx_pos) = nullable_positions.iter().position(|&p| p == i) {
                            if (mask & (1 << idx_pos)) != 0 {
                                continue;
                            }
                        }
                        new_rhs.push(sym.clone());
                    }
                    if new_rhs.is_empty() {
                        // If everything was dropped, we get ε. Only keep that if r.lhs == new_start.
                        if r.lhs == *new_start {
                            new_alternatives.insert(vec![Symbol::Epsilon]);
                        }
                    } else {
                        new_alternatives.insert(new_rhs);
                    }
                }
            }
            // If this rule’s head is not the new_start, remove the ε alternative entirely.
            if r.lhs != *new_start {
                new_alternatives.remove(&vec![Symbol::Epsilon]);
            }
            after_eps_elim.push(Rule {
                lhs: r.lhs.clone(),
                rhs: new_alternatives.into_iter().collect(),
            });
        }
        after_eps_elim
    }

    /// Step 3: Eliminate unit productions (A → B where B is a non-terminal).
    fn eliminate_units(&self, rules: Vec<Rule<T>>, nonterminals: &HashSet<String>) -> Vec<Rule<T>> {
        // (a) Build a graph of unit edges A → B.
        let mut unit_graph: HashMap<String, HashSet<String>> = HashMap::new();
        for r in &rules {
            unit_graph.entry(r.lhs.clone()).or_default();
            for alt in &r.rhs {
                if alt.len() == 1 {
                    if let Symbol::NonTerminal(nt) = &alt[0] {
                        unit_graph
                            .entry(r.lhs.clone())
                            .or_default()
                            .insert(nt.clone());
                    }
                }
            }
        }

        // (b) Compute the transitive closure (unit-closure) for each non-terminal.
        let mut unit_closure: HashMap<String, HashSet<String>> = HashMap::new();
        for nt in nonterminals {
            let mut closure: HashSet<String> = HashSet::new();
            let mut stack = vec![nt.clone()];
            while let Some(curr) = stack.pop() {
                if closure.insert(curr.clone()) {
                    if let Some(neighbors) = unit_graph.get(&curr) {
                        for n in neighbors {
                            stack.push(n.clone());
                        }
                    }
                }
            }
            unit_closure.insert(nt.clone(), closure);
        }

        // (c) Gather all non-unit productions.
        let mut nonunit_rules: Vec<Rule<T>> = Vec::new();
        for r in &rules {
            let filtered: Vec<Vec<Symbol<T>>> = r
                .rhs
                .iter()
                .filter(|alt| !(alt.len() == 1 && matches!(&alt[0], Symbol::NonTerminal(_))))
                .cloned()
                .collect();
            if !filtered.is_empty() {
                nonunit_rules.push(Rule {
                    lhs: r.lhs.clone(),
                    rhs: filtered,
                });
            }
        }

        // (d) For each non-terminal A, gather all non-unit productions of any B in closure(A).
        let mut expanded: HashMap<String, HashSet<Vec<Symbol<T>>>> = HashMap::new();
        for nt in nonterminals {
            let mut rhs_set: HashSet<Vec<Symbol<T>>> = HashSet::new();
            if let Some(closure_set) = unit_closure.get(nt) {
                for b in closure_set {
                    for r in &rules {
                        if &r.lhs == b {
                            for alt in &r.rhs {
                                if !(alt.len() == 1 && matches!(&alt[0], Symbol::NonTerminal(_))) {
                                    rhs_set.insert(alt.clone());
                                }
                            }
                        }
                    }
                }
            }
            if !rhs_set.is_empty() {
                expanded.insert(nt.clone(), rhs_set);
            }
        }

        // (e) Build the set of rules after unit-elimination.
        let mut after_unit: Vec<Rule<T>> = Vec::new();
        for (nt, prods) in expanded {
            after_unit.push(Rule {
                lhs: nt.clone(),
                rhs: prods.into_iter().collect(),
            });
        }
        after_unit
    }

    /// Step 4: Replace terminals in any RHS of length > 1 with fresh non-terminals.
    fn replace_terminals(&self, rules: Vec<Rule<T>>, nonterminals: &mut HashSet<String>) -> Vec<Rule<T>> {
        let mut terminal_map: HashMap<T, String> = HashMap::new();
        let mut replaced: Vec<Rule<T>> = Vec::new();

        for r in &rules {
            let mut new_alts: Vec<Vec<Symbol<T>>> = Vec::new();
            for alt in &r.rhs {
                if alt.len() > 1 {
                    // Replace each terminal in a long alt with a fresh non-terminal.
                    let mut new_alt: Vec<Symbol<T>> = Vec::with_capacity(alt.len());
                    for sym in alt {
                        match sym {
                            Symbol::Terminal(t) => {
                                if !terminal_map.contains_key(t) {
                                    let base = format!("T_{}", t);
                                    let mut nt_candidate = base.clone();
                                    while nonterminals.contains(&nt_candidate) {
                                        nt_candidate.push('_');
                                        if nt_candidate.len() > 50 {
                                            // In practice, choose another strategy. For now, panic.
                                            panic!(
                                                "Unable to generate a fresh terminal non-terminal"
                                            );
                                        }
                                    }
                                    nonterminals.insert(nt_candidate.clone());
                                    terminal_map.insert(t.clone(), nt_candidate.clone());
                                }
                                let nt_name = terminal_map.get(t).unwrap().clone();
                                new_alt.push(Symbol::NonTerminal(nt_name));
                            }
                            _ => new_alt.push(sym.clone()),
                        }
                    }
                    new_alts.push(new_alt);
                } else {
                    // length ≤ 1 or length == 2: leave as-is
                    new_alts.push(alt.clone());
                }
            }
            replaced.push(Rule {
                lhs: r.lhs.clone(),
                rhs: new_alts,
            });
        }

        // Add the new terminal rules: for each (terminal → non-terminal), add rule NT → ['terminal'].
        for (t, nt) in &terminal_map {
            replaced.push(Rule {
                lhs: nt.clone(),
                rhs: vec![vec![Symbol::Terminal(t.clone())]],
            });
        }

        replaced
    }

    /// Step 5: Binarize any production whose RHS has length > 2 by introducing intermediate non-terminals.
    fn binarize(&self, rules: Vec<Rule<T>>, nonterminals: &mut HashSet<String>) -> Vec<Rule<T>> {
        let mut cnf_rules: Vec<Rule<T>> = Vec::new();
        let mut intermediate_counter = 0;

        for r in &rules {
            let mut new_alts: Vec<Vec<Symbol<T>>> = Vec::new();
            for alt in &r.rhs {
                let mut remaining = alt.clone();
                while remaining.len() > 2 {
                    // Pop off the last two symbols and create a new intermediate non-terminal.
                    let right = remaining.pop().unwrap();
                    let left = remaining.pop().unwrap();
                    let new_symbol = format!("I_{}_{}", r.lhs, intermediate_counter);
                    intermediate_counter += 1;
                    nonterminals.insert(new_symbol.clone());
                    cnf_rules.push(Rule {
                        lhs: new_symbol.clone(),
                        rhs: vec![vec![left, right]],
                    });
                    // Push the new symbol back to remaining for the next iteration.
                    remaining.push(Symbol::NonTerminal(new_symbol));
                }
                new_alts.push(remaining);
            }

            if !new_alts.is_empty() {
                cnf_rules.push(Rule {
                    lhs: r.lhs.clone(),
                    rhs: new_alts,
                });
            }
        }

        cnf_rules
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_new_start() {
        // Grammar:
        //   S -> a
        // Expect a new start S0 → S and original rule preserved.
        let g = Grammar::parse("start S\nS -> 'a'").unwrap();
        let (new_rules, nonterms, new_start) =
            g.add_new_start(g.rules.clone(), g.start.clone()).unwrap();

        // new_start should start with "S0"
        assert!(new_start.starts_with("S0"));
        // nonterms should contain new_start and "S"
        assert!(nonterms.contains("S"));
        assert!(nonterms.contains(&new_start));

        // new_rules[0] should be new_start → [S]
        assert_eq!(new_rules[0].lhs, new_start);
        assert_eq!(
            new_rules[0].rhs,
            vec![vec![Symbol::NonTerminal("S".into())]]
        );
        // new_rules[1] should be S → ['a']
        assert_eq!(new_rules[1].lhs, "S");
        assert_eq!(new_rules[1].rhs, vec![vec![Symbol::Terminal("a".into())]]);
    }

    #[test]
    fn test_eliminate_epsilon() {
        // Grammar:
        //   S0 → S
        //   S  → a S b | ε
        let grammar = Grammar::parse(
            "start S0\n\
             S0 -> S\n\
             S -> 'a' S 'b' | ε",
        )
        .unwrap();
        let rules = grammar.rules;
        let after = Grammar {
            start: "S0".into(),
            rules: vec![],
        }
        .eliminate_epsilon(rules.clone(), &"S0".into());
        println!("After ε-elimination: {:#?}", after);

        // No ε-rhs in any rule except possibly in S0.
        for r in &after {
            for alt in &r.rhs {
                assert!(
                    !(alt.len() == 1 && alt[0] == Symbol::Epsilon && r.lhs != "S0"),
                    "Found stray ε in {}",
                    r.lhs
                );
            }
        }
        // S's alternatives should include: [a S b], [a b], [b], [a], but not [].
        let mut s_alts = Vec::new();
        for r in &after {
            if r.lhs == "S" {
                s_alts.extend(r.rhs.clone());
            }
        }
        // Check presence of expected expansions.
        assert!(s_alts.contains(&vec![
            Symbol::Terminal("a".into()),
            Symbol::NonTerminal("S".into()),
            Symbol::Terminal("b".into())
        ]));
        assert!(s_alts.contains(&vec![
            Symbol::Terminal("a".into()),
            Symbol::Terminal("b".into())
        ]));
    }

    #[test]
    fn test_eliminate_units() {
        // Grammar:
        //   A → B | 'a'
        //   B → C
        //   C → 'c'
        let grammar = Grammar::parse(
            "start A\n\
             A -> B | 'a'\n\
             B -> C\n\
             C -> 'c'",
        )
        .unwrap();
        let rules = grammar.rules;
        let nonterms: HashSet<String> = ["A".into(), "B".into(), "C".into()]
            .iter()
            .cloned()
            .collect();
        let after = Grammar {
            start: "A".into(),
            rules: vec![],
        }
        .eliminate_units(rules.clone(), &nonterms);

        // After eliminating units, A should have ['a'] and ['c'], B should have ['c'].
        let mut got_a = HashSet::new();
        let mut got_b = HashSet::new();
        for r in &after {
            if r.lhs == "A" {
                for alt in &r.rhs {
                    if alt.len() == 1 {
                        if let Symbol::Terminal(t) = &alt[0] {
                            got_a.insert(t.clone());
                        }
                    }
                }
            }
            if r.lhs == "B" {
                for alt in &r.rhs {
                    if alt.len() == 1 {
                        if let Symbol::Terminal(t) = &alt[0] {
                            got_b.insert(t.clone());
                        }
                    }
                }
            }
        }
        assert!(got_a.contains("a"));
        assert!(got_a.contains("c"));
        assert!(got_b.contains("c"));
    }

    #[test]
    fn test_replace_terminals() {
        // Grammar:
        //   S → 'a' B
        //   B → 'b'
        let grammar = Grammar::parse(
            "start S\n\
             S -> 'a' B\n\
             B -> 'b'",
        )
        .unwrap();
        let rules = grammar.rules;
        let mut nonterms: HashSet<String> = ["S".into(), "B".into()].iter().cloned().collect();
        let after: Vec<Rule> = Grammar::<String> {
            start: "S".into(),
            rules: vec![],
        }
        .replace_terminals(rules.clone(), &mut nonterms);
        println!("After terminal replacement: {:#?}", after);

        assert_eq!(
            after,
            [
                Rule {
                    lhs: "S".into(),
                    rhs: vec![vec![
                        Symbol::NonTerminal("T_a".into()),
                        Symbol::NonTerminal("B".into()),
                    ]],
                },
                Rule {
                    lhs: "B".into(),
                    rhs: vec![vec![Symbol::Terminal("b".into())]],
                },
                Rule {
                    lhs: "T_a".into(),
                    rhs: vec![vec![Symbol::Terminal("a".into())]],
                },
            ]
        )
    }

    #[test]
    fn test_binarize() {
        // Grammar:
        //   S → A B C D
        let grammar = Grammar::parse(
            "start S\n\
             S -> A B C D\n\
             A -> 'a'\n\
             B -> 'b'\n\
             C -> 'c'\n\
             D -> 'd'",
        )
        .unwrap();
        let rules = grammar.rules;
        let mut nonterms: HashSet<String> =
            ["S".into(), "A".into(), "B".into(), "C".into(), "D".into()]
                .iter()
                .cloned()
                .collect();
        let after = Grammar::<String> {
            start: "S".into(),
            rules: vec![],
        }
        .binarize(rules.clone(), &mut nonterms);

        println!("After binarization: {:#?}", after);

        // After binarization, every RHS must be length ≤ 2.
        for r in &after {
            for alt in &r.rhs {
                assert!(
                    alt.len() <= 2,
                    "Found RHS length {} in rule {}",
                    alt.len(),
                    r.lhs
                );
            }
        }
    }
}
