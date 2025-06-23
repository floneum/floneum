use std::fmt::Display;

use crate::{
    parse::Grammar,
    tokenizer::{Merge, Tokenizer},
};

mod cnf;
mod parse;
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

    let cnf_grammar = grammar.to_cnf().unwrap();
    let bump = bumpalo::Bump::new();
    let cnf_grammar = cnf_grammar.replace_tokenizer_terminals(&tokenizer);
    let mut cnf_grammar = cnf_grammar.shortcut_merge(&Merge {
        rank: 0,
        pair: [
            tokenizer.bytes[b't' as usize],
            tokenizer.bytes[b'o' as usize],
        ],
        new_token: 10_000,
    });
    cnf_grammar.garbage_collect_non_terminals();
    println!("CNF grammar:\n{}", cnf_grammar);
    let dense_grammar = cnf_grammar.reallocate(&bump);
    println!("dense size: {}", bump.allocated_bytes());

    assert!(dense_grammar.recognizes(b"0", &tokenizer));
    assert!(dense_grammar.recognizes(b"1", &tokenizer));
    assert!(dense_grammar.recognizes(b"2", &tokenizer));
    assert!(dense_grammar.recognizes(b"(+ 1 2)", &tokenizer));
    assert!(dense_grammar.recognizes(b"(- 2 1)", &tokenizer));
    assert!(dense_grammar.recognizes(b"(str.len name)", &tokenizer));
    assert!(!dense_grammar.recognizes(b"(str.to.int name)", &tokenizer));

    let mut recognizer = Recognizer::new(&dense_grammar, &bump);
    let mut text = String::new();
    loop {
        let mut new_text = String::new();
        std::io::stdin().read_line(&mut new_text).unwrap();
        let new_text = new_text.trim_end_matches('\n');
        for byte in new_text.as_bytes() {
            // map the byte to the tokenizer's vocabulary
            let token = tokenizer.bytes[*byte as usize];
            text.push(*byte as char);
            if recognizer.push_byte(token) || recognizer.finish() {
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
    pub fn recognizes(&self, input: &[u8], tokenizer: &Tokenizer) -> bool {
        let bump = bumpalo::Bump::new();
        let mut recognizer = Recognizer::new(self, &bump);

        for byte in input {
            recognizer.push_byte(tokenizer.bytes[*byte as usize]);
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
        let mut myself = Grammar {
            start: self.start.clone(),
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
                            new_sequence.push(parse::Symbol::NonTerminal(nt.clone()));
                        }
                        parse::Symbol::Epsilon => {
                            new_sequence.push(parse::Symbol::Epsilon);
                        }
                    }
                }
                new_rhs.push(new_sequence);
            }
            myself.rules.push(parse::Rule {
                lhs: rule.lhs.clone(),
                rhs: new_rhs,
            });
        }

        myself
    }
}

impl<T> Grammar<T> {
    fn garbage_collect_non_terminals(&mut self) {
        // Remove any non-terminals that are not used in any rules
        let mut used_non_terminals = std::collections::HashSet::new();
        let mut queue = vec![self.start.clone()];
        while let Some(nt) = queue.pop() {
            if used_non_terminals.insert(nt.clone()) {
                for rule in &self.rules {
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
            .retain(|rule| used_non_terminals.contains(&rule.lhs));
    }
}

impl Grammar<u32> {
    // pub struct Merge {
    //     rank: u32,
    //     pair: [u32; 2],
    //     new_token: u32,
    // }
    pub fn shortcut_merge(&self, merge: &Merge) -> Grammar<u32> {
        // Since our grammar is in CNF, we can only need to handle two cases:
        // Q1 -> T - this is handles by `replace_tokenizer_terminals`
        // Q2 -> Q1 Q2 - this is handled by the loop below
        // First split every Q2 into two rules:
        // Q2beforetoken1 -> Q1 Q2
        // Q2aftertoken1 -> Q1 Q2

        let mut new_rules = Vec::new();

        #[derive(Debug)]
        struct NewRule {
            after_first_token_incoming: bool,
            after_first_token_outgoing: bool,
            lhs: String,
            rhs: Vec<Vec<parse::Symbol<u32>>>,
        }

        for rule in &*self.rules {
            for after_first_token_incoming in [false, true] {
                for after_first_token_outgoing in [false, true] {
                    let mut new_rule = NewRule {
                        after_first_token_incoming,
                        after_first_token_outgoing,
                        lhs: rule.lhs.clone(),
                        rhs: Vec::new(),
                    };
                    let mut new_rhs = Vec::new();
                    for rhs in &*rule.rhs {
                        let mut new_sequence = Vec::new();
                        for symbol in &**rhs {
                            match symbol {
                                parse::Symbol::Terminal(lit) => {
                                    // If this is a terminal that matches the first token, the output must have the first token true
                                    if *lit == merge.pair[0] {
                                        if new_rule.after_first_token_outgoing {
                                            new_sequence.push(parse::Symbol::Terminal(*lit));
                                        }
                                    } else if *lit == merge.pair[1] {
                                        // If this is a terminal that matches the second token, and we are after the first token,
                                        // the first token must be false next
                                        if !new_rule.after_first_token_outgoing {
                                            // If we are already after the second token, we just push the terminal
                                            // TODO: This is where we need to create the shortcut rule from the last terminal to this rule
                                            // new_sequence.push(parse::Symbol::Terminal(*lit));
                                        }
                                    } else {
                                        if !new_rule.after_first_token_outgoing {
                                            // After the first token must be false
                                            new_sequence.push(parse::Symbol::Terminal(*lit));
                                        }
                                    }
                                }
                                parse::Symbol::NonTerminal(nt) => {
                                    new_sequence.push(parse::Symbol::NonTerminal(nt.clone()));
                                }
                                parse::Symbol::Epsilon => {
                                    new_sequence.push(parse::Symbol::Epsilon);
                                }
                            }
                        }
                        new_rhs.push(new_sequence);
                    }
                    new_rule.rhs = new_rhs;
                    new_rules.push(new_rule);
                }
            }
        }

        // Then run through each new rule and eliminate impossible states. Keep going until there are no changes.
        let mut changed = true;
        while changed {
            changed = false;
            println!("iteration with {} rules", new_rules.len());
            let mut index = 0;
            while index < new_rules.len() {
                let rule = &new_rules[index];
                let after_first_token_incoming = rule.after_first_token_incoming;
                let after_first_token_outgoing = rule.after_first_token_outgoing;
                // Filter out any invalid right-hand sides
                let mut rhs_index = 0;
                while rhs_index < new_rules[index].rhs.len() {
                    let sequence = new_rules[index].rhs[rhs_index].clone();
                    match sequence.as_slice() {
                        // Since we are in CNF, we can only have two symbols in the sequence
                        [
                            parse::Symbol::NonTerminal(first_nt),
                            parse::Symbol::NonTerminal(second_nt),
                        ] => {
                            // Find the non-terminals with a matching after_first_token_incoming
                            // and keep track of what the intermediate_after_first_token_states state should be
                            let valid_intermediate_after_first_token_states = new_rules
                                .iter()
                                .filter_map(|r| {
                                    (r.lhs == *first_nt
                                        && r.after_first_token_incoming
                                            == after_first_token_incoming)
                                        .then(|| r.after_first_token_outgoing)
                                })
                                .collect::<Vec<_>>();
                            if valid_intermediate_after_first_token_states.is_empty() {
                                // If there are no valid states for the first non-terminal, remove this rule
                                new_rules[index].rhs.remove(rhs_index);
                                changed = true;
                                continue;
                            }
                            // Find the non-terminals with a matching intermediate_after_first_token_states and
                            // matching after_first_token_outgoing
                            let has_valid_second_non_terminal = new_rules.iter().any(|r| {
                                r.lhs == *second_nt
                                    && valid_intermediate_after_first_token_states
                                        .contains(&r.after_first_token_incoming)
                                    && r.after_first_token_outgoing == after_first_token_outgoing
                            });
                            if !has_valid_second_non_terminal {
                                // If there are no valid states for the second non-terminal, remove this rule
                                new_rules[index].rhs.remove(rhs_index);
                                changed = true;
                                continue;
                            }
                        }
                        [] => {
                            // Remove empty sequences
                            new_rules[index].rhs.remove(rhs_index);
                            changed = true;
                            continue;
                        }
                        // Or one terminal or epsilon which are always valid or removed above
                        [parse::Symbol::Terminal(_)] | [parse::Symbol::Epsilon] => {}
                        _ => unreachable!("Unexpected sequence in CNF: {:?}", sequence),
                    }
                    rhs_index += 1;
                }

                // If there are no valid right-hand sides left, remove the rule
                let rhs = &new_rules[index].rhs;
                if rhs.is_empty() {
                    new_rules.remove(index);
                    changed = true;
                } else {
                    index += 1; // Move to the next rule
                }
            }
        }

        println!("New rules after merge:\n{:#?}", new_rules);

        // Finally, split up all of the rules into a new grammar
        let mut new_grammar_rules = Vec::new();

        fn format_new_rule(rule: &NewRule) -> String {
            let NewRule {
                after_first_token_incoming,
                after_first_token_outgoing,
                lhs,
                ..
            } = rule;
            format!(
                "{}_{}_{}",
                after_first_token_incoming, lhs, after_first_token_outgoing
            )
        }

        for rule in &new_rules {
            let NewRule {
                after_first_token_incoming,
                after_first_token_outgoing,
                rhs,
                ..
            } = rule;
            let new_lhs = format_new_rule(rule);
            let mut new_rhs = Vec::new();

            for rhs in rhs {
                // Split up the right-hand sides into all possible combinations
                match rhs.as_slice() {
                    // Since we are in CNF, we can only have two symbols in the sequence
                    [
                        parse::Symbol::NonTerminal(first_nt),
                        parse::Symbol::NonTerminal(second_nt),
                    ] => {
                        // Find the non-terminals with a matching after_first_token_incoming
                        // and keep track of what the intermediate_after_first_token_states state should be
                        for valid_first_token in new_rules.iter().filter(|r| {
                            r.lhs == *first_nt
                                && r.after_first_token_incoming == *after_first_token_incoming
                        }) {
                            for valid_second_token in new_rules.iter().filter(|r| {
                                r.lhs == *second_nt
                                    && valid_first_token.after_first_token_outgoing
                                        == r.after_first_token_incoming
                                    && r.after_first_token_outgoing == *after_first_token_outgoing
                            }) {
                                // Create a new right-hand side with the valid non-terminals
                                let new_sequence = vec![
                                    parse::Symbol::NonTerminal(format_new_rule(valid_first_token)),
                                    parse::Symbol::NonTerminal(format_new_rule(valid_second_token)),
                                ];
                                new_rhs.push(new_sequence);
                            }
                        }
                    }
                    // Just push other valid sequences as they are
                    [parse::Symbol::Terminal(_)] | [parse::Symbol::Epsilon] => {
                        // Valid sequences with one terminal or epsilon are always valid
                        new_rhs.push(rhs.clone());
                    }
                    _ => unreachable!("Unexpected sequence in CNF: {:?}", rhs),
                }
            }

            // Add the new rule to the grammar
            new_grammar_rules.push(parse::Rule {
                lhs: new_lhs,
                rhs: new_rhs,
            });
        }

        // Create a new start rule
        let start_lhs: Vec<_> = new_rules.iter().filter(|r| r.lhs == self.start).collect();
        let new_start_lhs = "Start".to_string();
        let new_start = parse::Rule {
            lhs: new_start_lhs.clone(),
            rhs: start_lhs
                .iter()
                .map(|r| vec![parse::Symbol::NonTerminal(format_new_rule(r))])
                .collect(),
        };

        new_grammar_rules.push(new_start);

        Grammar {
            start: new_start_lhs,
            rules: new_grammar_rules,
        }
    }

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
