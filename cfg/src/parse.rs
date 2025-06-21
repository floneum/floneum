#![warn(missing_docs)]

use std::fmt::{Display, Formatter};

use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::{tag, take_till, take_while1},
    character::complete::{multispace0, multispace1, space0, space1},
    combinator::{map, opt},
    multi::{many0, separated_list1},
    sequence::{delimited, preceded, separated_pair, terminated},
};

/// A single grammar symbol.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Symbol<T = String> {
    /// A non‑terminal `A`.
    NonTerminal(String),
    /// A terminal literal `'+'`.
    Terminal(T),
    /// The empty string `ε`.
    Epsilon,
}

impl<T: Display> Display for Symbol<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Symbol::NonTerminal(id) => write!(f, "{}", id),
            Symbol::Terminal(lit) => write!(f, "'{}'", lit),
            Symbol::Epsilon => write!(f, "ε"),
        }
    }
}

/// One production rule: *lhs → rhs1 | rhs2 | …*
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule<T = String> {
    /// The left‑hand non‑terminal.
    pub lhs: String,
    /// Alternative right‑hand sides, each a vector of symbols composing a *sequence*.
    pub rhs: Vec<Vec<Symbol<T>>>, // sequence list
}

impl<T: Display> Display for Rule<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -> {}",
            self.lhs,
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

/// A parsed context‑free grammar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grammar<T = String> {
    pub start: String,
    /// Map `lhs → rule` (keeps original order via `Vec`).
    pub rules: Vec<Rule<T>>,
}

impl Grammar {
    /// Parses the given source string into a [`Grammar`].
    pub fn parse(src: &str) -> Result<Self, String> {
        // Parse using top‑level parser and ensure we consumed all meaningful input.
        let (rest, g) = all_consuming_ws(parse_grammar)
            .parse(src)
            .map_err(|e| format!("parser error: {e:?}"))?;
        if !rest.trim().is_empty() {
            return Err(format!("unexpected trailing input: {rest:?}"));
        }

        Ok(g)
    }

    #[cfg(test)]
    fn rule(&self, lhs: &str) -> Option<&Rule> {
        self.rules.iter().find(|r| r.lhs == lhs)
    }
}

impl<T: Display> Display for Grammar<T> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        writeln!(f, "start: {}", self.start)?;
        for rule in &self.rules {
            writeln!(f, "{}", rule)?;
        }
        Ok(())
    }
}

/// Wrap a parser so that it skips leading & trailing spaces.
fn ws<'a, F, O>(inner: F) -> impl Parser<&'a str, Output = O, Error = nom::error::Error<&'a str>>
where
    F: Parser<&'a str, Output = O, Error = nom::error::Error<&'a str>>,
{
    delimited(space0, inner, space0)
}

/// Consume a *single* line comment (`# …\n`). Returned as `()`.
fn comment(input: &str) -> IResult<&str, ()> {
    map(preceded(tag("#"), take_till(|c| c == '\n')), |_| ()).parse(input)
}

/// Either `multispace+` or a comment.
fn junk(input: &str) -> IResult<&str, ()> {
    map(alt((multispace1.map(|_| ()), comment)), |_| ()).parse(input)
}

/// Zero or more spaces/comments.
fn skip_junk(input: &str) -> IResult<&str, ()> {
    map(many0(junk), |_| ()).parse(input)
}

/// A helper combinator that runs `skip_junk` then `parser`.
fn all_consuming_ws<'a, F, O>(
    parser: F,
) -> impl Parser<&'a str, Output = O, Error = nom::error::Error<&'a str>>
where
    F: Parser<&'a str, Output = O, Error = nom::error::Error<&'a str>>,
{
    preceded(skip_junk, parser)
}

/// Parse an **identifier** (non‑terminal).
fn identifier(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| c.is_ascii_alphabetic() || c == '_')(input).and_then(|(rest, _)| {
        use nom::bytes::complete::take_while;

        let (rest2, _) =
            take_while(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-')(rest)?;
        Ok((rest2, &input[..input.len() - rest2.len()]))
    })
}

/// `'terminal'` – a single quoted literal with backslash escaping.
fn terminal_literal(input: &str) -> IResult<&str, &str> {
    let (rest, lit) = delimited(
        tag("'"),
        map(
            take_while1(|c| c != '\\' && c != '\n' && c != '\''),
            |s: &str| s,
        ),
        tag("'"),
    )
    .parse(input)?;
    Ok((rest, lit))
}

/// The special empty string symbol.
fn epsilon(input: &str) -> IResult<&str, Symbol> {
    map(tag("ε"), |_| Symbol::Epsilon).parse(input)
}

/// Any *symbol*: non‑terminal | terminal | ε.
fn symbol(input: &str) -> IResult<&str, Symbol> {
    // Only skip leading spaces/comments—do NOT swallow the whitespace that separates symbols.
    preceded(
        space0,
        alt((
            map(epsilon, |s| s),
            map(identifier, |id: &str| Symbol::NonTerminal(id.to_string())),
            map(terminal_literal, |t: &str| Symbol::Terminal(t.to_string())),
        )),
    )
    .parse(input)
}

/// A **sequence** ‑ at least one symbol separated by space(s).
fn sequence(input: &str) -> IResult<&str, Vec<Symbol>> {
    separated_list1(space1, symbol).parse(input)
}

/// RHS alternatives separated by `|`.
fn rhs_alternatives(input: &str) -> IResult<&str, Vec<Vec<Symbol>>> {
    separated_list1(ws(tag("|")), sequence).parse(input)
}

/// One production rule: `LHS -> RHS`.
fn rule(input: &str) -> IResult<&str, Rule> {
    map(
        all_consuming_ws(separated_pair(
            delimited(multispace0, identifier, space0),
            ws(tag("->")),
            rhs_alternatives,
        )),
        |(lhs, rhs)| Rule {
            lhs: lhs.to_string(),
            rhs,
        },
    )
    .parse(input)
}

/// Optional `start <id>` directive.
fn start_directive(input: &str) -> IResult<&str, &str> {
    preceded(ws(tag("start")), identifier).parse(input)
}

/// Full grammar.
fn parse_grammar(input: &str) -> IResult<&str, Grammar> {
    let (input, _) = skip_junk(input)?;

    // Optional `start`. We must capture the result but not fail if absent.
    let (input, start_sym_opt) = opt(terminated(start_directive, skip_junk)).parse(input)?;

    // At least one rule.
    let (input, rules) = many0(terminated(rule, skip_junk)).parse(input)?;
    if rules.is_empty() {
        return Err(nom::Err::Failure(nom::error::Error::new(
            "failed to parse grammar: no rules found",
            nom::error::ErrorKind::Fail,
        )));
    }

    let start = start_sym_opt
        .map(|s| s.to_string())
        .unwrap_or_else(|| rules[0].lhs.clone());

    Ok((input, Grammar { start, rules }))
}

#[cfg(test)]
mod tests {
    use super::*;

    static EXAMPLE: &str = r#"
        # G -> a* b* (classic)
        start S

        S -> A B | 'c'
        A -> 'a' A | ε
        B -> 'b' B | ε
    "#;

    #[test]
    fn parses_example() {
        let g = Grammar::parse(EXAMPLE).expect("parse ok");
        println!("{:#?}", g);
        assert_eq!(g.start, "S");
        assert_eq!(g.rules.len(), 3);
        assert_eq!(g.rule("A").unwrap().rhs.len(), 2);
    }

    #[test]
    fn epsilon_only() {
        let src = "S -> ε";
        let g = Grammar::parse(src).unwrap();
        let alt = &g.rule("S").unwrap().rhs[0];
        assert_eq!(alt, &[Symbol::Epsilon]);
    }
}
