use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::{is_not, tag, take_while1},
    character::complete::multispace1,
    combinator::map,
    multi::many0,
    sequence::{delimited, preceded},
};

/// Skip zero-or-more comments (`;; …\n`) or whitespace
fn ws(input: &str) -> IResult<&str, ()> {
    let comment = preceded(tag(";;"), is_not("\n\r"));
    let skip = many0(alt((multispace1, map(comment, |_| ""))));
    map(skip, |_| ()).parse(input)
}

#[test]
fn skip_ws() {
    assert_eq!(ws(""), Ok(("", ())));
    assert_eq!(ws("  "), Ok(("", ())));
    assert_eq!(ws("  \n"), Ok(("", ())));
    assert_eq!(ws("  ;; comment\n"), Ok(("", ())));
    assert_eq!(ws("  ;; comment\n  "), Ok(("", ())));
    assert_eq!(
        ws(";; comment only\nthis should parse"),
        Ok(("this should parse", ()))
    );
}

/// Parse an atomic symbol: letters, digits, or punctuation (simple)
fn atom(input: &str) -> IResult<&str, SExpr> {
    map(
        take_while1(|c: char| !c.is_whitespace() && !['(', ')', ';'].contains(&c)),
        |s: &str| SExpr::Atom(s.to_string()),
    )
    .parse(input)
}

#[test]
fn test_atom() {
    assert_eq!(atom("x"), Ok(("", SExpr::Atom("x".to_string()))));
    assert_eq!(atom("x y"), Ok((" y", SExpr::Atom("x".to_string()))));
}

/// Parse a parenthesized list: '(' SExpr* ')'
fn list(input: &str) -> IResult<&str, SExpr> {
    map(
        delimited(
            preceded(ws, tag("(")),
            many0(preceded(ws, sexpr)),
            preceded(ws, tag(")")),
        ),
        SExpr::List,
    )
    .parse(input)
}

#[test]
fn test_list() {
    assert_eq!(
        list("(x y)"),
        Ok((
            "",
            SExpr::List(vec![
                SExpr::Atom("x".to_string()),
                SExpr::Atom("y".to_string())
            ])
        ))
    );
    assert_eq!(
        list("(x (y z))"),
        Ok((
            "",
            SExpr::List(vec![
                SExpr::Atom("x".to_string()),
                SExpr::List(vec![
                    SExpr::Atom("y".to_string()),
                    SExpr::Atom("z".to_string())
                ])
            ])
        ))
    );
}

/// Generic S-expression parser
fn sexpr(input: &str) -> IResult<&str, SExpr> {
    alt((list, atom)).parse(input) // try list first, then atom :contentReference[oaicite:5]{index=5}
}

#[test]
fn test_sexpr() {
    assert_eq!(
        sexpr("(x y)"),
        Ok((
            "",
            SExpr::List(vec![
                SExpr::Atom("x".to_string()),
                SExpr::Atom("y".to_string())
            ])
        ))
    );
    assert_eq!(sexpr("x"), Ok(("", SExpr::Atom("x".to_string()))));
}

/// Transform an SExpr into a Command
fn to_command(expr: SExpr) -> Option<Command> {
    match expr {
        SExpr::List(items) => match items.split_first()? {
            (SExpr::Atom(cmd), rest) if cmd == "set-logic" => {
                if let [SExpr::Atom(logic)] = rest {
                    Some(Command::SetLogic(logic.clone()))
                } else {
                    None
                }
            }
            (SExpr::Atom(cmd), rest) if cmd == "synth-fun" => {
                // (synth-fun name ((x T) ...) Ret
                //   ((Nt Ty) ...)          ; non-terminals
                //   ((Nt1 ...) (Nt2 ...) …) ; grammar
                let name = match &rest[0] {
                    SExpr::Atom(n) => n.clone(),
                    _ => return None,
                };
                let args = if let SExpr::List(a) = &rest[1] {
                    a.iter()
                        .filter_map(|p| {
                            if let SExpr::List(v) = p {
                                if let [SExpr::Atom(n), SExpr::Atom(t)] = &v[..] {
                                    Some((n.clone(), t.clone()))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    return None;
                };
                let ret_ty = match &rest[2] {
                    SExpr::Atom(t) => t.clone(),
                    _ => return None,
                };
                // non-terms
                let non_terms = if let SExpr::List(nt) = &rest[3] {
                    nt.iter()
                        .filter_map(|p| {
                            if let SExpr::List(v) = p {
                                if let [SExpr::Atom(n), SExpr::Atom(t)] = &v[..] {
                                    Some((n.clone(), t.clone()))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    return None;
                };
                // grammar rules
                let grammar = if let SExpr::List(gr) = &rest[4] {
                    gr.iter()
                        .filter_map(|p| {
                            if let SExpr::List(v) = p {
                                if let [SExpr::Atom(n), SExpr::Atom(ty), SExpr::List(rhs)] = &v[..]
                                {
                                    Some((n.clone(), ty.clone(), rhs.clone()))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    return None;
                };
                Some(Command::SynthFun(SynthFun {
                    name,
                    args,
                    ret_ty,
                    non_terms,
                    grammar,
                }))
            }
            (SExpr::Atom(cmd), [SExpr::Atom(var), SExpr::Atom(ty)]) if cmd == "declare-var" => {
                Some(Command::DeclareVar(var.clone(), ty.clone()))
            }
            (SExpr::Atom(cmd), [c]) if cmd == "constraint" => Some(Command::Constraint(c.clone())),
            (SExpr::Atom(cmd), []) if cmd == "check-synth" => Some(Command::CheckSynth),
            _ => None,
        },
        _ => None,
    }
}

/// Minimal S-expression AST
#[derive(Debug, Clone, PartialEq)]
pub enum SExpr {
    Atom(String),
    List(Vec<SExpr>),
}

/// A synthesized function
#[derive(Debug, Clone, PartialEq)]
struct SynthFun {
    name: String,
    args: Vec<(String, String)>,
    ret_ty: String,
    non_terms: Vec<(String, String)>,
    grammar: Vec<(String, String, Vec<SExpr>)>,
}

/// High-level SyGuS commands
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    SetLogic(String),
    SynthFun(SynthFun),
    DeclareVar(String, String),
    Constraint(SExpr),
    CheckSynth,
}

/// Parse an entire SyGuS input into a Vec<Command>
pub fn parse_sygus(input: &str) -> Result<Vec<Command>, String> {
    let mut cmds = Vec::new();
    let mut rest = input;
    while let Ok((r, sex)) = preceded(ws, sexpr).parse(rest) {
        rest = r;
        if let Some(cmd) = to_command(sex) {
            cmds.push(cmd);
        }
    }
    if rest.trim().is_empty() {
        Ok(cmds)
    } else {
        Err(format!("Failed to parse at: {}", rest))
    }
}

fn main() {
    let src = r#"
      ;; The background theory is linear integer arithmetic
      (set-logic LIA)
      ;; Name and signature of the function to be synthesized
      (synth-fun max2 ((x Int) (y Int)) Int
          ((I Int) (B Bool))
          ((I Int (x y 0 1 (+ I I) (- I I) (ite B I I)))
           (B Bool ((and B B) (or B B) (not B) (= I I) (<= I I) (>= I I))))
      )
      (declare-var x Int)
      (declare-var y Int)
      (constraint (>= (max2 x y) x))
      (constraint (>= (max2 x y) y))
      (constraint (or (= x (max2 x y)) (= y (max2 x y))))
      (check-synth)
    "#;
    match parse_sygus(src) {
        Ok(cmds) => println!("{:#?}", cmds),
        Err(e) => eprintln!("Error: {}", e),
    }
}
