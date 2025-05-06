use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
    u32,
};

use kalosm::language::*;
use kalosm_llama::{EvaluationTrie, InferenceSettings, LlamaModel};
use kalosm_sample::{
    ArcParser, LazyParser, LiteralParser, Parser, ParserExt, SendCreateParserState,
};
use std::io::Write;
use tokio::sync::oneshot;

use nom::{
    IResult, Parser as _,
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
    alt((
        // parse quoted strings
        tag("\"")
            .and(is_not("\""))
            .and(tag("\""))
            .map(|((_, s), _)| SExpr::Atom(format!("\"{}\"", s))),
        // parse idents
        take_while1(|c: char| !c.is_whitespace() && !['(', ')'].contains(&c))
            .map(|s: &str| SExpr::Atom(s.to_string())),
    ))
    .parse(input)
}

#[test]
fn test_atom() {
    assert_eq!(atom("x"), Ok(("", SExpr::Atom("x".to_string()))));
    assert_eq!(atom("x y"), Ok((" y", SExpr::Atom("x".to_string()))));

    assert_eq!(atom("\"x y\""), Ok(("", SExpr::Atom("x y".to_string()))));
    assert_eq!(
        atom("\"x y\" z"),
        Ok((" z", SExpr::Atom("x y".to_string())))
    );
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
                    println!("args was none");
                    return None;
                };
                let ret_ty = match &rest[2] {
                    SExpr::Atom(t) => t.clone(),
                    _ => {
                        println!("ret was none");
                        return None;
                    }
                };
                let non_terms = Vec::new();
                // grammar rules
                let grammar = if let Some(SExpr::List(gr)) = &rest.get(3) {
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
                    Vec::new()
                };
                Some(Command::SynthFun(SynthFun {
                    name,
                    args,
                    ret_ty,
                    non_terminals: non_terms,
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

impl SExpr {
    pub fn as_int(&self) -> Option<i32> {
        if let SExpr::Atom(s) = self {
            s.parse::<i32>().ok()
        } else {
            None
        }
    }

    pub fn as_string(&self) -> Option<String> {
        if let SExpr::Atom(s) = self {
            Some(s.clone())
        } else {
            None
        }
    }
}

/// A synthesized function
#[derive(Debug, Clone, PartialEq)]
pub struct SynthFun {
    pub name: String,
    pub args: Vec<(String, String)>,
    pub ret_ty: String,
    pub non_terminals: Vec<(String, String)>,
    pub grammar: Vec<(String, String, Vec<SExpr>)>,
}

impl SynthFun {
    fn parser(&self) -> ArcParser<SExpr> {
        let map = TermMap::new(self.grammar.iter().map(|(n, _, _)| n.clone()));

        for (name, _, rules) in &self.grammar {
            let parser = parse_any_of_non_empty_list(rules.iter().map(|term| map.parser(term)));
            if map.functions_with_return_type[name].set(parser).is_err() {
                panic!("Multiple grammars for the same term")
            }
        }

        map.parser_for_term("Start").unwrap()
    }
}

struct TermMap {
    functions_with_return_type: HashMap<String, Arc<OnceLock<ArcParser<SExpr>>>>,
}

impl TermMap {
    fn new(terms: impl IntoIterator<Item = String>) -> Self {
        Self {
            functions_with_return_type: terms
                .into_iter()
                .map(|term| (term, Arc::new(OnceLock::new())))
                .collect(),
        }
    }

    fn parser(&self, expr: &SExpr) -> ArcParser<SExpr> {
        match expr {
            SExpr::Atom(atom) => self.parser_for_term(atom).unwrap_or_else(|| {
                let atom = atom.clone();
                LiteralParser::new(atom.clone())
                    .map_output(move |_| SExpr::Atom(atom.clone()))
                    .boxed()
            }),
            SExpr::List(sexprs) => LiteralParser::new("(")
                .ignore_output_then(
                    parse_non_empty_list_sequentially(sexprs.iter().map(|term| self.parser(term)))
                        .map_output(SExpr::List),
                )
                .then_literal(")")
                .boxed(),
        }
    }

    fn parser_for_term(&self, term: &str) -> Option<ArcParser<SExpr>> {
        self.functions_with_return_type
            .get(term)
            .map(|parser_list| {
                let parser_list = parser_list.clone();
                LazyParser::new(move || parser_list.get().unwrap().clone()).boxed()
            })
    }
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

fn parse_non_empty_list_sequentially<P: SendCreateParserState + 'static>(
    parsers: impl IntoIterator<Item = P>,
) -> ArcParser<Vec<P::Output>> {
    let mut iter = parsers.into_iter();
    let mut parser = iter.next().unwrap().map_output(|out| vec![out]).boxed();
    for next in iter {
        parser = parser
            .then_literal(" ")
            .then(next)
            .map_output(|(mut list, new)| {
                list.push(new);
                list
            })
            .boxed();
    }
    parser
}

fn parse_any_of_non_empty_list<P: SendCreateParserState + 'static>(
    parsers: impl IntoIterator<Item = P>,
) -> ArcParser<P::Output> {
    let mut iter = parsers.into_iter();
    let mut parser = iter.next().unwrap().boxed();
    for next in iter {
        parser = parser.or(next).boxed();
    }
    parser
}

#[tokio::main]
async fn main() {
    let grammar_text = include_str!("./grammar.sl");
    let parsed = parse_sygus(grammar_text).unwrap();
    println!("parsed: {:#?}", parsed);
    let synth_fun = parsed
        .into_iter()
        .find_map(|command| {
            if let Command::SynthFun(fun) = command {
                Some(fun)
            } else {
                None
            }
        })
        .unwrap();

    let parser = synth_fun.parser();
    let parser = MaxLengthParser::new(
        LiteralParser::new("(define-fun f ((firstname String) (lastname String)) String ")
            .then(parser.clone()),
        200,
    );

    tracing_subscriber::fmt().init();

    let sampler = GenerationParameters::new();

    let mut llm = LlamaModel::from_builder(
        Llama::builder().with_source(LlamaSource::deepseek_r1_distill_qwen_7b()),
        ModelLoadingProgress::multi_bar_loading_indicator(),
    )
    .await
    .unwrap();

    tokio::task::spawn_blocking(move || {
        let mut trie = EvaluationTrie::new();
        let mut last_entropy = 0.0;
        let prompt = include_str!("prompt");
        let task = grammar_text;
        let session = llm.new_session();
        // let prompt = format!("<|im_start|>system\n{prompt}<|im_end|>\n<|im_start|>user\nQuestion:\n{task}<|im_end|>\n<|im_start|>assistant\n");
        let prompt =
            format!("<｜begin▁of▁sentence｜>{prompt}<｜User｜>Solve this problem:\n{task}<｜Assistant｜><think>\n");

        let (finished, _keep) = oneshot::channel();
        llm._infer(
            InferenceSettings::new(
                &prompt,
                session.clone(),
                Arc::new(Mutex::new(sampler.clone())),
                u32::MAX,
                Some("</think>".to_string()),
                None,
            ),
            Box::new(|token| {
                print!("{}", token);
                std::io::stdout().flush().unwrap();
                Ok(())
            }),
            &finished
        )
        .unwrap();
        for generation in 0.. {
            let mut session = session.deep_clone();
            let output = match llm.generate_structured_with_trie(
                &mut session,
                &prompt,
                sampler.clone(),
                parser.clone(),
                |token| {
                    print!("{}", token);
                    std::io::stdout().flush().unwrap();
                    Ok(())
                },
                &mut trie,
            ) {
                Ok(output) => output,
                Err(e) => {
                    println!("Error: {e}");
                    continue;
                }
            };
            println!("\n\n");

            println!("generation {generation}:\n{output:?}");
            let shannon_entropy = trie.shannon_entropy();
            let entropy_diff = last_entropy - shannon_entropy;
            println!("entropy diff: {entropy_diff}");
            // if entropy_diff.abs() < 0.00001 {
            //     println!("looks like entropy is converging, stopping generation");
            //     break;
            // }
            println!("shannon entropy: {shannon_entropy}");
            last_entropy = shannon_entropy;
        }
    })
    .await
    .unwrap();
}

#[derive(Debug, Clone, PartialEq)]
struct MaxLengthParser<P: SendCreateParserState> {
    parser: P,
    max_length: usize,
}

impl<P: SendCreateParserState> MaxLengthParser<P> {
    fn new(parser: P, max_length: usize) -> Self {
        Self { parser, max_length }
    }
}

impl<P: SendCreateParserState> Parser for MaxLengthParser<P> {
    type Output = P::Output;
    type PartialState = (P::PartialState, usize);

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        let (partial_state, length) = state;
        let new_length = length + input.len();
        if new_length >= self.max_length {
            return Err(ParserError::msg("Max length exceeded"));
        }
        let new_state = self.parser.parse(partial_state, input)?;
        let new_state = new_state.map_state(|s| (s, new_length));
        Ok(new_state)
    }
}

impl<P: SendCreateParserState> CreateParserState for MaxLengthParser<P> {
    fn create_parser_state(&self) -> Self::PartialState {
        (self.parser.create_parser_state(), 0)
    }
}

fn built_in_functions() -> HashMap<String, Box<dyn Fn(&[SExpr]) -> SExpr>> {
    let mut functions = HashMap::new();
    fn binary_op(op: fn(i32, i32) -> i32) -> impl Fn(&[SExpr]) -> SExpr {
        move |args: &[SExpr]| {
            let first = args[0].as_int().unwrap();
            let second = args[1].as_int().unwrap();
            SExpr::Atom((op(first, second)).to_string())
        }
    }

    fn insert(
        functions: &mut HashMap<String, Box<dyn Fn(&[SExpr]) -> SExpr>>,
        name: impl ToString,
        op: impl Fn(&[SExpr]) -> SExpr + 'static,
    ) {
        functions.insert(name.to_string(), Box::new(op));
    }
    
    insert(&mut functions, "+", binary_op(|a, b| a + b));
    insert(&mut functions, "-", binary_op(|a, b| a - b));
    insert(&mut functions, "*", binary_op(|a, b| a * b));
    insert(&mut functions, "/", binary_op(|a, b| a / b));
    insert(&mut functions, "=", binary_op(|a, b| (a == b) as i32));

    insert(&mut functions, "str.++", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let second = args[1].as_string().unwrap();
        let merged = first + &second;
        SExpr::Atom(merged)
    });
    insert(&mut functions, "str.len", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        SExpr::Atom(first.len().to_string())
    });
    insert(&mut functions, "str.substr", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let start = args[1].as_int().unwrap();
        let end = args[2].as_int().unwrap();
        SExpr::Atom(first[start as usize..end as usize].to_string())
    });
    insert(&mut functions, "str.at", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let index = args[1].as_int().unwrap();
        SExpr::Atom(first[index as usize..index as usize + 1].to_string())
    });
    insert(&mut functions, "str.to.int", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        SExpr::Atom(first.parse::<i32>().unwrap().to_string())
    });
    insert(&mut functions, "str.indexof", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let second = args[1].as_string().unwrap();
        let offset = args[2].as_int().unwrap();
        let index = first[offset as usize..]
            .find(&second)
            .map(|i| offset + i as i32)
            .unwrap_or(-1);
        SExpr::Atom(index.to_string())
    });
    insert(&mut functions, "str.replace", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let second = args[1].as_string().unwrap();
        let third = args[2].as_string().unwrap();
        SExpr::Atom(first.replace(&second, &third))
    });
    insert(&mut functions, "str.prefixof", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let second = args[1].as_string().unwrap();
        SExpr::Atom(first.starts_with(&second).to_string())
    });
    insert(&mut functions, "str.suffixof", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let second = args[1].as_string().unwrap();
        SExpr::Atom(first.ends_with(&second).to_string())
    });
    insert(&mut functions, "str.contains", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let second = args[1].as_string().unwrap();
        SExpr::Atom(first.contains(&second).to_string())
    });

    insert(&mut functions, "int.to.str", |args: &[SExpr]| {
        let first = args[0].as_int().unwrap();
        SExpr::Atom(first.to_string())
    });

    todo!()
}
