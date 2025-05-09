use std::{
    collections::HashMap,
    fmt::{Debug, Display},
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

use clap::Parser as _;
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
        // parse booleans
        alt((
            tag("true").map(|_| SExpr::Atom(Atom::Bool(true))),
            tag("false").map(|_| SExpr::Atom(Atom::Bool(false))),
        )),
        // parse maybe negative integers
        nom::character::complete::i32.map(|i| SExpr::Atom(Atom::Int(i))),
        // parse quoted strings
        delimited(tag("\""), is_not("\""), tag("\""))
            .map(|s: &str| SExpr::Atom(Atom::String(s.to_string()))),
        // parse idents
        take_while1(|c: char| !c.is_whitespace() && !['(', ')'].contains(&c))
            .map(|s: &str| SExpr::Atom(Atom::Ident(s.to_string()))),
    ))
    .parse(input)
}

#[test]
fn test_atom() {
    assert_eq!(
        atom("x"),
        Ok(("", SExpr::Atom(Atom::Ident("x".to_string()))))
    );
    assert_eq!(
        atom("x y"),
        Ok((" y", SExpr::Atom(Atom::Ident("x".to_string()))))
    );

    assert_eq!(
        atom("\"x y\""),
        Ok(("", SExpr::Atom(Atom::String("x y".to_string()))))
    );
    assert_eq!(
        atom("\"x y\" z"),
        Ok((" z", SExpr::Atom(Atom::String("x y".to_string()))))
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
                SExpr::Atom(Atom::Ident("x".to_string())),
                SExpr::Atom(Atom::Ident("y".to_string()))
            ])
        ))
    );
    assert_eq!(
        list("(x (y z))"),
        Ok((
            "",
            SExpr::List(vec![
                SExpr::Atom(Atom::Ident("x".to_string())),
                SExpr::List(vec![
                    SExpr::Atom(Atom::Ident("y".to_string())),
                    SExpr::Atom(Atom::Ident("z".to_string()))
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
                SExpr::Atom(Atom::Ident("x".to_string())),
                SExpr::Atom(Atom::Ident("y".to_string()))
            ])
        ))
    );
    assert_eq!(
        sexpr("x"),
        Ok(("", SExpr::Atom(Atom::Ident("x".to_string()))))
    );
}

/// Transform an SExpr into a Command
fn to_command(expr: SExpr) -> Option<Command> {
    match expr {
        SExpr::List(items) => match items.split_first()? {
            (SExpr::Atom(Atom::Ident(cmd)), rest) if cmd == "set-logic" => {
                if let [SExpr::Atom(Atom::Ident(logic))] = rest {
                    Some(Command::SetLogic(logic.clone()))
                } else {
                    None
                }
            }
            (SExpr::Atom(Atom::Ident(cmd)), rest) if cmd == "synth-fun" => {
                // (synth-fun name ((x T) ...) Ret
                //   ((Nt Ty) ...)          ; non-terminals
                //   ((Nt1 ...) (Nt2 ...) …) ; grammar
                let name = match &rest[0] {
                    SExpr::Atom(Atom::Ident(n)) => n.clone(),
                    _ => return None,
                };
                let args = if let SExpr::List(a) = &rest[1] {
                    a.iter()
                        .filter_map(|p| {
                            if let SExpr::List(v) = p {
                                if let [SExpr::Atom(Atom::Ident(n)), SExpr::Atom(Atom::Ident(t))] =
                                    v.as_slice()
                                {
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
                    SExpr::Atom(Atom::Ident(t)) => t.clone(),
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
                                if let [
                                    SExpr::Atom(Atom::Ident(n)),
                                    SExpr::Atom(Atom::Ident(ty)),
                                    SExpr::List(rhs),
                                ] = &v[..]
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
            (
                SExpr::Atom(Atom::Ident(cmd)),
                [SExpr::Atom(Atom::Ident(var)), SExpr::Atom(Atom::Ident(ty))],
            ) if cmd == "declare-var" => Some(Command::DeclareVar(var.clone(), ty.clone())),
            (SExpr::Atom(Atom::Ident(cmd)), [c]) if cmd == "constraint" => {
                Some(Command::Constraint(c.clone()))
            }
            (SExpr::Atom(Atom::Ident(cmd)), []) if cmd == "check-synth" => {
                Some(Command::CheckSynth)
            }
            _ => None,
        },
        _ => None,
    }
}

/// Minimal S-expression AST
#[derive(Debug, Clone, PartialEq)]
pub enum SExpr {
    Atom(Atom),
    List(Vec<SExpr>),
}

impl SExpr {
    pub fn as_bool(&self) -> Option<bool> {
        if let SExpr::Atom(Atom::Bool(b)) = self {
            Some(*b)
        } else {
            None
        }
    }

    pub fn as_int(&self) -> Option<i32> {
        if let SExpr::Atom(Atom::Int(i)) = self {
            Some(*i)
        } else {
            None
        }
    }

    pub fn as_string(&self) -> Option<String> {
        if let SExpr::Atom(Atom::String(s)) = self {
            Some(s.clone())
        } else {
            None
        }
    }

    pub fn as_atom(&self) -> Option<&Atom> {
        if let SExpr::Atom(a) = self {
            Some(a)
        } else {
            None
        }
    }

    pub fn from_string(s: impl Display) -> SExpr {
        SExpr::Atom(Atom::String(s.to_string()))
    }
}

impl From<i32> for SExpr {
    fn from(value: i32) -> Self {
        SExpr::Atom(Atom::Int(value))
    }
}

impl From<bool> for SExpr {
    fn from(value: bool) -> Self {
        SExpr::Atom(Atom::Bool(value))
    }
}

impl From<String> for SExpr {
    fn from(value: String) -> Self {
        SExpr::Atom(Atom::String(value))
    }
}

impl From<&str> for SExpr {
    fn from(value: &str) -> Self {
        SExpr::Atom(Atom::String(value.to_string()))
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Atom {
    String(String),
    Int(i32),
    Bool(bool),
    Ident(String),
}

impl Display for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Atom::String(s) => write!(f, "\"{}\"", s),
            Atom::Int(i) => write!(f, "{}", i),
            Atom::Bool(b) => write!(f, "{}", b),
            Atom::Ident(s) => write!(f, "{}", s),
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
            SExpr::Atom(atom) => self.parser_for_term(&atom.to_string()).unwrap_or_else(|| {
                let atom = atom.clone();
                LiteralParser::new(atom.to_string())
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

/// Simple program to greet a person
#[derive(clap::Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long)]
    think: bool,

    #[arg(long)]
    grammar: String,

    #[arg(long)]
    task: String,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let think = args.think;

    let grammar_text = std::fs::read_to_string(args.grammar).unwrap();
    let prompt = std::fs::read_to_string(args.task).unwrap();
    let parsed = parse_sygus(&grammar_text).unwrap();
    println!("parsed: {:#?}", parsed);
    let synth_fun = parsed
        .iter()
        .find_map(|command| {
            if let Command::SynthFun(fun) = command {
                Some(fun.clone())
            } else {
                None
            }
        })
        .unwrap();

    let vars = synth_fun
        .args
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let constraints = parsed
        .iter()
        .filter_map(|command| {
            if let Command::Constraint(expr) = command {
                Some(expr.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let parser = synth_fun.parser();
    let parser = MaxLengthParser::new(
        LiteralParser::new("(define-fun f ((firstname String) (lastname String)) String ")
            .ignore_output_then(parser.clone()),
        200,
    );

    tracing_subscriber::fmt().init();

    let sampler = GenerationParameters::new()
        .with_repetition_penalty_range(256)
        .with_top_k(1);

    let source = if think {
        LlamaSource::deepseek_r1_distill_qwen_7b()
    } else {
        LlamaSource::qwen_2_5_7b_instruct()
    };
    let mut llm = LlamaModel::from_builder(
        Llama::builder().with_source(source),
        ModelLoadingProgress::multi_bar_loading_indicator(),
    )
    .await
    .unwrap();

    tokio::task::spawn_blocking(move || {
        let interpreter = Interpreter::new();
        
        let mut trie = EvaluationTrie::new();
        let mut last_entropy = 0.0;
        let task = grammar_text;
        let mut session = llm.new_session();
        let prompt = if think {
            format!("<｜begin▁of▁sentence｜>{prompt}<｜User｜>Solve this problem:\n{task}<｜Assistant｜><think>\n")
        } else {
            format!("<|im_start|>system\n{prompt}<|im_end|>\n<|im_start|>user\nQuestion:\n{task}<|im_end|>\n<|im_start|>assistant\n")
        };

        if think {
            while llm.generate_structured_with_trie(
                &mut session,
                &prompt,
                sampler.clone(),
                MaxLengthParser::new(StopOn::new("</think>"), 8192),
                Box::new(|token| {
                    print!("{}", token);
                    std::io::stdout().flush().unwrap();
                    Ok(())
                }),
                &mut EvaluationTrie::new(),
            )
            .is_err() {
                println!("Initial prompt too long, retrying");
                session = llm.new_session();
            }
        }

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

            println!("Checking constraints:");
            for constraint in &constraints {
                let result = interpreter.check(constraint, vars.clone(), &output);
                println!("  {constraint:?} => {result}");
            }

            let shannon_entropy = trie.shannon_entropy();
            let entropy_diff = last_entropy - shannon_entropy;
            println!("entropy diff: {entropy_diff}");
            if entropy_diff.abs() < 0.00001 {
                println!("looks like entropy is converging, stopping generation");
                break;
            }
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

#[derive(Clone)]
struct Interpreter {
    functions: HashMap<String, Arc<dyn Fn(&[SExpr], &mut Self) -> SExpr>>,
    bindings: Vec<HashMap<String, Atom>>,
}

impl Interpreter {
    fn new() -> Self {
        Self {
            functions: built_in_functions(),
            bindings: Vec::new(),
        }
    }

    fn check(&self, constraints: &SExpr, variables: Vec<String>, fn_body: &SExpr) -> bool {
        let mut fn_map = self.clone();
        let fn_body = fn_body.clone();
        fn_map.functions.insert(
            "f".to_string(),
            Arc::new(move |args, fn_map| {
                fn_map.bindings.push(
                    variables
                        .iter()
                        .zip(args.iter().cloned())
                        .map(|(name, value)| (name.clone(), value.as_atom().unwrap().clone()))
                        .collect(),
                );
                let value = fn_map.eval(&fn_body.clone());
                fn_map.bindings.pop();
                value
            }),
        );
        fn_map.eval(constraints).as_bool().unwrap()
    }

    fn eval(&mut self, expr: &SExpr) -> SExpr {
        match expr {
            SExpr::Atom(Atom::Ident(name)) => {
                if let Some(binding) = self.bindings.last() {
                    if let Some(value) = binding.get(name) {
                        SExpr::Atom(value.clone())
                    } else {
                        panic!("Unknown variable: {}", name)
                    }
                } else {
                    panic!("No bindings available")
                }
            }
            SExpr::Atom(_) => expr.clone(),
            SExpr::List(items) => {
                let (first, rest) = items.split_first().unwrap();
                if let SExpr::Atom(Atom::Ident(name)) = first {
                    if let Some(func) = self.functions.get(name).cloned() {
                        let rest = rest.iter().map(|item| self.eval(item)).collect::<Vec<_>>();
                        func(&rest, self)
                    } else {
                        panic!("Unknown function: {}", name)
                    }
                } else {
                    panic!("Expected function name, got: {:?}", first)
                }
            }
        }
    }
}

fn built_in_functions() -> HashMap<String, Arc<dyn Fn(&[SExpr], &mut Interpreter) -> SExpr>> {
    let mut functions = HashMap::new();
    fn binary_op(op: fn(i32, i32) -> i32) -> impl Fn(&[SExpr]) -> i32 {
        move |args: &[SExpr]| {
            let first = args[0].as_int().unwrap();
            let second = args[1].as_int().unwrap();
            op(first, second)
        }
    }

    fn insert<O: Into<SExpr>>(
        functions: &mut HashMap<String, Arc<dyn Fn(&[SExpr], &mut Interpreter) -> SExpr>>,
        name: impl ToString,
        op: impl Fn(&[SExpr]) -> O + 'static,
    ) {
        functions.insert(name.to_string(), Arc::new(move |i, _| op(i).into()));
    }

    insert(&mut functions, "+", binary_op(|a, b| a + b));
    insert(&mut functions, "-", binary_op(|a, b| a - b));
    insert(&mut functions, "*", binary_op(|a, b| a * b));
    insert(&mut functions, "/", binary_op(|a, b| a / b));
    insert(&mut functions, "=", |args| {
        let [first, second] = args else {
            unreachable!()
        };
        first == second
    });

    insert(&mut functions, "str.++", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let second = args[1].as_string().unwrap();
        let merged = first + &second;
        merged
    });
    insert(&mut functions, "str.len", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        SExpr::Atom(Atom::Int(first.len() as _))
    });
    insert(&mut functions, "str.substr", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let start = args[1].as_int().unwrap();
        let end = args[2].as_int().unwrap();
        if end < 0 || start < 0 || start > first.len() as _ {
            "".to_string()
        } else {
            first[start as usize..end as usize].to_string()
        }
    });
    insert(&mut functions, "str.at", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let index = args[1].as_int().unwrap();
        SExpr::Atom(Atom::String(
            first[index as usize..index as usize + 1].to_string(),
        ))
    });
    insert(&mut functions, "str.to.int", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        SExpr::Atom(Atom::Int(first.parse::<i32>().unwrap_or(-1)))
    });
    insert(&mut functions, "str.indexof", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let second = args[1].as_string().unwrap();
        let offset = args[2].as_int().unwrap();
        let index = first[offset as usize..]
            .find(&second)
            .map(|i| offset + i as i32)
            .unwrap_or(-1);
        SExpr::Atom(Atom::String(index.to_string()))
    });
    insert(&mut functions, "str.replace", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let second = args[1].as_string().unwrap();
        let third = args[2].as_string().unwrap();
        SExpr::from_string(first.replace(&second, &third))
    });
    insert(&mut functions, "str.prefixof", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let second = args[1].as_string().unwrap();
        SExpr::Atom(Atom::String(first.starts_with(&second).to_string()))
    });
    insert(&mut functions, "str.suffixof", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let second = args[1].as_string().unwrap();
        SExpr::Atom(Atom::String(first.ends_with(&second).to_string()))
    });
    insert(&mut functions, "str.contains", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let second = args[1].as_string().unwrap();
        SExpr::Atom(Atom::Bool(first.contains(&second)))
    });

    insert(&mut functions, "int.to.str", |args: &[SExpr]| {
        let first = args[0].as_int().unwrap();
        if first >= 0 {
            SExpr::from(first)
        } else {
            SExpr::from_string("")
        }
    });

    functions
}

#[test]
fn test_interpreter() {
    let mut interpreter = Interpreter::new();
    let expr = "(+ 1 2)";
    let expr = sexpr(expr).unwrap().1;
    let result = interpreter.eval(&expr);
    assert_eq!(result, SExpr::Atom(Atom::Int(3)));
}

#[test]
fn test_interpreter_str() {
    let mut interpreter = Interpreter::new();
    let expr = "(str.++ \"Hello, \" \"world!\")";
    let expr = sexpr(expr).unwrap().1;
    let result = interpreter.eval(&expr);
    assert_eq!(
        result,
        SExpr::Atom(Atom::String("Hello, world!".to_string()))
    );
}

#[test]
fn test_interpreter_str_len() {
    let mut interpreter = Interpreter::new();
    let expr = "(str.len \"Hello, world!\")";
    let expr = sexpr(expr).unwrap().1;
    let result = interpreter.eval(&expr);
    assert_eq!(result, SExpr::Atom(Atom::Int(13)));
}

#[test]
fn test_nested_expressions() {
    let mut interpreter = Interpreter::new();
    let expr = "(str.++ (str.substr \"Hello, world!\" 0 5) \"!\")";
    let expr = sexpr(expr).unwrap().1;
    let result = interpreter.eval(&expr);
    assert_eq!(result, SExpr::Atom(Atom::String("Hello!".to_string())));
}

#[test]
fn test_check_solution() {
    let interpreter = Interpreter::new();
    let constraints = r#"(= (f "Launa" "Withers") "Withers, L.")"#;
    let constraints = sexpr(constraints).unwrap().1;
    let arguments = ["firstname".to_string(), "lastname".to_string()];
    let body = r#"(str.++ (str.++ (str.++ lastname ", ") (str.substr firstname 0 1)) ".")"#;
    let body = sexpr(body).unwrap().1;
    assert!(interpreter.check(&constraints, arguments.to_vec(), &body));
}
