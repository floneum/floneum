// usage:
// cargo run --features metal -- --model qwen0.5b --grammar src/firstname.sl --task src/prompt

use cfg::{
    parse::{Grammar, Rule, Symbol},
    slab_grammar::SlabGrammar,
    tokenizer::Tokenizer,
};
use kalosm::language::*;
use kalosm_llama::{Cache, EvaluationTrie, LlamaModel, StructuredInferenceTimingInfo};
use kalosm_sample::{
    ArcParser, LazyParser, LiteralParser, Parser, ParserExt, SendCreateParserState,
};
use llm_samplers::{
    configure::{SamplerChainBuilder, SamplerSlot},
    prelude::{SampleFreqPresence, SampleGreedy, SampleRepetition, SampleSeqRepetition},
    types::Sampler,
};
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    path::Path,
    sync::{Arc, Mutex, RwLock},
};
use std::{io::Write, num::NonZeroUsize};

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
#[derive(Debug, Clone, PartialEq, Hash, Eq)]
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

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum Atom {
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
    fn parser(&self, recursion_depth: usize) -> ArcParser<SExpr> {
        let map = TermMap::new(
            self.grammar
                .iter()
                .map(|(n, _, rule)| (n.clone(), rule.clone())),
        );

        map.parser_for_term("Start", recursion_depth).unwrap()
    }

    fn grammar(&self) -> Grammar<String> {
        let mut grammar = Grammar {
            start: "Start".to_string(),
            rules: Vec::new(),
        };

        fn atom_to_symbol(atom: &Atom, grammar: &Vec<(String, String, Vec<SExpr>)>) -> Symbol {
            match atom {
                Atom::Ident(name) => {
                    if grammar.iter().any(|(lhs, _, _)| lhs == name) {
                        Symbol::NonTerminal(name.clone())
                    } else {
                        Symbol::Terminal(name.clone())
                    }
                }
                Atom::String(s) => Symbol::Terminal(format!("\"{}\"", s)),
                Atom::Int(i) => Symbol::Terminal(i.to_string()),
                Atom::Bool(b) => Symbol::Terminal(b.to_string()),
            }
        }

        fn s_expr_to_symbols(
            sexpr: &SExpr,
            grammar: &Vec<(String, String, Vec<SExpr>)>,
        ) -> Vec<Symbol> {
            match sexpr {
                SExpr::Atom(atom) => {
                    vec![atom_to_symbol(atom, grammar)]
                }
                SExpr::List(list) => {
                    let mut symbols = Vec::new();
                    symbols.push(Symbol::Terminal("(".to_string()));
                    for (i, item) in list.iter().enumerate() {
                        if i > 0 {
                            symbols.push(Symbol::Terminal(" ".to_string()));
                        }
                        symbols.extend(s_expr_to_symbols(item, grammar));
                    }
                    symbols.push(Symbol::Terminal(")".to_string()));
                    symbols
                }
            }
        }

        for (lhs, _, rhs) in &self.grammar {
            let rhs = rhs
                .iter()
                .map(|r| s_expr_to_symbols(r, &self.grammar).into())
                .collect();
            grammar.rules.push(Rule {
                lhs: lhs.clone(),
                rhs,
            });
        }
        grammar
    }
}

#[derive(Debug, Clone)]
struct TermMap {
    functions_with_return_type: Arc<HashMap<String, Vec<SExpr>>>,
}

impl TermMap {
    fn new(terms: impl IntoIterator<Item = (String, Vec<SExpr>)>) -> Self {
        Self {
            functions_with_return_type: Arc::new(terms.into_iter().collect()),
        }
    }

    fn parser(&self, expr: &SExpr, recursion_depth_left: usize) -> ArcParser<SExpr> {
        if recursion_depth_left == 0 {
            return FailParser(std::marker::PhantomData).boxed();
        }
        match expr {
            SExpr::Atom(atom) => self
                .parser_for_term(&atom.to_string(), recursion_depth_left)
                .unwrap_or_else(|| {
                    let atom = atom.clone();
                    LiteralParser::new(atom.to_string())
                        .map_output(move |_| SExpr::Atom(atom.clone()))
                        .boxed()
                }),
            SExpr::List(sexprs) => LiteralParser::new("(")
                .ignore_output_then(
                    parse_non_empty_list_sequentially(sexprs.iter().map(|term| {
                        let term = term.clone();
                        let this = self.clone();
                        LazyParser::new(move || this.parser(&term, recursion_depth_left - 1))
                    }))
                    .map_output(SExpr::List),
                )
                .then_literal(")")
                .boxed(),
        }
    }

    fn parser_for_term(&self, term: &str, recursion_depth_left: usize) -> Option<ArcParser<SExpr>> {
        self.functions_with_return_type
            .get(term)
            .map(|parser_list| {
                parse_any_of_non_empty_list(
                    parser_list
                        .iter()
                        .map(|parser| self.parser(parser, recursion_depth_left)),
                )
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
    #[arg(long, value_enum)]
    model: Model,

    #[arg(long)]
    grammar: String,

    #[arg(long)]
    task: String,

    #[arg(long)]
    vis: bool,

    #[arg(long, default_value_t = 10)]
    recursion_depth: usize,

    #[arg(long, default_value_t = 25)]
    iterations: usize,

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    multipass: bool,

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    fast_case: bool,
}

/// Doc comment
#[derive(clap::ValueEnum, Clone, Debug)]
enum Model {
    #[value(name = "qwen0.5b")]
    Qwen0_5b,

    #[value(name = "qwen1.5b")]
    Qwen1_5b,

    #[value(name = "qwen3b")]
    Qwen3b,

    #[value(name = "qwen7b")]
    Qwen7b,

    #[value(name = "qwen1.5b-think")]
    Qwen1_5bThink,

    #[value(name = "qwen7b-think")]
    Qwen7bThink,

    #[value(name = "smol-lm")]
    SmolLM,

    #[value(name = "llama1b")]
    Llama1b,

    #[value(name = "llama3b")]
    Llama3b,

    #[value(name = "llama8b")]
    Llama8b,
}

impl Model {
    fn qwen_normal(&self) -> bool {
        match self {
            Model::Qwen0_5b | Model::Qwen1_5b | Model::Qwen3b | Model::Qwen7b => true,
            _ => false,
        }
    }

    fn qwen_think(&self) -> bool {
        match self {
            Model::Qwen1_5bThink | Model::Qwen7bThink => true,
            _ => false,
        }
    }

    fn llama(&self) -> bool {
        match self {
            Model::Llama1b | Model::Llama3b | Model::Llama8b => true,
            _ => false,
        }
    }

    fn smol_lm(&self) -> bool {
        match self {
            Model::SmolLM => true,
            _ => false,
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();
    run().await;
}

async fn run() {
    let args = Args::parse();
    let model = args.model;
    let recursion_depth = args.recursion_depth;
    let iterations = args.iterations;
    let multipass = args.multipass;

    let grammar_text = std::fs::read_to_string(args.grammar).unwrap();
    let prompt = std::fs::read_to_string(args.task).unwrap();
    let parsed = parse_sygus(&grammar_text).unwrap();
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

    let args_str = synth_fun
        .args
        .iter()
        .map(|(name, ty)| format!("({name} {ty})"))
        .collect::<Vec<_>>()
        .join(" ");

    let parser = synth_fun.parser(recursion_depth);
    let parser = parser
        .clone()
        .then_lazy({
            let interpreter = Interpreter::new();
            let constraints = constraints.clone();
            let vars = vars.clone();
            let cache: RwLock<LruCache<SExpr, bool>> =
                RwLock::new(LruCache::new(NonZeroUsize::new(1024).unwrap()));
            move |result| {
                let mut cache = cache.write().unwrap();
                let valid = if let Some(cached) = cache.get(&result) {
                    *cached
                } else {
                    let mut valid = true;
                    for constraint in &constraints {
                        let result = interpreter.check(constraint, vars.clone(), &result);
                        valid &= result;
                    }
                    cache.put(result.clone(), valid);
                    valid
                };

                if valid {
                    SuccessParser(()).boxed()
                } else {
                    FailParser(std::marker::PhantomData).boxed()
                }
            }
        })
        .map_output(|(a, _)| a)
        .boxed();

    let prefix = format!("(define-fun f ({args_str}) String ",);
    let parser = LiteralParser::new(prefix.clone()).ignore_output_then(parser);

    let sampler = SamplerChainBuilder::from([
        (
            "repetition",
            SamplerSlot::new_static(move || Box::new(SampleRepetition::default())),
        ),
        (
            "freqpresence",
            SamplerSlot::new_static(move || Box::new(SampleFreqPresence::default().last_n(64))),
        ),
        (
            "seqrepetition",
            SamplerSlot::new_static(move || Box::<SampleSeqRepetition>::default()),
        ),
        (
            "greedy",
            SamplerSlot::new_static(move || Box::new(SampleGreedy::default())),
        ),
    ])
    .into_chain();
    let sampler = Arc::new(Mutex::new(sampler)) as Arc<Mutex<dyn Sampler>>;

    let source = match model {
        Model::SmolLM => LlamaSource::new(
            // Llama source takes a gguf file to load the model, tokenizer, and chat template from
            FileSource::HuggingFace {
                model_id: "QuantFactory/SmolLM2-135M-Instruct-GGUF".to_string(),
                revision: "main".to_string(),
                file: "SmolLM2-135M-Instruct.Q4_K_M.gguf".to_string(),
            },
        )
        .with_tokenizer(FileSource::HuggingFace {
            model_id: "HuggingFaceTB/SmolLM2-135M-Instruct".to_string(),
            revision: "main".to_string(),
            file: "tokenizer.json".to_string(),
        }),
        Model::Qwen0_5b => LlamaSource::qwen_2_5_0_5b_instruct(),
        Model::Qwen1_5b => LlamaSource::qwen_2_5_1_5b_instruct(),
        Model::Qwen3b => LlamaSource::qwen_2_5_3b_instruct(),
        Model::Qwen7b => LlamaSource::qwen_2_5_7b_instruct(),
        Model::Qwen1_5bThink => LlamaSource::deepseek_r1_distill_qwen_1_5b(),
        Model::Qwen7bThink => LlamaSource::deepseek_r1_distill_qwen_7b(),
        Model::Llama1b => LlamaSource::llama_3_2_1b_chat(),
        Model::Llama3b => LlamaSource::llama_3_2_3b_chat(),
        Model::Llama8b => LlamaSource::llama_3_1_8b_chat(),
    };

    let tokenizer = source.tokenizer.clone().unwrap();
    let cache = Cache::default();
    let tokenizer_path = cache.get(&tokenizer, |_| {}).await.unwrap();
    let grammar = create_grammar(synth_fun, &prefix, &tokenizer_path, args.fast_case);
    let mut llm = LlamaModel::from_builder(
        Llama::builder().with_source(source),
        ModelLoadingProgress::multi_bar_loading_indicator(),
    )
    .await
    .unwrap();

    tokio::task::spawn_blocking(move || {
        let overall_start_time = std::time::Instant::now();
        let mut trie = EvaluationTrie::new();
        let mut last_entropy = 1000.0;
        let task = grammar_text;
        let mut session = llm.new_session();
        let bump = bumpalo::Bump::new();
        let grammar = grammar.reallocate(&bump);
        let prompt = if model.qwen_normal() || model.smol_lm() {
            format!("<|im_start|>system\n{prompt}<|im_end|>\n<|im_start|>user\nQuestion:\n{task}<|im_end|>\n")
        } else if model.llama() {
            format!("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{task}<|eot_id|>")
        } else {
            format!("<｜begin▁of▁sentence｜>{prompt}<｜User｜>Solve this problem:\n{task}<｜Assistant｜><think>\n")
        };
        let end = if model.qwen_normal() || model.smol_lm() {
            format!("<|im_start|>assistant\n")
        } else if model.llama() {
            format!("<|start_header_id|>assistant<|end_header_id|>")
        } else {
            format!("\n")
        };

        let (tx, _rx) = tokio::sync::oneshot::channel();
        {
            if model.qwen_think() {
                while llm._infer_inner(
                    prompt.clone(),
                    Some("</think>".to_string()),
                    sampler.clone(),
                    &mut session,
                    2048,
                    None,
                    Box::new(|_| {
                        Ok(())
                    }),
                    &tx
                )
                .is_err() {
                    session = llm.new_session();
                }
            } else {
                llm._infer_inner(
                    prompt.clone(),
                    None,
                    sampler.clone(),
                    &mut session,
                    0,
                    None,
                    Box::new(|token| {
                        std::io::stdout().flush().unwrap();
                        Ok(())
                    }),
                    &tx
                ).unwrap();
            }
        }

        for generation in 0..iterations {
            let shannon_entropy = trie.shannon_entropy();
            let entropy_diff = last_entropy - shannon_entropy;
            last_entropy = shannon_entropy;
            // If we aren't doing multipass generation, reset the trie
            if !multipass {
                trie.clear();
            }
            if args.vis {
                println!("{}", trie.graphvis(&llm.tokenizer));
            }
            let mut session = session.deep_clone();
            let all_tokens = Arc::new(RwLock::new(String::new()));
            let all_token_ids = Arc::new(RwLock::new(Vec::new()));
            let new_text =end.clone();
            let (result, timing) =  llm.generate_structured_with_trie(
                &mut session,
                &new_text,
                sampler.clone(),
                Some(&grammar),
                &parser,
                {
                    let all_tokens = all_tokens.clone();
                    let all_token_ids = all_token_ids.clone();
                    move |token, id| {
                        all_tokens.write().unwrap().push_str(&token.to_string());
                        all_token_ids.write().unwrap().push(id);
                        std::io::stdout().flush().unwrap();
                        Ok(())
                    }
                },
                &mut trie,
            );
            let all_tokens = all_tokens.read().unwrap().clone();

            let all_token_ids = all_token_ids.read().unwrap();
            let retokenized = llm.tokenizer.encode_fast(all_tokens.as_str(), false).unwrap();
            let tokenization_error = retokenized.get_ids() != all_token_ids.as_slice();

            let valid = if let Ok(output) = result {
            let interpreter = Interpreter::new();
            let mut valid = true;
            for constraint in &constraints {
                let result = interpreter.check(constraint, vars.clone(), &output);
                valid = valid && result;
            }
            valid
        } else {false};
        let run = Run { generation: generation as u32,
            metadata: timing,
            pass: valid,
            entropy: shannon_entropy,
            entropy_diff: entropy_diff,
            tokenization_error,
            result: all_tokens,
            tokens: all_token_ids.clone()
        };

        println!("{}", serde_json::to_string(&run).unwrap());

            if valid {
                break;
            }
        }
        let total_duration = overall_start_time.elapsed();
        eprintln!("Total generation took: {total_duration:?}");
    })
    .await
    .unwrap();
}

#[derive(Clone)]
struct Interpreter {
    functions: HashMap<String, Arc<dyn Fn(&[SExpr], &mut Self) -> SExpr + Send + Sync>>,
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

fn built_in_functions()
-> HashMap<String, Arc<dyn Fn(&[SExpr], &mut Interpreter) -> SExpr + Send + Sync>> {
    let mut functions = HashMap::new();
    fn binary_op(op: fn(i32, i32) -> i32) -> impl Fn(&[SExpr]) -> i32 + Send + Sync {
        move |args: &[SExpr]| {
            let first = args[0].as_int().unwrap();
            let second = args[1].as_int().unwrap();
            op(first, second)
        }
    }

    fn insert<O: Into<SExpr>>(
        functions: &mut HashMap<
            String,
            Arc<dyn Fn(&[SExpr], &mut Interpreter) -> SExpr + Send + Sync>,
        >,
        name: impl ToString,
        op: impl Fn(&[SExpr]) -> O + Send + Sync + 'static,
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
        if end < 0 || start < 0 || start > end || end >= first.len() as _ {
            "".to_string()
        } else {
            first[start as usize..end as usize].to_string()
        }
    });
    insert(&mut functions, "str.at", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let index = args[1].as_int().unwrap();
        first
            .get(index as usize..(index as usize).saturating_add(1))
            .unwrap_or_default()
            .to_string()
    });
    insert(&mut functions, "str.to.int", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        SExpr::Atom(Atom::Int(first.parse::<i32>().unwrap_or(-1)))
    });
    insert(&mut functions, "str.indexof", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let second = args[1].as_string().unwrap();
        let offset = args[2].as_int().unwrap();
        first
            .get(offset as usize..)
            .and_then(|s| s.find(&second))
            .and_then(|i| offset.checked_add(i as i32))
            .unwrap_or(-1)
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
        SExpr::Atom(Atom::Bool(first.starts_with(&second)))
    });
    insert(&mut functions, "str.suffixof", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let second = args[1].as_string().unwrap();
        SExpr::Atom(Atom::Bool(first.ends_with(&second)))
    });
    insert(&mut functions, "str.contains", |args: &[SExpr]| {
        let first = args[0].as_string().unwrap();
        let second = args[1].as_string().unwrap();
        SExpr::Atom(Atom::Bool(first.contains(&second)))
    });

    insert(&mut functions, "int.to.str", |args: &[SExpr]| {
        let first = args[0].as_int().unwrap();
        if first >= 0 {
            SExpr::from(first.to_string())
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

    // (str.++ (int.to.str 0) (str.substr " " 0 1))
    let expr = SExpr::List(vec![
        SExpr::Atom(Atom::Ident("str.++".to_string())),
        SExpr::List(vec![
            SExpr::Atom(Atom::Ident("int.to.str".to_string())),
            SExpr::Atom(Atom::Int(0)),
        ]),
        SExpr::List(vec![
            SExpr::Atom(Atom::Ident("str.substr".to_string())),
            SExpr::Atom(Atom::String(" ".to_string())),
            SExpr::Atom(Atom::Int(0)),
            SExpr::Atom(Atom::Int(1)),
        ]),
    ]);
    let result = interpreter.eval(&expr);
    assert_eq!(result, SExpr::Atom(Atom::String("0 ".to_string())));
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

struct FailParser<T>(std::marker::PhantomData<T>);
impl<T: Clone> Parser for FailParser<T> {
    type Output = T;
    type PartialState = ();

    fn parse<'a>(
        &self,
        _: &Self::PartialState,
        _: &'a [u8],
    ) -> ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        Err(ParserError::msg("Fail"))
    }
}
impl<T: Clone> CreateParserState for FailParser<T> {
    fn create_parser_state(&self) -> Self::PartialState {}
}
struct SuccessParser<T>(T);
impl<T: Clone> Parser for SuccessParser<T> {
    type Output = T;
    type PartialState = ();

    fn parse<'a>(
        &self,
        _: &Self::PartialState,
        _: &'a [u8],
    ) -> ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        Ok(ParseStatus::Finished {
            result: self.0.clone(),
            remaining: &[],
        })
    }
}
impl<T: Clone> CreateParserState for SuccessParser<T> {
    fn create_parser_state(&self) -> Self::PartialState {}
}

fn create_grammar(
    synth_fun: SynthFun,
    prefix: &str,
    path: &Path,
    force_correct_tokenization: bool,
) -> Grammar<u32> {
    let tokenizer = Tokenizer::load_tokenizer(path);
    let huggingface_tokenizer = tokenizers::Tokenizer::from_file(path).unwrap();

    let prefix_tokenized = huggingface_tokenizer.encode_fast(prefix, false).unwrap();
    let prefix_tokenized = prefix_tokenized.get_ids();
    let [start @ .., last] = prefix_tokenized else {
        panic!("Prefix must have at least one token");
    };
    let last_as_string = huggingface_tokenizer.decode(&[*last], false).unwrap();

    let mut grammar = synth_fun.grammar();
    let new_start = "NewStart".to_string();
    grammar.rules.push(Rule {
        lhs: new_start.clone(),
        rhs: vec![
            vec![
                Symbol::Terminal(last_as_string.to_string()),
                Symbol::NonTerminal("Start".to_string()),
            ]
            .into(),
        ],
    });
    grammar.start = new_start;

    let mut grammar = grammar.split_terminals();
    grammar = grammar.to_cnf().unwrap();
    let grammar = grammar.replace_tokenizer_terminals(&tokenizer);
    let mut grammar = SlabGrammar::new(&grammar);
    let entry = grammar.rules.vacant_entry();
    let new_start = entry.key() as u32;
    let old_start = grammar.start;
    entry.insert(Rule {
        lhs: new_start,
        rhs: vec![
            start
                .iter()
                .map(|id| Symbol::Terminal(*id))
                .chain(std::iter::once(Symbol::NonTerminal(old_start)))
                .collect::<Vec<_>>()
                .into(),
        ],
    });
    grammar.start = new_start;
    grammar.garbage_collect_non_terminals();
    let merges = &tokenizer.merges;
    let mut processed_merges = Vec::new();
    for merge in merges {
        let changed = grammar.shortcut_merge(merge, !force_correct_tokenization);
        if changed {
            grammar.garbage_collect_non_terminals();
            grammar.deduplicate_non_terminals();
        }
        processed_merges.push(merge.clone());
    }

    grammar.inline_optimize();

    grammar.to_grammar()
}

#[derive(Serialize, Deserialize)]
struct Run {
    generation: u32,
    metadata: StructuredInferenceTimingInfo,
    pass: bool,
    entropy: f64,
    entropy_diff: f64,
    tokenization_error: bool,
    result: String,
    tokens: Vec<u32>,
}
