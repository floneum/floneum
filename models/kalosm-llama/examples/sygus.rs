// ;; The background theory is linear integer arithmetic
// (set-logic LIA)

// ;; Name and signature of the function to be synthesized
// (synth-fun max2 ((x Int) (y Int)) Int

//     ;; Declare the non-terminals that would be used in the grammar
//     ((I Int) (B Bool))

//     ;; Define the grammar for allowed implementations of max2
//     ((I Int (x y 0 1
//              (+ I I) (- I I)
//              (ite B I I)))
//      (B Bool ((and B B) (or B B) (not B)
//               (= I I) (<= I I) (>= I I))))
// )

// (declare-var x Int)
// (declare-var y Int)

// ;; Define the semantic constraints on the function
// (constraint (>= (max2 x y) x))
// (constraint (>= (max2 x y) y))
// (constraint (or (= x (max2 x y)) (= y (max2 x y))))

// (check-synth)

// (set-logic SLIA)

// (synth-fun f ((firstname String) (lastname String)) String
//     ((Start String (ntString))
//      (ntString String (firstname lastname " " "."
//                        (str.++ ntString ntString)
//                        (str.replace ntString ntString ntString)
//                        (str.at ntString ntInt)
//                        (str.substr ntString ntInt ntInt)))
//       (ntInt Int (0 1 2
//                   (+ ntInt ntInt)
//                   (- ntInt ntInt)
//                   (str.len ntString)
//                   (str.indexof ntString ntString ntInt)))
//       (ntBool Bool (true false
//                     (str.prefixof ntString ntString)
//                     (str.suffixof ntString ntString)))))

// (declare-var firstname String)
// (declare-var lastname String)

// (constraint (= (f "Nancy" "FreeHafer") "Nancy F."))
// (constraint (= (f "Andrew" "Cencici") "Andrew C."))
// (constraint (= (f "Jan" "Kotas") "Jan K."))
// (constraint (= (f "Mariya" "Sergienko") "Mariya S."))

// (check-synth)

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use kalosm::language::*;
use kalosm_llama::{EvaluationTrie, LlamaModel};
use kalosm_sample::{
    ArcParser, CreateParserState, LazyParser, LiteralParser, Parse, Parser, ParserExt,
    SendCreateParserState, SequenceParser,
};
use std::io::Write;

#[derive(Clone, Debug)]
enum Expression {
    Literal(Literal),
    Call(Call),
}

#[derive(Clone, Debug)]
enum Literal {
    Int(i32),
    Bool(bool),
    String(String),
}

#[derive(Clone, Debug)]
struct Call {
    name: String,
    args: Vec<Expression>,
}

struct FunctionMap {
    functions_with_return_type: HashMap<TypeSchema, Arc<RwLock<Vec<ArcParser<Expression>>>>>,
}

impl FunctionMap {
    fn new() -> Self {
        FunctionMap {
            functions_with_return_type: TypeSchema::ALL
                .iter()
                .map(|ty| {
                    (
                        *ty,
                        Arc::new(RwLock::new(vec![ty
                            .literal_parser()
                            .map_output(Expression::Literal)
                            .boxed()])),
                    )
                })
                .collect(),
        }
    }

    fn parser(&self, ty: TypeSchema) -> ArcParser<Expression> {
        let parser_list = self.functions_with_return_type[&ty].clone();
        LazyParser::new(move || {
            let parser_list = parser_list.read().unwrap();
            let mut parser_list_iter = parser_list.iter();
            let mut parser = parser_list_iter.next().unwrap().clone();
            for p in parser_list_iter {
                parser = parser.or(p.clone()).boxed();
            }
            parser
        })
        .boxed()
    }
}

struct FunctionSchema {
    name: String,
    args: Vec<TypeSchema>,
    return_type: TypeSchema,
}

impl FunctionSchema {
    fn new(name: String, args: Vec<TypeSchema>, return_type: TypeSchema) -> Self {
        FunctionSchema {
            name,
            args,
            return_type,
        }
    }

    fn arg_parser(
        &self,
        functions: &FunctionMap,
    ) -> impl SendCreateParserState<Output = Vec<Expression>> {
        let parsers: Vec<_> = self.args.iter().map(|arg| functions.parser(*arg)).collect();

        let mut parser = LiteralParser::new("").map_output(|_| Vec::new()).boxed();
        for p in parsers {
            parser = parser
                .then_literal(" ")
                .then(p)
                .map_output(|(mut args, arg)| {
                    args.push(arg);
                    args
                })
                .boxed();
        }
        parser
    }

    fn parser(&self, functions: &FunctionMap) -> impl SendCreateParserState<Output = Call> {
        let name = self.name.clone();
        LiteralParser::new(format!("({}", self.name))
            .ignore_output_then(self.arg_parser(functions))
            .then_literal(")")
            .map_output(move |args| Call {
                name: name.clone(),
                args,
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TypeSchema {
    Int,
    Bool,
    String,
}

impl TypeSchema {
    const ALL: [TypeSchema; 3] = [TypeSchema::Int, TypeSchema::Bool, TypeSchema::String];

    fn literal_parser(&self) -> impl SendCreateParserState<Output = Literal> {
        match self {
            TypeSchema::Int => i32::new_parser()
                .map_output(|value| Literal::Int(value))
                .boxed(),
            TypeSchema::Bool => bool::new_parser()
                .map_output(|value| Literal::Bool(value))
                .boxed(),
            TypeSchema::String => String::new_parser()
                .map_output(|value| Literal::String(value))
                .boxed(),
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();

    let mut function_map = FunctionMap::new();

    let function_schemas = vec![
        FunctionSchema::new(
            "max2".to_string(),
            vec![TypeSchema::Int, TypeSchema::Int],
            TypeSchema::Int,
        ),
        FunctionSchema::new(
            "str.len".to_string(),
            vec![TypeSchema::String],
            TypeSchema::Int,
        ),
    ];

    for function_schema in function_schemas {
        let parser = function_schema.parser(&function_map);
        function_map
            .functions_with_return_type
            .entry(function_schema.return_type)
            .or_default()
            .write()
            .unwrap()
            .push(parser.map_output(Expression::Call).boxed());
    }

    let parser = function_map.parser(TypeSchema::Int);
    let result = parser
        .parse(&parser.create_parser_state(), b"(max2 1 2)")
        .unwrap();
    println!("{:?}", result);

    let result = parser
        .parse(
            &parser.create_parser_state(),
            b"(max2 (str.len \"hello\") (max2 1 2))",
        )
        .unwrap();
    println!("{:?}", result);

    let another_result = parser
        .parse(&parser.create_parser_state(), b"(str.len \"hello\")")
        .unwrap();
    println!("{:?}", another_result);

    let sampler = GenerationParameters::new();

    let mut llm = LlamaModel::from_builder(
        Llama::builder().with_source(LlamaSource::qwen_2_5_0_5b_instruct()),
        ModelLoadingProgress::multi_bar_loading_indicator(),
    )
    .await
    .unwrap();

    tokio::task::spawn_blocking(move || {
       let mut trie = EvaluationTrie::new();
         let mut last_entropy = 0.0;
         let task = r#"Write code for the maximum length of the two strings "hello" and "world". Functions available are: (max2 int int), and (str.len string)."#;
       for generation in 0.. {
            let mut session = llm.new_session();
            let prompt = format!("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{task}<|im_end|>\n<|im_start|>assistant\nHere is the code:\n");

            let output = llm.generate_structured_with_trie(
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
            ).unwrap();
            println!("\n\n");

            println!("generation {generation}:\n{output:?}");
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
   }).await.unwrap();
}
