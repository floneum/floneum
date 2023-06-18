#![allow(unused)]

use core::panic;
use std::vec;

use pest::{iterators::Pair, Parser};
use pest_derive::Parser;
use rust_adapter::*;

#[export_plugin]
/// loads a model and runs it
fn structured_inference(
    /// the structure to use when running the model
    structure: String,
    /// the maximum length of the output
    max_output_length: i64,
) -> String {
    let structure = structured_from_string(&structure);
    let model = ModelType::Llama(LlamaType::Vicuna);
    let session = ModelInstance::new(model);

    let max_output_length = if max_output_length == 0 {
        None
    } else {
        max_output_length.try_into().ok()
    };

    let mut responce = session.infer_structured("", max_output_length, structure);
    responce += "\n";

    print(&responce);

    responce
}

fn structured_from_string(input: &str) -> Structured {
    let pattern = StructuredParser::parse(Rule::format, input).map(|mut iter| iter.next());
    match pattern {
        Ok(Some(pattern)) => multiple_structured_from_rule(pattern),
        Err(err) => Structured::str(),
        _ => Structured::str(),
    }
}

fn multiple_structured_from_rule(rule: Pair<Rule>) -> Structured {
    let mut iter = rule.into_inner();
    let mut current = structured_from_rule(iter.next().unwrap());
    for item in iter {
        current = current.then(structured_from_rule(item));
    }
    current
}

fn structured_from_rule(rule: Pair<Rule>) -> Structured {
    match rule.as_rule() {
        Rule::literal => Structured::literal(rule.as_str()),
        Rule::string => Structured::str(),
        Rule::boolean => Structured::boolean(),
        Rule::hashtag => Structured::float(),
        Rule::either => {
            let mut iter = rule.into_inner();
            let mut current = structured_from_rule(iter.next().unwrap());
            for other in iter {
                current = current.or(structured_from_rule(other));
            }
            current
        }
        Rule::array => {
            let mut iter = rule.into_inner();
            let item = multiple_structured_from_rule(iter.next().unwrap());
            let seperator = Structured::literal(",");
            let range = if let Some(rule) = iter.next().filter(|pair| pair.as_rule() == Rule::range)
            {
                let mut iter = rule.into_inner();
                let min = iter.next().unwrap().as_str().parse().unwrap();
                let max = iter.next().unwrap().as_str().parse().unwrap();
                min..=max
            } else {
                0..=u64::MAX
            };
            Structured::sequence_of(item, seperator, range)
        }
        _ => {
            let error = format!("unexpected rule: {:?}\n", rule);
            todo!();
        }
    }
}

#[derive(Parser)]
#[grammar = "structured.pest"]
struct StructuredParser;
