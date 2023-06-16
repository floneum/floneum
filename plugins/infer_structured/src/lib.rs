#![allow(unused)]

use core::panic;
use std::vec;

use pest::{iterators::Pair, Parser};
use pest_derive::Parser;
use rust_adapter::*;

export_plugin_world!(Plugin);

pub struct Plugin;

impl Definitions for Plugin {
    fn structure() -> Definition {
        Definition {
            name: "structured inference".to_string(),
            description: "loads a model and runs it".to_string(),
            inputs: vec![
                IoDefinition {
                    name: "structure".to_string(),
                    ty: ValueType::Single(PrimitiveValueType::Text),
                },
                IoDefinition {
                    name: "max output length".to_string(),
                    ty: ValueType::Single(PrimitiveValueType::Number),
                },
            ],
            outputs: vec![IoDefinition {
                name: "output".to_string(),
                ty: ValueType::Single(PrimitiveValueType::Text),
            }],
        }
    }

    fn run(input: Vec<Value>) -> Vec<Value> {
        let model = ModelType::Llama(LlamaType::Vicuna);

        let session = ModelInstance::new(model);

        let structured = match &input[0] {
            Value::Single(PrimitiveValue::Text(text)) => text,
            _ => panic!("expected text input"),
        };

        let structure = structured_from_string(structured);

        let max_output_length = match &input[2] {
            Value::Single(PrimitiveValue::Number(num)) => *num,
            _ => panic!("expected number input"),
        };

        let max_output_length = if max_output_length == 0 {
            None
        } else {
            max_output_length.try_into().ok()
        };

        let mut responce = session.infer_structured("", max_output_length, structure);
        responce += "\n";


        vec![Value::Single(PrimitiveValue::Text(responce))]
    }
}

fn structured_from_string(input: &str) -> Structured {
    let pattern = StructuredParser::parse(Rule::format, input)
        .map(|mut iter| iter.next());
    match pattern {
        Ok(Some(pattern)) =>{
            multiple_structured_from_rule(pattern)
        },
        Err(err) => {
            Structured::str()
        },
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
