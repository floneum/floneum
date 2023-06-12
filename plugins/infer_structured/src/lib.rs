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
                    name: "input".to_string(),
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

        let text_input = match &input[1] {
            Value::Single(PrimitiveValue::Text(text)) => text,
            _ => panic!("expected text input"),
        };

        let max_output_length = match &input[2] {
            Value::Single(PrimitiveValue::Number(num)) => *num,
            _ => panic!("expected number input"),
        };

        let max_output_length = if max_output_length == 0 {
            None
        }else{
            Some(max_output_length)
        };

        let mut responce = session.infer_structured(&text_input, max_output_length, structure);
        responce += "\n";

        print(&responce);

        vec![Value::Single(PrimitiveValue::Text(responce))]
    }
}

fn structured_from_string(input: &str) -> Structured {
    let Some(rule) = StructuredParser::parse(Rule::value, input).ok().and_then(|mut iter| iter.next())
        else {
            return Structured::str();
        };

    structured_from_rule(rule)
}

fn structured_from_rule(rule: Pair<Rule>) -> Structured {
    match rule.as_rule() {
        Rule::empty_string => Structured::str(),
        Rule::boolean => Structured::boolean(),
        Rule::number => Structured::num(),
        Rule::array => {
            Structured::sequence_of(structured_from_rule(rule.into_inner().next().unwrap()))
        }
        Rule::object => {
            let mut pairs = rule.into_inner();
            let mut fields = Vec::new();
            while let Some(pair) = pairs.next() {
                let mut inner = pair.into_inner();
                let name = inner.next().unwrap().as_str();
                let name = name[1..name.len()-2].to_string();
                let value = structured_from_rule(inner.next().unwrap());
                fields.push((name, value));
            }
            Structured::map_of(fields)
        }
        _ => {
            let error = format!("unexpected rule: {:?}", rule);
            print(&error);
            todo!();
        },
    }
}

#[derive(Parser)]
#[grammar = "structured.pest"]
struct StructuredParser;
