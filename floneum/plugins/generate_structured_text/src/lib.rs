#![allow(unused)]

use core::panic;
use std::vec;

use floneum_rust::{plugins::main::imports::log_to_user, *};
use pest::{iterators::Pair, Parser};
use pest_derive::Parser;

#[export_plugin]
/// Calls a large language model to generate structured text. You can create a template for the language model to fill in. The model will fill in any segments that contain {**type**} where **type** is "", bool, or #
///
/// It is important to keep in mind that the language model is just generating text. Because the model is merely continuing the text you give it, the formatting of that text can be important.
///
/// It is commonly helpful to provide a few examples to the model before your new data so that the model can pick up on the pattern of the text
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![ModelType::Llama(LlamaType::LlamaSevenChat).into_input_value(), String::from("The following is a chat between a user and an assistant. The assistant helpfully and succinctly answers questions posed by the user.\n### USER\nWhat is 3 + 6?\n### ASSISTANT\n9\n### USER\nWhat is 1 + 1?\n### ASSISTANT\n2\n### USER\nWhat is 2 + 8?\n### ASSISTANT\n{#}").into_input_value()],
///         outputs: vec![String::from("10").into_return_value()],
///     },
/// ]
fn generate_structured_text(
    /// the model to use
    model: ModelType,
    /// the structure to use when running the model
    structure: String,
    /// the maximum length of the output
    max_output_length: i64,
) -> String {
    if !Model::model_downloaded(model) {
        log_to_user("downloading model... This could take several minutes");
    }

    let structure = structured_from_string(&structure);
    let session = Model::new(model);

    let max_output_length = if max_output_length == 0 {
        None
    } else {
        max_output_length.try_into().ok()
    };

    let mut responce = session.infer_structured("", max_output_length, structure);
    responce += "\n";

    responce
}

fn structured_from_string(input: &str) -> Structure {
    let pattern = StructuredParser::parse(Rule::format, input).map(|mut iter| iter.next());
    match pattern {
        Ok(Some(pattern)) => multiple_structured_from_rule(pattern),
        Err(err) => {
            log::error!("error parsing pattern: {:?}\n", err);
            Structure::str(..)
        }
        _ => Structure::str(..),
    }
}

fn multiple_structured_from_rule(rule: Pair<Rule>) -> Structure {
    let mut iter = rule.into_inner();
    let mut current = structured_from_rule(iter.next().unwrap());
    for item in iter {
        current = current.then(structured_from_rule(item));
    }
    current
}

fn structured_from_rule(rule: Pair<Rule>) -> Structure {
    match rule.as_rule() {
        Rule::literal => Structure::literal(rule.as_str()),
        Rule::string => Structure::str(..),
        Rule::boolean => Structure::boolean(),
        Rule::hashtag => Structure::float(),
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
            let separator = Structure::literal(",");
            let range = if let Some(rule) = iter.next().filter(|pair| pair.as_rule() == Rule::range)
            {
                let mut iter = rule.into_inner();
                let min = iter.next().unwrap().as_str().parse().unwrap();
                let max = iter.next().unwrap().as_str().parse().unwrap();
                min..=max
            } else {
                0..=u64::MAX
            };
            Structure::sequence_of(item, separator, range)
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
