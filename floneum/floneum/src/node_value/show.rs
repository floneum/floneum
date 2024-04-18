use crate::node_value::Named;
use dioxus::prelude::*;
use floneum_plugin::plugins::main::types::*;

#[derive(Clone, Props, PartialEq)]
pub struct ShowOutputProps {
    name: String,
    ty: ValueType,
    value: Vec<PrimitiveValue>,
}

pub fn ShowOutput(props: ShowOutputProps) -> Element {
    let ShowOutputProps { name, ty, value } = &props;
    match ty {
        ValueType::Single(_) => {
            rsx! {
                div {
                    class: "flex flex-col whitespace-pre-line",
                    "{name}:\n"
                    {show_primitive_value(&value.first().unwrap().borrow())}
                }
            }
        }
        ValueType::Many(_) => {
            rsx! {
                div {
                    class: "flex flex-col",
                    "{name}:"
                    for value in &value {
                        div {
                            class: "whitespace-pre-line",
                            {show_primitive_value(&value.borrow())}
                        }
                    }
                }
            }
        }
    }
}

pub fn show_primitive_value(value: &PrimitiveValue) -> Element {
    match value {
        PrimitiveValue::Text(value)
        | PrimitiveValue::File(value)
        | PrimitiveValue::Folder(value) => {
            rsx! {"{value}"}
        }
        PrimitiveValue::Embedding(value) => {
            let first_five = value
                .vector
                .iter()
                .take(5)
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            rsx! {"[{first_five}, ...]"}
        }
        PrimitiveValue::Model(id) => {
            rsx! {"Model: {id:?}"}
        }
        PrimitiveValue::EmbeddingModel(id) => {
            rsx! {"Embedding Model: {id:?}"}
        }
        PrimitiveValue::Database(id) => {
            rsx! {"Database: {id:?}"}
        }
        PrimitiveValue::Number(value) => {
            rsx! {"{value}"}
        }
        PrimitiveValue::Float(value) => {
            rsx! {"{value}"}
        }
        PrimitiveValue::ModelType(ty) => {
            rsx! {"{ty.name()}"}
        }
        PrimitiveValue::EmbeddingModelType(ty) => {
            rsx! {"{ty.name()}"}
        }
        PrimitiveValue::Boolean(val) => {
            rsx! {"{val:?}"}
        }
        PrimitiveValue::Page(id) => {
            rsx! {"Page: {id:?}"}
        }
        PrimitiveValue::Node(id) => {
            rsx! {"Node: {id:?}"}
        }
    }
}

#[derive(Clone, Props, PartialEq)]
pub struct ShowInputProps {
    label: String,
    ty: ValueType,
    value: Vec<PrimitiveValue>,
}

pub fn ShowInput(props: ShowInputProps) -> Element {
    let ShowInputProps { label, ty, value } = &props;
    match ty {
        ValueType::Single(_) => {
            let value = value.first()?;
            rsx! {
                div {
                    class: "flex flex-col whitespace-pre-line",
                    "{label}:\n"
                    {show_primitive_value(value)}
                }
            }
        }
        ValueType::Many(_) => {
            rsx! {
                div {
                    class: "flex flex-col",
                    "{label}:"
                    for value in &value {
                        div {
                            class: "whitespace-pre-line",
                            {show_primitive_value(value)}
                        }
                    }
                }
            }
        }
    }
}
