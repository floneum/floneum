use dioxus::prelude::*;
use floneum_plugin::plugins::main::types::*;
use crate::node_value::Named;

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

pub fn show_primitive_value(value: &BorrowedPrimitiveValue) -> Element {
    match value {
        BorrowedPrimitiveValue::Text(value)
        | BorrowedPrimitiveValue::File(value)
        | BorrowedPrimitiveValue::Folder(value) => {
            rsx! {"{value}"}
        }
        BorrowedPrimitiveValue::Embedding(value) => {
            let first_five = value
                .vector
                .iter()
                .take(5)
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            rsx! {"[{first_five}, ...]"}
        }
        BorrowedPrimitiveValue::Model(id) => {
            rsx! {"Model: {id:?}"}
        }
        BorrowedPrimitiveValue::EmbeddingModel(id) => {
            rsx! {"Embedding Model: {id:?}"}
        }
        BorrowedPrimitiveValue::Database(id) => {
            rsx! {"Database: {id:?}"}
        }
        BorrowedPrimitiveValue::Number(value) => {
            rsx! {"{value}"}
        }
        BorrowedPrimitiveValue::ModelType(ty) => {
            rsx! {"{ty.name()}"}
        }
        BorrowedPrimitiveValue::EmbeddingModelType(ty) => {
            rsx! {"{ty.name()}"}
        }
        BorrowedPrimitiveValue::Boolean(val) => {
            rsx! {"{val:?}"}
        }
        BorrowedPrimitiveValue::Page(id) => {
            rsx! {"Page: {id:?}"}
        }
        BorrowedPrimitiveValue::Node(id) => {
            rsx! {"Node: {id:?}"}
        }
    }
}

#[derive(Clone, Props, PartialEq)]
pub struct ShowInputProps {
    label: String,
    ty: ValueType,
    value: Vec<BorrowedPrimitiveValue>,
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
