use dioxus::prelude::*;
use floneum_plugin::{
    exports::plugins::main::definitions::{
        Input, Output, PrimitiveValue, PrimitiveValueType, ValueType,
    },
    plugins::main::types::{GptNeoXType, LlamaType, ModelType, MptType},
};

use crate::{
    node_value::{NodeInput, NodeOutput},
    Signal,
};

#[inline_props]
pub fn ShowOutput(cx: Scope, value: Signal<NodeOutput>) -> Element {
    let output = value.read();
    let key = &output.definition.name;
    match &output.value {
        Output::Single(value) => {
            render! {
                div {
                    class: "flex flex-col",
                    "{key}:"
                    show_primitive_value(cx, value)
                }
            }
        }
        Output::Many(value) => {
            render! {
                div {
                    class: "flex flex-col",
                    "{key}:"
                    for value in &value {
                        show_primitive_value(cx, value)
                    }
                }
            }
        }
        _ => {
            render! {
                div {
                    class: "flex flex-col",
                    "{key}: Unset"
                }
            }
        }
    }
}

fn show_primitive_value<'a>(cx: &'a ScopeState, value: &PrimitiveValue) -> Element<'a> {
    match value {
        PrimitiveValue::Text(value) => {
            render! {"{value}"}
        }
        PrimitiveValue::Embedding(value) => {
            let first_five = value.vector.iter().take(5).collect::<Vec<_>>();
            render! {"{first_five:?}"}
        }
        PrimitiveValue::Model(id) => {
            render! {"Model: {id:?}"}
        }
        PrimitiveValue::Database(id) => {
            render! {"Database: {id:?}"}
        }
        PrimitiveValue::Number(value) => {
            render! {"{value}"}
        }
        PrimitiveValue::ModelType(ty) => {
            render! {"{ty.name()}"}
        }
        PrimitiveValue::Boolean(val) => {
            render! {"{val:?}"}
        }
        PrimitiveValue::Tab(id) => {
            render! {"Tab: {id:?}"}
        }
        PrimitiveValue::Node(id) => {
            render! {"Node: {id:?}"}
        }
    }
}

#[inline_props]
pub fn ModifyInput(cx: &ScopeState, value: Signal<NodeInput>) -> Element {
    let node = value;
    let current_value = node.read();
    let name = &current_value.definition.name;
    match &current_value.value {
        Input::Single(current_primitive) => match current_primitive {
            PrimitiveValue::Text(value) => {
                render! {
                    div {
                        class: "flex flex-col",
                        "{name}: "
                        input {
                            class: "border border-gray-400 rounded hover:border-gray-500 focus:outline-none focus:border-blue-500",
                            value: "{value}",
                            oninput: |e| {
                                node.write().value = Input::Single(PrimitiveValue::Text(e.value.to_string()));
                            }
                        }
                    }
                }
            }
            PrimitiveValue::Embedding(_)
            | PrimitiveValue::Model(_)
            | PrimitiveValue::Database(_)
            | PrimitiveValue::Tab(_)
            | PrimitiveValue::Node { .. } => None,
            PrimitiveValue::Number(value) => {
                render! {
                    div {
                        class: "flex flex-col",
                        "{name}: "
                        input {
                            class: "border border-gray-400 rounded hover:border-gray-500 focus:outline-none focus:border-blue-500",
                            r#type: "number",
                            value: "{value}",
                            oninput: |e| {
                                node
                                    .write().value = Input::Single(PrimitiveValue::Number(e.value.parse().unwrap_or(0)));
                            }
                        }
                    }
                }
            }
            PrimitiveValue::ModelType(ty) => {
                render! {
                    div {
                        class: "flex flex-col",
                        "{name}: "
                        select {
                            class: "border border-gray-400 rounded hover:border-gray-500 focus:outline-none focus:border-blue-500",
                            onchange: |e| {
                                node
                                    .write().value = Input::Single(
                                    PrimitiveValue::ModelType(
                                        model_type_from_str(&e.value)
                                            .unwrap_or(ModelType::Llama(LlamaType::LlamaThirteenChat)),
                                    ),
                                );
                            },
                            for variant in ModelType::VARIANTS {
                                option {
                                    value: "{variant.name()}",
                                    selected: "{variant.name() == ty.name()}",
                                    "{variant.name()}"
                                }
                            }
                        }
                    }
                }
            }
            PrimitiveValue::Boolean(val) => {
                render! {
                    div {
                        class: "flex flex-col",
                        "{name}: "
                        input {
                            class: "border border-gray-400 rounded hover:border-gray-500 focus:outline-none focus:border-blue-500",
                            r#type: "checkbox",
                            checked: "{val}",
                            onchange: |e| {
                                node.write().value = Input::Single(PrimitiveValue::Boolean(e.value == "on"));
                            }
                        }
                    }
                }
            }
        },
        Input::Many(values) => {
            render! {
                div {
                    div {
                        class: "flex flex-col",
                        "{name}: "
                        for value in values.iter() {
                            div { show_primitive_value(cx, value) }
                        }
                    }
                }
            }
        }
    }
}

pub trait Variants: Sized + 'static {
    const VARIANTS: &'static [Self];
}

impl Variants for ModelType {
    const VARIANTS: &'static [Self] = &[
        ModelType::Llama(LlamaType::Guanaco),
        ModelType::Llama(LlamaType::Orca),
        ModelType::Llama(LlamaType::Vicuna),
        ModelType::Llama(LlamaType::Wizardlm),
        ModelType::Llama(LlamaType::LlamaSevenChat),
        ModelType::Llama(LlamaType::LlamaThirteenChat),
        ModelType::GptNeoX(GptNeoXType::TinyPythia),
        ModelType::GptNeoX(GptNeoXType::LargePythia),
        ModelType::GptNeoX(GptNeoXType::Stablelm),
        ModelType::GptNeoX(GptNeoXType::DollySevenB),
        ModelType::Mpt(MptType::Base),
        ModelType::Mpt(MptType::Chat),
        ModelType::Mpt(MptType::Story),
        ModelType::Mpt(MptType::Instruct),
    ];
}

impl Variants for ValueType {
    const VARIANTS: &'static [Self] = &[
        ValueType::Single(PrimitiveValueType::Text),
        ValueType::Single(PrimitiveValueType::Number),
        ValueType::Single(PrimitiveValueType::Boolean),
        ValueType::Single(PrimitiveValueType::Embedding),
        ValueType::Single(PrimitiveValueType::Model),
        ValueType::Single(PrimitiveValueType::ModelType),
        ValueType::Single(PrimitiveValueType::Database),
        ValueType::Single(PrimitiveValueType::Tab),
        ValueType::Single(PrimitiveValueType::Node),
        ValueType::Single(PrimitiveValueType::Any),
        ValueType::Many(PrimitiveValueType::Text),
        ValueType::Many(PrimitiveValueType::Number),
        ValueType::Many(PrimitiveValueType::Boolean),
        ValueType::Many(PrimitiveValueType::Embedding),
        ValueType::Many(PrimitiveValueType::Model),
        ValueType::Many(PrimitiveValueType::ModelType),
        ValueType::Many(PrimitiveValueType::Database),
        ValueType::Many(PrimitiveValueType::Tab),
        ValueType::Many(PrimitiveValueType::Node),
        ValueType::Many(PrimitiveValueType::Any),
    ];
}

pub trait Named {
    fn name(&self) -> &'static str;
}

impl Named for ModelType {
    fn name(&self) -> &'static str {
        match self {
            ModelType::Llama(LlamaType::Guanaco) => "Guanaco",
            ModelType::Llama(LlamaType::Orca) => "Orca",
            ModelType::Llama(LlamaType::Vicuna) => "Vicuna",
            ModelType::Llama(LlamaType::Wizardlm) => "Wizardlm",
            ModelType::Llama(LlamaType::LlamaSevenChat) => "Llama Seven Chat",
            ModelType::Llama(LlamaType::LlamaThirteenChat) => "Llama Thirteen Chat",
            ModelType::GptNeoX(GptNeoXType::TinyPythia) => "Tiny Pythia",
            ModelType::GptNeoX(GptNeoXType::LargePythia) => "Large Pythia",
            ModelType::GptNeoX(GptNeoXType::Stablelm) => "Stablelm",
            ModelType::GptNeoX(GptNeoXType::DollySevenB) => "Dolly",
            ModelType::Mpt(MptType::Base) => "Mpt base",
            ModelType::Mpt(MptType::Chat) => "Mpt chat",
            ModelType::Mpt(MptType::Story) => "Mpt story",
            ModelType::Mpt(MptType::Instruct) => "Mpt instruct",
        }
    }
}

fn model_type_from_str(s: &str) -> Option<ModelType> {
    match &*s.to_lowercase() {
        "guanaco" => Some(ModelType::Llama(LlamaType::Guanaco)),
        "orca" => Some(ModelType::Llama(LlamaType::Orca)),
        "vicuna" => Some(ModelType::Llama(LlamaType::Vicuna)),
        "wizardlm" => Some(ModelType::Llama(LlamaType::Wizardlm)),
        "llama seven chat" => Some(ModelType::Llama(LlamaType::LlamaSevenChat)),
        "llama thirteen chat" => Some(ModelType::Llama(LlamaType::LlamaThirteenChat)),
        "tiny pythia" => Some(ModelType::GptNeoX(GptNeoXType::TinyPythia)),
        "large pythia" => Some(ModelType::GptNeoX(GptNeoXType::LargePythia)),
        "stablelm" => Some(ModelType::GptNeoX(GptNeoXType::Stablelm)),
        "dolly" => Some(ModelType::GptNeoX(GptNeoXType::DollySevenB)),
        "mpt base" => Some(ModelType::Mpt(MptType::Base)),
        "mpt chat" => Some(ModelType::Mpt(MptType::Chat)),
        "mpt story" => Some(ModelType::Mpt(MptType::Story)),
        "mpt instruct" => Some(ModelType::Mpt(MptType::Instruct)),
        _ => None,
    }
}

pub trait Colored {
    fn color(&self) -> String;
}

impl Colored for ValueType {
    fn color(&self) -> String {
        let index = Self::VARIANTS.iter().position(|v| v == self).unwrap();
        let hue = index * 360 / Self::VARIANTS.len();
        format!("hsl({hue}, 100%, 50%)")
    }
}
