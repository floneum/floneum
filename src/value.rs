use dioxus::prelude::*;
use floneum_plugin::{
    exports::plugins::main::definitions::{Input, PrimitiveValue},
    plugins::main::types::{GptNeoXType, LlamaType, ModelType, MptType},
};

use crate::LocalSubscription;

fn show_primitive_value<'a>(cx: &'a ScopeState, value: &PrimitiveValue) -> Element<'a> {
    match value {
        PrimitiveValue::Text(value) => {
            render! {"{value}"}
        }
        PrimitiveValue::Embedding(value) => {
            render! {"{&value.vector[..5]:?}"}
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
fn ModifyInput(cx: &ScopeState, param_name: String, value: LocalSubscription<Input>) -> Element {
    let current_value = value.use_(cx);
    match &*current_value.read() {
        Input::Single(current_primitive) => match current_primitive {
            PrimitiveValue::Text(value) => {
                render! {
                    input {
                        value: "{value}",
                        oninput: |e| {
                            *current_value.write() = Input::Single(PrimitiveValue::Text(e.value.to_string()));
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
                    input {
                        r#type: "number",
                        value: "{value}",
                        oninput: |e| {
                            *current_value
                                .write() = Input::Single(PrimitiveValue::Number(e.value.parse().unwrap_or(0)));
                        }
                    }
                }
            }
            PrimitiveValue::ModelType(ty) => {
                render! {
                    select { onchange: |e| {
                            *current_value
                                .write() = Input::Single(
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
            PrimitiveValue::Boolean(val) => {
                render! {
                    input {
                        r#type: "checkbox",
                        checked: "{val}",
                        onchange: |e| {
                            *current_value.write() = Input::Single(PrimitiveValue::Boolean(e.value == "on"));
                        }
                    }
                }
            }
        },
        Input::Many(values) => {
            render! {
                div {
                    for value in values.iter() {
                        div { show_primitive_value(cx, value) }
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
