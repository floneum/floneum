use crate::Color;
use dioxus::prelude::*;
use floneum_plugin::{
    exports::plugins::main::definitions::{Input, Output},
    plugins::main::types::*,
};
use std::path::PathBuf;

use crate::{node_value::NodeInput, Signal};

#[derive(Props, PartialEq)]
pub struct ShowOutputProps {
    name: String,
    value: Output,
}

pub fn ShowOutput(cx: Scope<ShowOutputProps>) -> Element {
    let ShowOutputProps { name, value } = &cx.props;
    match value {
        Output::Single(value) => {
            render! {
                div {
                    class: "flex flex-col whitespace-pre-line",
                    "{name}:\n"
                    show_primitive_value(cx, value)
                }
            }
        }
        Output::Many(value) => {
            render! {
                div {
                    class: "flex flex-col",
                    "{name}:"
                    for value in &value {
                        div {
                            class: "whitespace-pre-line",
                            show_primitive_value(cx, value)
                        }
                    }
                }
            }
        }
        _ => {
            render! {
                div {
                    class: "flex flex-col",
                    "{name}: Unset"
                }
            }
        }
    }
}

fn show_primitive_value<'a>(cx: &'a ScopeState, value: &PrimitiveValue) -> Element<'a> {
    match value {
        PrimitiveValue::Text(value)
        | PrimitiveValue::File(value)
        | PrimitiveValue::Folder(value) => {
            render! {"{value}"}
        }
        PrimitiveValue::Embedding(value) => {
            let first_five = value.vector.iter().take(5).collect::<Vec<_>>();
            render! {"{first_five:?}"}
        }
        PrimitiveValue::Model(id) => {
            render! {"Model: {id:?}"}
        }
        PrimitiveValue::EmbeddingModel(id) => {
            render! {"Embedding Model: {id:?}"}
        }
        PrimitiveValue::Database(id) => {
            render! {"Database: {id:?}"}
        }
        PrimitiveValue::Number(value) => {
            render! {"{value}"}
        }
        PrimitiveValue::Float(value) => {
            render! {"{value}"}
        }
        PrimitiveValue::ModelType(ty) => {
            render! {"{ty.name()}"}
        }
        PrimitiveValue::EmbeddingModelType(ty) => {
            render! {"{ty.name()}"}
        }
        PrimitiveValue::Boolean(val) => {
            render! {"{val:?}"}
        }
        PrimitiveValue::Page(id) => {
            render! {"Page: {id:?}"}
        }
        PrimitiveValue::Node(id) => {
            render! {"Node: {id:?}"}
        }
    }
}

#[derive(Props, PartialEq)]
pub struct ShowInputProps {
    label: String,
    value: Input,
}

pub fn ShowInput(cx: Scope<ShowInputProps>) -> Element {
    let ShowInputProps { label, value } = &cx.props;
    match value {
        Input::Single(value) => {
            render! {
                div {
                    class: "flex flex-col whitespace-pre-line",
                    "{label}:\n"
                    show_primitive_value(cx, value)
                }
            }
        }
        Input::Many(value) => {
            render! {
                div {
                    class: "flex flex-col",
                    "{label}:"
                    for value in &value {
                        div {
                            class: "whitespace-pre-line",
                            show_primitive_value(cx, value)
                        }
                    }
                }
            }
        }
    }
}

#[inline_props]
pub fn ModifyInput(cx: &ScopeState, value: Signal<NodeInput>) -> Element {
    let node = value;
    let current_value = node.read();
    let name = &current_value.definition.name;
    match current_value.value() {
        Input::Single(current_primitive) => match current_primitive {
            PrimitiveValue::Text(value) => {
                render! {
                    div {
                        class: "flex flex-col",
                        "{name}: "
                        textarea {
                            class: "border {Color::outline_color()} {Color::foreground_color()} {Color::foreground_hover()} rounded focus:outline-none focus:border-blue-500",
                            value: "{value}",
                            oninput: |e| {
                                node.write().value = vec![Input::Single(PrimitiveValue::Text(e.value.to_string()))];
                            }
                        }
                    }
                }
            }
            PrimitiveValue::File(file) => {
                render! {
                    button {
                        class: "border {Color::outline_color()} {Color::foreground_hover()} rounded focus:outline-none focus:border-blue-500",
                        onclick: |_| {
                            node.write().value = rfd::FileDialog::new()
                                .set_directory("./sandbox")
                                .set_file_name("Floneum")
                                .set_title("Select File")
                                .save_file()
                                .map(|path| vec![Input::Single(PrimitiveValue::File(path.strip_prefix(PathBuf::from("./sandbox").canonicalize().unwrap()).unwrap_or(&path).to_string_lossy().to_string()))])
                                .unwrap_or_else(|| vec![Input::Single(PrimitiveValue::File("".to_string()))])
                        },
                        "Select File"
                    }
                    "{file}"
                }
            }
            PrimitiveValue::Folder(folder) => {
                render! {
                    button {
                        class: "border {Color::outline_color()} rounded {Color::foreground_hover()} focus:outline-none focus:border-blue-500",
                        onclick: |_| {
                            node.write().value = rfd::FileDialog::new()
                                .set_directory("./sandbox")
                                .set_file_name("Floneum")
                                .set_title("Select Folder")
                                .pick_folder()
                                .map(|path| vec![Input::Single(PrimitiveValue::File(path.strip_prefix(PathBuf::from("./sandbox").canonicalize().unwrap()).unwrap_or(&path).to_string_lossy().to_string()))])
                                .unwrap_or_else(|| vec![Input::Single(PrimitiveValue::File("".to_string()))]);
                        },
                        "Select Folder"
                    }
                    "{folder}"
                }
            }
            PrimitiveValue::Embedding(_)
            | PrimitiveValue::Model(_)
            | PrimitiveValue::EmbeddingModel(_)
            | PrimitiveValue::Database(_)
            | PrimitiveValue::Page(_)
            | PrimitiveValue::Node(_) => show_primitive_value(cx, &current_primitive),
            PrimitiveValue::Number(value) => {
                render! {
                    div {
                        class: "flex flex-col",
                        "{name}: "
                        input {
                            class: "border {Color::outline_color()} {Color::foreground_color()} rounded {Color::foreground_hover()} focus:outline-none focus:border-blue-500",
                            r#type: "number",
                            value: "{value}",
                            oninput: |e| {
                                node
                                    .write().value = vec![Input::Single(PrimitiveValue::Number(e.value.parse().unwrap_or(0)))];
                            }
                        }
                    }
                }
            }
            PrimitiveValue::Float(value) => {
                render! {
                    div {
                        class: "flex flex-col",
                        "{name}: "
                        input {
                            class: "border {Color::outline_color()} {Color::foreground_color()} rounded {Color::foreground_hover()} focus:outline-none focus:border-blue-500",
                            r#type: "number",
                            value: "{value}",
                            oninput: |e| {
                                node
                                    .write().value = vec![Input::Single(PrimitiveValue::Float(e.value.parse().unwrap_or(0.)))];
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
                            class: "border {Color::outline_color()} {Color::foreground_color()} rounded {Color::foreground_hover()} focus:outline-none focus:border-blue-500",
                            style: "-webkit-appearance:none; -moz-appearance:none; -ms-appearance:none; appearance: none;",
                            onchange: |e| {
                                node
                                    .write().value = vec![Input::Single(
                                    PrimitiveValue::ModelType(
                                        model_type_from_str(&e.value)
                                            .unwrap_or(ModelType::MistralSeven),
                                    ),
                                )];
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
            PrimitiveValue::EmbeddingModelType(ty) => {
                render! {
                    div {
                        class: "flex flex-col",
                        "{name}: "
                        select {
                            class: "border {Color::outline_color()} {Color::foreground_color()} rounded {Color::foreground_hover()} focus:outline-none focus:border-blue-500",
                            style: "-webkit-appearance:none; -moz-appearance:none; -ms-appearance:none; appearance: none;",
                            onchange: |e| {
                                node
                                    .write().value = vec![Input::Single(
                                    PrimitiveValue::EmbeddingModelType(
                                        embedding_model_type_from_str(&e.value)
                                            .unwrap_or(EmbeddingModelType::Bert),
                                    ),
                                )];
                            },
                            for variant in EmbeddingModelType::VARIANTS {
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
                            class: "border {Color::outline_color()} {Color::foreground_color()} rounded {Color::foreground_hover()} focus:outline-none focus:border-blue-500",
                            r#type: "checkbox",
                            checked: "{val}",
                            onchange: |e| {
                                node.write().value = vec![Input::Single(PrimitiveValue::Boolean(e.value == "on"))];
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
                            div {
                                class: "whitespace-pre-line",
                                show_primitive_value(cx, value)
                            }
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
        ModelType::MistralSeven,
        ModelType::MistralSevenInstruct,
        ModelType::MistralSevenInstructTwo,
        ModelType::ZephyrSevenAlpha,
        ModelType::ZephyrSevenBeta,
        ModelType::OpenChatSeven,
        ModelType::StarlingSevenAlpha,
        ModelType::TinyLlamaChat,
        ModelType::TinyLlama,
        ModelType::LlamaSeven,
        ModelType::LlamaThirteen,
        ModelType::LlamaSeventy,
        ModelType::LlamaSevenChat,
        ModelType::LlamaThirteenChat,
        ModelType::LlamaSeventyChat,
        ModelType::LlamaSevenCode,
        ModelType::LlamaThirteenCode,
        ModelType::LlamaThirtyFourCode,
        ModelType::SolarTen,
        ModelType::SolarTenInstruct,
        ModelType::PhiOne,
        ModelType::PhiOnePointFive,
        ModelType::PhiTwo,
        ModelType::PuffinPhiTwo,
        ModelType::DolphinPhiTwo,
    ];
}

impl Variants for EmbeddingModelType {
    const VARIANTS: &'static [Self] = &[EmbeddingModelType::Bert];
}

impl Variants for PrimitiveValueType {
    const VARIANTS: &'static [Self] = &[
        PrimitiveValueType::Text,
        PrimitiveValueType::File,
        PrimitiveValueType::Folder,
        PrimitiveValueType::Number,
        PrimitiveValueType::Float,
        PrimitiveValueType::Boolean,
        PrimitiveValueType::Embedding,
        PrimitiveValueType::Model,
        PrimitiveValueType::ModelType,
        PrimitiveValueType::Database,
        PrimitiveValueType::Page,
        PrimitiveValueType::Node,
        PrimitiveValueType::Any,
    ];
}

impl Variants for ValueType {
    const VARIANTS: &'static [Self] = &[
        ValueType::Single(PrimitiveValueType::Text),
        ValueType::Single(PrimitiveValueType::File),
        ValueType::Single(PrimitiveValueType::Folder),
        ValueType::Single(PrimitiveValueType::Number),
        ValueType::Single(PrimitiveValueType::Float),
        ValueType::Single(PrimitiveValueType::Boolean),
        ValueType::Single(PrimitiveValueType::Embedding),
        ValueType::Single(PrimitiveValueType::Model),
        ValueType::Single(PrimitiveValueType::ModelType),
        ValueType::Single(PrimitiveValueType::Database),
        ValueType::Single(PrimitiveValueType::Page),
        ValueType::Single(PrimitiveValueType::Node),
        ValueType::Single(PrimitiveValueType::Any),
        ValueType::Many(PrimitiveValueType::Text),
        ValueType::Many(PrimitiveValueType::File),
        ValueType::Many(PrimitiveValueType::Folder),
        ValueType::Many(PrimitiveValueType::Number),
        ValueType::Many(PrimitiveValueType::Float),
        ValueType::Many(PrimitiveValueType::Boolean),
        ValueType::Many(PrimitiveValueType::Embedding),
        ValueType::Many(PrimitiveValueType::Model),
        ValueType::Many(PrimitiveValueType::ModelType),
        ValueType::Many(PrimitiveValueType::Database),
        ValueType::Many(PrimitiveValueType::Page),
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
            ModelType::MistralSeven => "Mistral Seven",
            ModelType::MistralSevenInstruct => "Mistral Seven Instruct",
            ModelType::MistralSevenInstructTwo => "Mistral Seven Instruct Two",
            ModelType::ZephyrSevenAlpha => "Zephyr Seven Alpha",
            ModelType::ZephyrSevenBeta => "Zephyr Seven Beta",
            ModelType::OpenChatSeven => "Open Chat Seven",
            ModelType::StarlingSevenAlpha => "Starling Seven Alpha",
            ModelType::TinyLlamaChat => "Tiny Llama Chat",
            ModelType::TinyLlama => "Tiny Llama",
            ModelType::LlamaSeven => "Llama Seven",
            ModelType::LlamaThirteen => "Llama Thirteen",
            ModelType::LlamaSeventy => "Llama Seventy",
            ModelType::LlamaSevenChat => "Llama Seven Chat",
            ModelType::LlamaThirteenChat => "Llama Thirteen Chat",
            ModelType::LlamaSeventyChat => "Llama Seventy Chat",
            ModelType::LlamaSevenCode => "Llama Seven Code",
            ModelType::LlamaThirteenCode => "Llama Thirteen Code",
            ModelType::LlamaThirtyFourCode => "Llama Thirty Four Code",
            ModelType::SolarTen => "Solar Ten",
            ModelType::SolarTenInstruct => "Solar Ten Instruct",
            ModelType::PhiOne => "Phi One",
            ModelType::PhiOnePointFive => "Phi One Point Five",
            ModelType::PhiTwo => "Phi Two",
            ModelType::PuffinPhiTwo => "Puffin Phi Two",
            ModelType::DolphinPhiTwo => "Dolphin Phi Two",
        }
    }
}

impl Named for EmbeddingModelType {
    fn name(&self) -> &'static str {
        match self {
            EmbeddingModelType::Bert => "Bert",
        }
    }
}

fn model_type_from_str(s: &str) -> Option<ModelType> {
    match &*s.to_lowercase() {
        "mistral seven" => Some(ModelType::MistralSeven),
        "mistral seven instruct" => Some(ModelType::MistralSevenInstruct),
        "mistral seven instruct two" => Some(ModelType::MistralSevenInstructTwo),
        "zephyr seven alpha" => Some(ModelType::ZephyrSevenAlpha),
        "zephyr seven beta" => Some(ModelType::ZephyrSevenBeta),
        "open chat seven" => Some(ModelType::OpenChatSeven),
        "starling seven alpha" => Some(ModelType::StarlingSevenAlpha),
        "tiny llama chat" => Some(ModelType::TinyLlamaChat),
        "tiny llama" => Some(ModelType::TinyLlama),
        "llama seven" => Some(ModelType::LlamaSeven),
        "llama thirteen" => Some(ModelType::LlamaThirteen),
        "llama seventy" => Some(ModelType::LlamaSeventy),
        "llama seven chat" => Some(ModelType::LlamaSevenChat),
        "llama thirteen chat" => Some(ModelType::LlamaThirteenChat),
        "llama seventy chat" => Some(ModelType::LlamaSeventyChat),
        "llama seven code" => Some(ModelType::LlamaSevenCode),
        "llama thirteen code" => Some(ModelType::LlamaThirteenCode),
        "llama thirty four code" => Some(ModelType::LlamaThirtyFourCode),
        "solar ten" => Some(ModelType::SolarTen),
        "solar ten instruct" => Some(ModelType::SolarTenInstruct),
        "phi one" => Some(ModelType::PhiOne),
        "phi one point five" => Some(ModelType::PhiOnePointFive),
        "phi two" => Some(ModelType::PhiTwo),
        "puffin phi two" => Some(ModelType::PuffinPhiTwo),
        "dolphin phi two" => Some(ModelType::DolphinPhiTwo),
        _ => None,
    }
}

fn embedding_model_type_from_str(s: &str) -> Option<EmbeddingModelType> {
    match &*s.to_lowercase() {
        "bert" => Some(EmbeddingModelType::Bert),
        _ => None,
    }
}

pub trait Colored {
    fn color(&self) -> String;
}

impl Colored for ValueType {
    fn color(&self) -> String {
        match self {
            ValueType::Single(ty) => ty.color(),
            ValueType::Many(ty) => ty.color(),
        }
    }
}

impl Colored for PrimitiveValueType {
    fn color(&self) -> String {
        let index = Self::VARIANTS.iter().position(|v| v == self).unwrap();
        let hue = index * 360 / Self::VARIANTS.len();
        format!("hsl({hue}, 100%, 50%)")
    }
}
