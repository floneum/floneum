use crate::node_value::embedding_model_type_from_str;
use crate::node_value::model_type_from_str;
use crate::node_value::Named;
use crate::node_value::Variants;
use crate::show_primitive_value;
use crate::{node_value::NodeInput, Signal};
use dioxus::prelude::*;
use floneum_plugin::plugins::main::types::*;
use std::path::PathBuf;
use std::rc::Rc;

#[component]
pub fn ModifyInput(node: Signal<NodeInput>) -> Element {
    let current_value = node.read();
    let name = &current_value.definition.name;
    let values = current_value.value();
    match &current_value.definition.ty {
        ValueType::Single(_) => rsx! {
            div {
                class: "flex flex-col",
                "{name}: "
                ModifySingleValue {
                    value: values[0].clone(),
                    set_value: Rc::new(move |value| {
                        node.write_unchecked().value = vec![vec![value]];
                    }),
                }
            }
        },
        ValueType::Many(_) => {
            rsx! {
                div {
                    div {
                        class: "flex flex-col",
                        "{name}: "
                        for (i, value) in values.into_iter().enumerate() {
                            div {
                                class: "whitespace-pre-line",
                                ModifySingleValue {
                                    value,
                                    set_value: Rc::new(move |value| {
                                        node.write_unchecked().value[0][i] = value;
                                    }),
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Props)]
struct ModifySingleValueProps {
    value: PrimitiveValue,
    set_value: Rc<dyn Fn(PrimitiveValue)>,
}

impl PartialEq for ModifySingleValueProps {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

fn ModifySingleValue(props: ModifySingleValueProps) -> Element {
    let ModifySingleValueProps { value, set_value } = props;
    match value {
        PrimitiveValue::Text(value) => {
            rsx! {
                textarea {
                    class: "border rounded focus:outline-none focus:border-blue-500",
                    value: "{value}",
                    oninput: move |e| {
                        set_value(PrimitiveValue::Text(e.value()));
                    }
                }
            }
        }
        PrimitiveValue::File(file) => {
            rsx! {
                button {
                    class: "border rounded focus:outline-none focus:border-blue-500",
                    onclick: move |_| {
                        set_value(rfd::FileDialog::new()
                            .set_directory("./sandbox")
                            .set_file_name("Floneum")
                            .set_title("Select File")
                            .save_file()
                            .map(|path| PrimitiveValue::File(path.strip_prefix(PathBuf::from("./sandbox").canonicalize().unwrap()).unwrap_or(&path).to_string_lossy().to_string()))
                            .unwrap_or_else(|| PrimitiveValue::File("".to_string())));
                    },
                    "Select File"
                }
                "{file}"
            }
        }
        PrimitiveValue::Folder(folder) => {
            rsx! {
                button {
                    class: "border rounded focus:outline-none focus:border-blue-500",
                    onclick: move |_| {
                        set_value(rfd::FileDialog::new()
                            .set_directory("./sandbox")
                            .set_file_name("Floneum")
                            .set_title("Select Folder")
                            .pick_folder()
                            .map(|path| PrimitiveValue::File(path.strip_prefix(PathBuf::from("./sandbox").canonicalize().unwrap()).unwrap_or(&path).to_string_lossy().to_string()
                        ))
                            .unwrap_or_else(|| PrimitiveValue::File("".to_string())))
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
        | PrimitiveValue::Node(_) => show_primitive_value(&value),
        PrimitiveValue::Number(value) => {
            rsx! {
                input {
                    class: "border rounded focus:outline-none focus:border-blue-500",
                    r#type: "number",
                    value: "{value}",
                    oninput: move |e| {
                        set_value(PrimitiveValue::Number(e.value().parse().unwrap_or(0)));
                    }
                }
            }
        }
        PrimitiveValue::Float(value) => {
            rsx! {
                input {
                    class: "border rounded focus:outline-none focus:border-blue-500",
                    r#type: "number",
                    value: "{value}",
                    oninput: move |e| {
                        set_value(PrimitiveValue::Float(e.value().parse().unwrap_or(0.)));
                    }
                }
            }
        }
        PrimitiveValue::ModelType(ty) => {
            rsx! {
                select {
                    class: "border rounded focus:outline-none focus:border-blue-500",
                    style: "-webkit-appearance:none; -moz-appearance:none; -ms-appearance:none; appearance: none;",
                    onchange: move |e| {
                        set_value(
                            PrimitiveValue::ModelType(
                                model_type_from_str(&e.value())
                                    .unwrap_or(ModelType::MistralSeven),
                            ),
                        );
                    },
                    for variant in ModelType::VARIANTS {
                        option {
                            value: "{variant.name()}",
                            selected: "{variant.name() == ty.name()}",
                            "{variant.name()}"
                            if variant.model_downloaded_sync() {
                                " (Downloaded)"
                            }
                        }
                    }
                }
            }
        }
        PrimitiveValue::EmbeddingModelType(ty) => {
            rsx! {
                select {
                    class: "border rounded focus:outline-none focus:border-blue-500",
                    style: "-webkit-appearance:none; -moz-appearance:none; -ms-appearance:none; appearance: none;",
                    onchange: move |e| {
                        set_value(
                            PrimitiveValue::EmbeddingModelType(
                                embedding_model_type_from_str(&e.value())
                                    .unwrap_or(EmbeddingModelType::Bert),
                            ),
                        );
                    },
                    for variant in EmbeddingModelType::VARIANTS {
                        option {
                            value: "{variant.name()}",
                            selected: "{variant.name() == ty.name()}",
                            "{variant.name()}"
                            if variant.model_downloaded_sync() {
                                " (Downloaded)"
                            }
                        }
                    }
                }
            }
        }
        PrimitiveValue::Boolean(val) => {
            rsx! {
                input {
                    class: "border rounded focus:outline-none focus:border-blue-500",
                    r#type: "checkbox",
                    checked: "{val}",
                    onchange: move |e| {
                        set_value(PrimitiveValue::Boolean(e.value() == "on"));
                    }
                }
            }
        }
    }
}
