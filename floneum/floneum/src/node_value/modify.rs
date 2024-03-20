use crate::node_value::embedding_model_type_from_str;
use crate::node_value::model_type_from_str;
use crate::show_primitive_value;
use dioxus::prelude::*;
use floneum_plugin::plugins::main::types::*;
use std::path::PathBuf;
use crate::node_value::Named;
use crate::{node_value::NodeInput, Signal};
use crate::node_value::Variants;
use std::rc::Rc;

#[component]
pub fn ModifyInput(value: Signal<NodeInput>) -> Element {
    let mut node = value;
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
struct ModifySingleValueProps{
    value: BorrowedPrimitiveValue,
    set_value: Rc<dyn Fn(BorrowedPrimitiveValue)>,
}

impl PartialEq for ModifySingleValueProps {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

fn ModifySingleValue(
    props: ModifySingleValueProps
) -> Element {
    let ModifySingleValueProps { value, set_value } = props;
    match value {
        BorrowedPrimitiveValue::Text(value) => {
            rsx! {
                textarea {
                    class: "border rounded focus:outline-none focus:border-blue-500",
                    value: "{value}",
                    oninput: move |e| {
                        set_value(BorrowedPrimitiveValue::Text(e.value()));
                    }
                }
            }
        }
        BorrowedPrimitiveValue::File(file) => {
            rsx! {
                button {
                    class: "border rounded focus:outline-none focus:border-blue-500",
                    onclick: move |_| {
                        set_value(rfd::FileDialog::new()
                            .set_directory("./sandbox")
                            .set_file_name("Floneum")
                            .set_title("Select File")
                            .save_file()
                            .map(|path| BorrowedPrimitiveValue::File(path.strip_prefix(PathBuf::from("./sandbox").canonicalize().unwrap()).unwrap_or(&path).to_string_lossy().to_string()))
                            .unwrap_or_else(|| BorrowedPrimitiveValue::File("".to_string())));
                    },
                    "Select File"
                }
                "{file}"
            }
        }
        BorrowedPrimitiveValue::Folder(folder) => {
            rsx! {
                button {
                    class: "border rounded focus:outline-none focus:border-blue-500",
                    onclick: move |_| {
                        set_value(rfd::FileDialog::new()
                            .set_directory("./sandbox")
                            .set_file_name("Floneum")
                            .set_title("Select Folder")
                            .pick_folder()
                            .map(|path| BorrowedPrimitiveValue::File(path.strip_prefix(PathBuf::from("./sandbox").canonicalize().unwrap()).unwrap_or(&path).to_string_lossy().to_string()
                        ))
                            .unwrap_or_else(|| BorrowedPrimitiveValue::File("".to_string())))
                    },
                    "Select Folder"
                }
                "{folder}"
            }
        }
        BorrowedPrimitiveValue::Embedding(_)
        | BorrowedPrimitiveValue::Model(_)
        | BorrowedPrimitiveValue::EmbeddingModel(_)
        | BorrowedPrimitiveValue::Database(_)
        | BorrowedPrimitiveValue::Page(_)
        | BorrowedPrimitiveValue::Node(_) => show_primitive_value(&value),
        BorrowedPrimitiveValue::Number(value) => {
            rsx! {
                input {
                    class: "border rounded focus:outline-none focus:border-blue-500",
                    r#type: "number",
                    value: "{value}",
                    oninput: move |e| {
                        set_value(BorrowedPrimitiveValue::Number(e.value().parse().unwrap_or(0)));
                    }
                }
            }
        }
        BorrowedPrimitiveValue::ModelType(ty) => {
            rsx! {
                select {
                    class: "border rounded focus:outline-none focus:border-blue-500",
                    style: "-webkit-appearance:none; -moz-appearance:none; -ms-appearance:none; appearance: none;",
                    onchange: move |e| {
                        set_value(
                            BorrowedPrimitiveValue::ModelType(
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
        BorrowedPrimitiveValue::EmbeddingModelType(ty) => {
            rsx! {
                select {
                    class: "border rounded focus:outline-none focus:border-blue-500",
                    style: "-webkit-appearance:none; -moz-appearance:none; -ms-appearance:none; appearance: none;",
                    onchange: move |e| {
                        set_value(
                            BorrowedPrimitiveValue::EmbeddingModelType(
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
        BorrowedPrimitiveValue::Boolean(val) => {
            rsx! {
                input {
                    class: "border rounded focus:outline-none focus:border-blue-500",
                    r#type: "checkbox",
                    checked: "{val}",
                    onchange: move |e| {
                        set_value(BorrowedPrimitiveValue::Boolean(e.value() == "on"));
                    }
                }
            }
        }
    }
}
