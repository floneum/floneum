use crate::{use_application_state, ModifyInput, Node, ShowInput, ShowOutput};
use dioxus::prelude::*;
use dioxus_signals::Signal;

#[derive(Clone, Copy)]
pub(crate) struct FocusedNodeInfo {
    pub node: Signal<Node>,
    pub active_example_index: Option<usize>,
}

pub fn CurrentNodeInfo() -> Element {
    let mut application = use_application_state();
    let focused = application.read().currently_focused;

    match focused {
        Some(node_info) => {
            let node = node_info.node.read();
            let md = node.instance.metadata();
            let name = &md.name;
            let description = &md.description;

            rsx! {
                div {
                    class: "p-4",
                    h1 {
                        class: "text-2xl font-bold",
                        "{name}"
                    }

                    if let Some(example_index) = node_info.active_example_index {
                        button {
                            class: "text-xl font-bold m-2 rounded-md p-2 border-2 ",
                            onclick: move |_| {
                                if let Some(focused) = &mut application.write().currently_focused {
                                    focused.active_example_index = None;
                                }
                            },
                            "Back to node"
                        }
                        div {
                            class: "text-left rounded-md m-2 p-2",
                            h2 {
                                class: "text-xl font-bold",
                                "inputs:"
                            }
                            for (i, input) in md.examples[example_index].inputs.iter().enumerate() {
                                div {
                                    ShowInput {
                                        key: "{input.read().definition.name}",
                                        ty: md.inputs[i].ty.clone(),
                                        label: md.inputs[i].name.clone(),
                                        value: input.clone()
                                    }
                                }
                            }
                        }
                        div {
                            class: "text-left rounded-md m-2 p-2",
                            h2 {
                                class: "text-xl font-bold",
                                "outputs:"
                            }
                            for (i, output) in md.examples[example_index].outputs.iter().enumerate() {
                                div {
                                    ShowOutput {
                                        key: "{output.read().definition.name}",
                                        ty: md.outputs[i].ty.clone(),
                                        name: md.outputs[i].name.clone(),
                                        value: output.clone()
                                    }
                                }
                            }
                        }
                    }
                    else {
                        // Inputs
                        div {
                            class: "text-left rounded-md m-2 p-2",
                            h2 {
                                class: "text-xl font-bold",
                                "inputs:"
                            }
                            for input in &node.inputs {
                                div {
                                    ModifyInput {
                                        key: "{input.read().definition.name}",
                                        value: *input
                                    }
                                }
                            }
                        }

                        // Outputs
                        div {
                            class: "text-left rounded-md m-2 p-2",
                            h2 {
                                class: "text-xl font-bold",
                                "outputs:"
                            }
                            for output in &node.outputs {
                                div {
                                    ShowOutput {
                                        key: "{output.read().definition.name}",
                                        ty: output.read().definition.ty.clone(),
                                        name: output.read().definition.name.clone(),
                                        value: output.read().value.clone()
                                    }
                                }
                            }
                        }
                    }

                    // Examples
                    for (i, example) in md.examples.iter().enumerate() {
                        button {
                            class: "text-xl font-bold m-2 rounded-md p-2 border-2 ",
                            onclick: move |_| {
                                if let Some(focused) = &mut application.write().currently_focused {
                                    focused.active_example_index = Some(i);
                                }
                            },
                            "{example.name}"
                        }
                    }

                    // Info
                    div {
                        class: "text-left whitespace-pre-line",
                        "{description}"
                    }
                }
            }
        }
        None => {
            rsx! {
                "Select a node to see its info"
            }
        }
    }
}
