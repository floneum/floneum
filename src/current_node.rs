use crate::ShowOutput;
use crate::{use_application_state, ModifyInput};
use dioxus::prelude::*;

pub fn CurrentNodeInfo(cx: Scope) -> Element {
    let focused = use_application_state(cx).read().currently_focused;

    match focused {
        Some(node) => {
            let node = node.read();
            let md = node.instance.metadata();
            let name = &md.name;
            let description = &md.description;

            render! {
                div {
                    class: "p-4",
                    h1 {
                        class: "text-2xl font-bold",
                        "{name}"
                    }

                    // Inputs
                    div {
                        class: "text-left bg-slate-200 rounded-md m-2 p-2",
                        h2 {
                            class: "text-xl font-bold",
                            "inputs:"
                        }
                        for input in &node.inputs {
                            ModifyInput {
                                key: "{input.read().definition.name}",
                                value: input.clone()
                            }
                        }
                    }

                    // Outputs
                    div {
                        class: "text-left bg-slate-200 rounded-md m-2 p-2",
                        h2 {
                            class: "text-xl font-bold",
                            "outputs:"
                        }
                        for output in &node.outputs {
                            ShowOutput {
                                key: "{output.read().definition.name}",
                                value: output.clone()
                            }
                        }
                    }

                    // Info
                    div {
                        class: "text-left",
                        "{description}"
                    }
                }
            }
        }
        None => {
            render! {
                "Select a node to see its info"
            }
        }
    }
}
