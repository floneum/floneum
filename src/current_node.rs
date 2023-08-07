use crate::{use_application_state, ModifyInput};
use dioxus::prelude::*;

pub fn CurrentNodeInfo(cx: Scope) -> Element {
    let focused = &use_application_state(cx).use_(cx).read().currently_focused;

    match focused {
        Some(node) => {
            let node = node.read(cx);
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
                        class: "text-left",
                        for input in &node.inputs {
                            ModifyInput {
                                key: "{input.read(cx).definition.name}",
                                value: input.clone()
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
