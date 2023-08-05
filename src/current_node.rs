use crate::use_application_state;
use dioxus::prelude::*;
use crate::LocalSubscription;
use crate::Node;

pub fn CurrentNodeInfo(cx: Scope) -> Element {
    let focused = &use_application_state(cx)
        .read_silent()
        .currently_focused;

    match focused {
        Some(node) => {
            render! {
                CurrentNodeInfoInner {
                    node: node.clone(),
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

#[inline_props]
pub fn CurrentNodeInfoInner(cx: Scope, node: LocalSubscription<Node>) -> Element {
    let node = node.use_(cx).read();
    let name = &node.instance.metadata().name;

    render! {
        "{name}"
    }
}
