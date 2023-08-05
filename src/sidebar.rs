use crate::plugin_search::PluginSearch;
use dioxus::prelude::*;

pub fn Sidebar(cx: Scope) -> Element {
    render! {
        div {
            class: "h-full w-64 bg-gray-800 top-0 bottom-0 right-0 z-10 fixed",
            div {
                class: "flex items-center justify-center mt-10",
                PluginSearch {}
            }
        }
    }
}
