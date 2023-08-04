use dioxus::prelude::*;

use crate::use_package_manager;

const BUILT_IN_PLUGINS: &[&str] = &[
    "add_embedding",
    "embedding",
    "embedding_db",
    "format",
    "generate_text",
    "generate_structured_text",
    "search",
    "search_engine",
    "if_statement",
    "contains",
    "write_to_file",
    "read_from_file",
    "run_python",
    "create_browser",
    "find_node",
    "find_child_node",
    "click_node",
    "node_text",
    "type_in_node",
    "navigate_to",
];

fn PluginSearch(cx: Scope) -> Element {
    let plugins = use_package_manager(cx);

    render! {
        LoadLocalPlugin {}
    }        
}

fn LoadLocalPlugin(cx: Scope) -> Element {
    let plugins = use_package_manager(cx);
    let search_text = use_state(cx, String::new);

    render! {
        div {
            class: "flex flex-col",
            "Load Plugin from File: "
            input {
                r#type: "file",
                oninput: move |event| {
                    search_text.set(event.value.clone());
                },
            }

            button {
                onclick: move |_| {
                    let plugins = plugins.clone();
                    let search_text = search_text.clone();
                    cx.spawn(async move {
                        
                    });
                },
                "Load"
            }
        }
    }        
}