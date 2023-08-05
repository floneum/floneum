use std::path::PathBuf;

use dioxus::prelude::*;
use floneum_plugin::{load_plugin, load_plugin_from_source};

use crate::{use_application_state, use_package_manager};

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

pub fn PluginSearch(cx: Scope) -> Element {
    render! {
        LoadRegisteredPlugin {}
        LoadLocalPlugin {}
    }
}

fn LoadRegisteredPlugin(cx: Scope) -> Element {
    let plugins = use_package_manager(cx);
    let application = use_application_state(cx);

    render! {
        div {
            class: "flex flex-col",
            "Add Plugin:"
            match &plugins {
                Some(plugins) => {
                    rsx! {
                        for entry in plugins.entries() {
                            button {
                                onclick: {
                                    let entry = entry.clone();
                                    move |_| {
                                        let plugin = load_plugin_from_source(entry.clone());
                                        to_owned![application];
                                        async move {
                                            let mut application = application.write();
                                            let name = plugin.name().await.unwrap();
                                            if application.get_plugin(&name).is_none() {
                                                let _ = application.add_plugin(plugin).await;
                                            }

                                            application.insert_plugin(&name).await.unwrap();
                                        }
                                    }
                                },
                                if let Some(meta) = entry.meta() {
                                    rsx! {
                                        "{meta.name}"
                                    }
                                } else {
                                    rsx! {
                                        "{entry.path().display()}"
                                    }
                                }
                            }
                        }
                    }
                }
                None => {
                    rsx! {
                        "Loading..."
                    }
                }
            }
        }
    }
}

fn LoadLocalPlugin(cx: Scope) -> Element {
    let search_text = use_state(cx, String::new);
    let application = use_application_state(cx);

    render! {
        div {
            class: "flex flex-col",
            "Add Plugin from File: "
            input {
                value: "{search_text}",
                oninput: move |event| {
                    search_text.set(event.value.clone());
                },
            }

            button {
                onclick: move |_| {
                    let path = PathBuf::from(search_text.get());
                    let plugin = load_plugin(&path);
                    to_owned![application];
                    async move {
                        let mut application = application.write();
                        let name = plugin.name().await.unwrap();
                        if application.get_plugin(&name).is_none() {
                            let _ = application.add_plugin(plugin).await;
                        }

                        application.insert_plugin(&name).await.unwrap();
                    }
                },
                "Load"
            }
        }
    }
}
