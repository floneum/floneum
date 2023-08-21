use std::path::PathBuf;

use dioxus::prelude::*;
use floneum_plugin::{load_plugin, load_plugin_from_source};

use crate::{use_application_state, use_package_manager};

const BUILT_IN_PLUGINS: &[&str] = &[
    "Add Embedding",
    "Embedding",
    "Embedding Db",
    "Format",
    "Generate Text",
    "Generate Structured Text",
    "Search",
    "Search Engine",
    "If Statement",
    "Contains",
    "Write To File",
    "Read From File",
    "Run Python",
    "Create Browser",
    "Find Node",
    "Find Child Node",
    "Click Node",
    "Node Text",
    "Type In Node",
    "Navigate To",
];

pub fn PluginSearch(cx: Scope) -> Element {
    render! {
        div {
            class: "flex flex-col",
            LoadRegisteredPlugin {}
            LoadLocalPlugin {}
        }
    }
}

fn LoadRegisteredPlugin(cx: Scope) -> Element {
    let plugins = use_package_manager(cx);
    let application = use_application_state(cx);
    let search_text = use_state(cx, || "".to_string());
    let text_words: Vec<&str> = search_text.split_whitespace().collect();

    render! {
        div {
            class: "flex flex-col",
            "Add Plugin"
            input {
                class: "border border-gray-400 rounded-md p-2 m-2",
                r#type: "text",
                oninput: {
                    let search_text = search_text.clone();
                    move |event| {
                        search_text.set(event.value.clone());
                    }
                },
            }
            match &plugins {
                Some(plugins) => {
                    rsx! {
                        for entry in plugins.entries().iter().filter(|entry| {
                            if let Some(meta) = entry.meta(){
                                text_words.iter().all(|word| meta.name.contains(word.trim()))
                            }
                            else {
                                false
                            }
                        }) {
                            button {
                                class: "hover:bg-gray-200 border border-gray-400 rounded-md p-2 m-2",
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
                                    let name = &meta.name;
                                    let built_in = BUILT_IN_PLUGINS.contains(&name.as_str());
                                    let extra = if built_in {
                                        " (built-in)"
                                    } else {
                                        ""
                                    };
                                    rsx! {
                                        "{name}{extra}"
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
            class: "flex flex-col items-left",
            "Add Plugin from File"
            input {
                class: "border border-gray-400 rounded-md p-2 m-2",
                value: "{search_text}",
                oninput: move |event| {
                    search_text.set(event.value.clone());
                },
            }

            button {
                class: "hover:bg-gray-200 border border-gray-400 rounded-md p-2 m-2",
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
