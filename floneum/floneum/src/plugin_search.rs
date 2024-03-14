use crate::Color;
use dioxus::prelude::*;
use floneum_plugin::{load_plugin, load_plugin_from_source};
use floneumite::Category;
use floneumite::PackageIndexEntry;
use std::collections::HashMap;
use std::path::PathBuf;

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
    "Get Article",
    "Read Rss Stream",
    "Split",
    "Slice",
    "Join",
    "Add To List",
    "New List",
    "Length",
    "More Than",
    "Less Than",
    "Equals",
    "And",
    "Or",
    "Not",
    "Number",
    "String",
    "Add",
    "Subtract",
    "Multiply",
    "Divide",
    "Power",
];

pub fn PluginSearch() -> Element {
    rsx! {
        LoadRegisteredPlugin {}
        LoadLocalPlugin {}
    }
}

fn LoadRegisteredPlugin() -> Element {
    let plugins = use_package_manager();
    let mut search_text = use_signal(|| "".to_string());
    let current_search_text = search_text();
    let text_words: Vec<&str> = current_search_text.split_whitespace().collect();

    rsx! {
        "Add Plugin"
        input {
            class: "border {Color::outline_color()} {Color::foreground_color()} rounded-md p-2 m-2",
            r#type: "text",
            oninput: 
                move |event| {
                    search_text.set(event.value());
                },
        }
        match &plugins {
            Some(plugins) => {
                let mut categories = HashMap::new();
                for category in Category::ALL {
                    categories.insert(category, Vec::new());
                }
                for entry in plugins.entries() {
                    if let Some(meta) = entry.meta() {
                        if text_words.iter().all(|word| {
                            let lowercase = word.trim().to_lowercase();
                            let title_contains_word = meta.name.to_lowercase().contains(&lowercase);
                            let description_contains_word = meta.description.to_lowercase().contains(&lowercase);
                            title_contains_word || description_contains_word
                        }) {
                            categories.get_mut(&meta.category).unwrap().push(entry.clone());
                        }
                    }
                }
                let mut categories_sorted = categories.into_iter().collect::<Vec<_>>();
                categories_sorted.sort_by(|(a, _), (b, _)| a.cmp(b));
                rsx! {
                    nav { "aria-label": "Directory", class: "h-full overflow-y-auto",
                        div { class: "relative",
                            for (name, category) in categories_sorted {
                                Category {
                                    key: "{name}",
                                    name: "{name}",
                                    plugins: category,
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

#[component]
fn Category(name: String, plugins: Vec<PackageIndexEntry>) -> Element {
    let application = use_application_state();

    rsx! {
        div { class: "sticky top-0 z-10 border-y border-b-gray-200 border-t-gray-100 {Color::foreground_color()} px-3 py-1.5 text-sm font-semibold leading-6",
            h3 { "{name}" }
        }
        ul { role: "list", class: "divide-y {Color::divide_color()} {Color::background_color()}",
            for entry in plugins {
                li { class: "flex gap-x-4 px-3 py-5",
                    button {
                        class: "min-w-0 w-full",
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

                                    if let Err(err) = application.insert_plugin(&name).await {
                                        log::error!("Failed to insert plugin: {}", err);
                                    }
                                }
                            }
                        },
                        p { class: "text-sm font-semibold leading-6",
                            if let Some(meta) = entry.meta() {
                                {
                                    let name = &meta.name;
                                    let built_in = BUILT_IN_PLUGINS.contains(&name.as_str());
                                    let extra = if built_in {
                                        ""
                                    } else {
                                        " (community)"
                                    };
                                    rsx! {
                                        "{name}{extra}"
                                    }
                                }
                            } else {
                                "{entry.path().display()}"
                            }
                        }
                        if let Some(meta) = entry.meta() {
                            p { class: "mt-1 truncate text-xs leading-5",
                                "{meta.description}"
                            }
                        }
                    }
                }
            }
        }
    }
}

fn LoadLocalPlugin() -> Element {
    let mut search_text = use_signal(String::new);
    let application = use_application_state();

    rsx! {
        div {
            class: "flex flex-col items-left",
            "Add Plugin from File"
            input {
                class: "border {Color::outline_color()} {Color::foreground_color()} rounded-md p-2 m-2",
                value: "{search_text}",
                oninput: move |event| {
                    search_text.set(event.value());
                },
            }

            button {
                class: "{Color::foreground_hover()} border {Color::outline_color()} rounded-md p-2 m-2",
                onclick: move |_| {
                    let path = PathBuf::from(search_text());
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
