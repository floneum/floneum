#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(non_snake_case)]

use anyhow::Result;
use dioxus::{html::geometry::euclid::Point2D, prelude::*};
use floneum_plugin::{Plugin, ResourceStorage};
use floneumite::FloneumPackageIndex;
use futures_util::stream::StreamExt;
use petgraph::stable_graph::{DefaultIx, NodeIndex};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File, io::Read, rc::Rc};

mod node;
pub use node::Node;
mod edge;
pub use edge::Edge;
mod graph;
pub use graph::{CurrentlyDraggingProps, DraggingIndex, FlowView, VisualGraph, VisualGraphInner};
mod connection;
pub use connection::Connection;
mod plugin_search;
mod sidebar;
use sidebar::Sidebar;
mod current_node;
use current_node::{CurrentNodeInfo, FocusedNodeInfo};
mod node_value;
// mod share;
mod theme;
use crate::window::{make_config, use_apply_menu_event};
pub use node_value::*;
mod input;
mod output;
mod window;

const SAVE_NAME: &str = "workflow.json";

pub type Point = Point2D<f32, f32>;

fn main() {
    use tracing_subscriber::filter::LevelFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::EnvFilter;

    #[cfg(any(
        target_os = "linux",
        target_os = "dragonfly",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
    ))]
    {
        use gtk::prelude::DisplayExtManual;

        gtk::init().unwrap();
        if gtk::gdk::Display::default().unwrap().backend().is_wayland() {
            panic!("This example doesn't support wayland!");
        }

        // we need to ignore this error here otherwise it will be catched by winit and will be
        // make the example crash
        winit::platform::x11::register_xlib_error_hook(Box::new(|_display, error| {
            let error = error as *mut x11_dl::xlib::XErrorEvent;
            (unsafe { (*error).error_code }) == 170
        }));
    }

    let log_path = directories::ProjectDirs::from("com", "floneum", "floneum")
        .unwrap()
        .data_dir()
        .join("debug.log");
    std::fs::create_dir_all(log_path.parent().unwrap()).unwrap();
    let file = File::create(log_path).unwrap();
    let debug_log = tracing_subscriber::fmt::layer().with_writer(std::sync::Arc::new(file));

    let logger = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::ERROR.into())
                .from_env_lossy(),
        )
        .pretty()
        .finish();

    logger.with(debug_log).init();

    let config = match make_config() {
        Ok(config) => config,
        Err(err) => {
            eprintln!("Failed to make config: {:?}", err);
            return;
        }
    };

    dioxus::prelude::LaunchBuilder::new()
        .with_cfg(config)
        .launch(App);
}

#[derive(Default)]
pub struct ApplicationState {
    graph: VisualGraph,
    currently_focused: Option<FocusedNodeInfo>,
    resource_storage: ResourceStorage,
    plugins: HashMap<String, Plugin>,
    // last_save_id: Option<share::StorageId<ApplicationState>>,
}

impl ApplicationState {
    async fn insert_plugin(&mut self, name: &str) -> Result<()> {
        match self.get_plugin(name) {
            Some(plugin) => {
                let instance = plugin.instance().await?;
                self.graph.create_node(instance)?;
                Ok(())
            }
            None => Err(anyhow::anyhow!("Plugin not found")),
        }
    }

    async fn add_plugin(&mut self, plugin: Plugin) -> Result<()> {
        let name = plugin.name().await?;
        self.plugins.insert(name.clone(), plugin);

        Ok(())
    }

    fn get_plugin(&self, name: &str) -> Option<&Plugin> {
        self.plugins.get(name)
    }

    fn remove(&mut self, node: NodeIndex<DefaultIx>) {
        self.graph.inner.write().graph.remove_node(node);
        if let Some(focused) = &self.currently_focused {
            if focused.node.read().id == node {
                self.currently_focused = None;
            }
        }
    }

    pub(crate) fn clear(&mut self) {
        self.graph.clear();
        self.currently_focused = None;
        self.resource_storage.clear();
    }
}

impl PartialEq for ApplicationState {
    fn eq(&self, other: &Self) -> bool {
        self.graph == other.graph
    }
}

pub fn use_provide_application_state() -> Signal<ApplicationState> {
    use_context_provider(|| {
        let mut current_dir = std::env::current_dir().unwrap();
        current_dir.push(SAVE_NAME);
        // let state = if let Ok(mut file) = File::open(current_dir) {
        //     let mut buffer = Vec::new();

        //     if file.read_to_end(&mut buffer).is_err() {
        //         ApplicationState::default()
        //     } else {
        //         let as_str = std::str::from_utf8(&buffer).unwrap();
        //         match serde_json::from_str(as_str) {
        //             Ok(from_storage) => from_storage,
        //             Err(err) => {
        //                 tracing::error!("Failed to deserialize state: {}", err);
        //                 eprintln!("Failed to deserialize state: {}", err);
        //                 ApplicationState::default()
        //             }
        //         }
        //     }
        // } else {
        //     ApplicationState::default()
        // };
        Signal::new(ApplicationState::default())
    })
}

pub fn use_application_state() -> Signal<ApplicationState> {
    use_context::<Signal<ApplicationState>>()
}

pub fn application_state() -> Signal<ApplicationState> {
    consume_context()
}

// struct DeserializeApplicationState {
//     new_state: StorageId<ApplicationState>,
// }

fn App() -> Element {
    use_package_manager_provider();
    let mut package_manager = use_context::<Signal<Option<Rc<FloneumPackageIndex>>>>();
    let mut state = use_provide_application_state();
    use_apply_menu_event(state);
    use_hook(|| {
        spawn(async move {
            let new_package_manager =
                tokio::spawn(async move { FloneumPackageIndex::load().await })
                    .await
                    .unwrap();
            package_manager.set(Some(Rc::new(new_package_manager)));
        });
    });
    // use_coroutine(|mut channel| async move {
    //     while let Some(DeserializeApplicationState { new_state }) = channel.next().await {
    //         let mut application = state.write();
    //         *application = new_state.load().await.unwrap();
    //         application.last_save_id = Some(new_state);
    //     }
    // });
    let graph = state.read().graph;

    rsx! {
        FlowView { graph }
        Sidebar {}
    }
}

fn use_package_manager_provider() {
    use_context_provider(|| {
        let state: Option<Rc<FloneumPackageIndex>> = None;
        Signal::new(state)
    });
}

pub fn use_package_manager() -> Option<Rc<FloneumPackageIndex>> {
    use_context::<Signal<Option<Rc<FloneumPackageIndex>>>>().cloned()
}
