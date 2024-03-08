#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(non_snake_case)]

use crate::theme::Color;
use anyhow::Result;
use dioxus::{html::geometry::euclid::Point2D, prelude::*};
use dioxus_signals::*;
use floneum_plugin::Plugin;
use floneumite::FloneumPackageIndex;
use futures_util::stream::StreamExt;
use petgraph::stable_graph::{DefaultIx, NodeIndex};
use serde::{Deserialize, Serialize};
use share::StorageId;
use std::{collections::HashMap, fs::File, io::Read, rc::Rc};
use tokio::sync::oneshot::Receiver;

mod node;
pub use node::Node;
mod edge;
pub use edge::Edge;
mod graph;
pub use graph::{CurrentlyDraggingProps, DraggingIndex, FlowView, VisualGraph, VisualGraphInner};
mod connection;
pub use connection::Connection;
mod value;
pub use value::*;
mod plugin_search;
mod sidebar;
use sidebar::Sidebar;
mod current_node;
use current_node::{CurrentNodeInfo, FocusedNodeInfo};
mod share;

use crate::window::{make_config, use_apply_menu_event};
mod input;
mod node_value;
mod output;
mod theme;
mod window;

const SAVE_NAME: &str = "workflow.json";

pub type Point = Point2D<f32, f32>;

#[tokio::main]
async fn main() {
    use tracing_subscriber::filter::LevelFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::EnvFilter;

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
                .with_default_directive(LevelFilter::DEBUG.into())
                .from_env_lossy(),
        )
        .pretty()
        .finish();

    logger.with(debug_log).init();

    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        tx.send(FloneumPackageIndex::load().await).unwrap();
    });

    dioxus_desktop::launch_with_props(
        App,
        AppProps {
            channel: RefCell::new(Some(rx)),
        },
        make_config(),
    );
}

pub struct PluginId(usize);

#[derive(Serialize, Deserialize, Default)]
pub struct ApplicationState {
    graph: VisualGraph,
    #[serde(skip)]
    currently_focused: Option<FocusedNodeInfo>,
    #[serde(skip)]
    plugins: HashMap<String, Plugin>,
    #[serde(skip)]
    last_save_id: Option<share::StorageId<ApplicationState>>,
}

impl ApplicationState {
    async fn insert_plugin(&mut self, name: &str) -> Result<()> {
        match self.get_plugin(name) {
            Some(plugin) => {
                let instance = plugin.instance().await?;
                self.graph.create_node(instance);
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
}

impl PartialEq for ApplicationState {
    fn eq(&self, other: &Self) -> bool {
        self.graph == other.graph
    }
}

pub fn use_provide_application_state(cx: &ScopeState) -> Signal<ApplicationState> {
    *use_context_provider(cx, || {
        let mut current_dir = std::env::current_dir().unwrap();
        current_dir.push(SAVE_NAME);
        let state = if let Ok(mut file) = File::open(current_dir) {
            let mut buffer = Vec::new();

            if file.read_to_end(&mut buffer).is_err() {
                ApplicationState::default()
            } else {
                let as_str = std::str::from_utf8(&buffer).unwrap();
                match serde_json::from_str(as_str) {
                    Ok(from_storage) => from_storage,
                    Err(err) => {
                        tracing::error!("Failed to deserialize state: {}", err);
                        eprintln!("Failed to deserialize state: {}", err);
                        ApplicationState::default()
                    }
                }
            }
        } else {
            ApplicationState::default()
        };
        Signal::new(state)
    })
}

pub fn use_application_state(cx: &ScopeState) -> Signal<ApplicationState> {
    *use_context::<Signal<ApplicationState>>(cx).unwrap()
}

pub fn application_state(cx: &ScopeState) -> Signal<ApplicationState> {
    cx.consume_context().unwrap()
}

struct DeserializeApplicationState {
    new_state: StorageId<ApplicationState>,
}

#[derive(Props)]
pub struct AppProps {
    #[props(into)]
    channel: RefCell<Option<Receiver<FloneumPackageIndex>>>,
}

impl PartialEq for AppProps {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

fn App(cx: Scope<AppProps>) -> Element {
    use_package_manager_provider(cx);
    let package_manager = use_shared_state::<Option<Rc<FloneumPackageIndex>>>(cx).unwrap();
    let state = use_provide_application_state(cx);
    cx.use_hook(|| {
        let channel = cx.props.channel.borrow_mut().take().unwrap();
        to_owned![package_manager];
        cx.spawn(async move {
            let new_package_manager = channel.await;
            *package_manager.write() = Some(Rc::new(new_package_manager.unwrap()));
        });
    });
    use_coroutine(cx, |mut channel| async move {
        while let Some(DeserializeApplicationState { new_state }) = channel.next().await {
            let mut application = state.write();
            *application = new_state.load().await.unwrap();
            application.last_save_id = Some(new_state);
        }
    });
    let graph = state.read().graph.clone();
    use_apply_menu_event(cx, state);

    render! {
        FlowView { graph: graph }
        Sidebar {}
    }
}

fn use_package_manager_provider(cx: &ScopeState) {
    use_shared_state_provider(cx, || {
        let state: Option<Rc<FloneumPackageIndex>> = None;
        state
    });
}

pub fn use_package_manager(cx: &ScopeState) -> Option<Rc<FloneumPackageIndex>> {
    use_shared_state::<Option<Rc<FloneumPackageIndex>>>(cx)
        .unwrap()
        .read()
        .clone()
}
