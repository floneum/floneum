#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(non_snake_case)]

use anyhow::Result;
use dioxus::{html::geometry::euclid::Point2D, prelude::*};
use dioxus_desktop::{tao::window::Icon, WindowBuilder};
use dioxus_signals::*;
use floneum_plugin::Plugin;
use floneumite::FloneumPackageIndex;
use petgraph::stable_graph::{DefaultIx, NodeIndex};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File, io::Read, io::Write, rc::Rc};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

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
use current_node::CurrentNodeInfo;
mod input;
mod node_value;
mod output;

const SAVE_NAME: &str = "workflow.toml";

pub type Point = Point2D<f32, f32>;

fn save_to_file<D: Serialize>(data: &D) {
    let mut current_dir = std::env::current_dir().unwrap();
    current_dir.push(SAVE_NAME);
    match File::create(current_dir) {
        Ok(mut file) => {
            log::info!("serializing");
            match toml::to_string(data) {
                Ok(bytes) => {
                    let _ = file.write_all(bytes.as_bytes());
                }
                Err(err) => {
                    log::error!("{}", err);
                }
            }
        }
        Err(err) => {
            log::error!("{}", err);
        }
    }
}

#[tokio::main]
async fn main() {
    use tracing_subscriber::EnvFilter;

    let file = File::create("debug.log").unwrap();
    let debug_log = tracing_subscriber::fmt::layer().with_writer(std::sync::Arc::new(file));

    let logger = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .pretty()
        .finish();

    logger.with(debug_log).init();

    dioxus_desktop::launch_with_props(App, (), make_config());
}

pub struct PluginId(usize);

#[derive(Serialize, Deserialize, Default)]
pub struct ApplicationState {
    graph: VisualGraph,
    #[serde(skip)]
    currently_focused: Option<Signal<Node>>,
    #[serde(skip)]
    plugins: HashMap<String, Plugin>,
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
            if focused.read().id == node {
                self.currently_focused = None;
            }
        }
    }
}

impl PartialEq for ApplicationState {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Drop for ApplicationState {
    fn drop(&mut self) {
        save_to_file(self);
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
                if let Ok(from_storage) = toml::from_str(as_str) {
                    from_storage
                } else {
                    ApplicationState::default()
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

fn App(cx: Scope) -> Element {
    use_package_manager_provider(cx);
    let state = use_provide_application_state(cx);
    let graph = state.read().graph.clone();

    render! {
        FlowView {
            graph: graph,
        }
        Sidebar {}
    }
}

fn use_package_manager_provider(cx: &ScopeState) {
    use_shared_state_provider(cx, || {
        let state: Option<Rc<FloneumPackageIndex>> = None;
        state
    });
    let provider = use_shared_state(cx).unwrap();

    cx.use_hook(|| {
        let registry = provider.clone();
        cx.spawn(async move {
            *registry.write() = Some(Rc::new(FloneumPackageIndex::load().await));
        });
    });
}

pub fn use_package_manager(cx: &ScopeState) -> Option<Rc<FloneumPackageIndex>> {
    use_shared_state::<Option<Rc<FloneumPackageIndex>>>(cx)
        .unwrap()
        .read()
        .clone()
}

fn make_config() -> dioxus_desktop::Config {
    let tailwind = include_str!("../public/tailwind.css");
    dioxus_desktop::Config::default()
        .with_window(WindowBuilder::new().with_title("Floneum"))
        .with_icon(Icon::from_rgba(include_bytes!("../public/Icon.rgba").to_vec(), 64, 64).unwrap())
        .with_custom_head(
            r#"
<style type="text/css">
    html, body {
        height: 100%;
        margin: 0;
        overscroll-behavior-y: none;
        overscroll-behavior-x: none;
        overflow: hidden;
    }
    #main, #bodywrap {
        height: 100%;
        margin: 0;
        overscroll-behavior-x: none;
        overscroll-behavior-y: none;
    }
</style>
<style type="text/css">
"#
            .to_owned()
                + tailwind
                + "</style>",
        )
}
