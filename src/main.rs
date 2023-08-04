#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(non_snake_case)]

use dioxus::{html::geometry::euclid::Point2D, prelude::*};
use floneumite::FloneumPackageIndex;
use serde::{Deserialize, Serialize};
use std::{fs::File, io::Read, io::Write, rc::Rc};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

mod node;
pub use node::Node;
mod local_sub;
pub use local_sub::{LocalSubscription, UseLocalSubscription};
mod edge;
pub use edge::Edge;
mod graph;
pub use graph::{CurrentlyDraggingProps, DraggingIndex, FlowView, VisualGraph, VisualGraphInner};
mod connection;
pub use connection::Connection;
mod help;
pub use help::Help;
mod value;
pub use value::*;
mod plugin_search;

pub type Point = Point2D<f32, f32>;

fn save_to_file<D: Serialize>(data: &D) {
    let mut current_dir = std::env::current_dir().unwrap();
    current_dir.push("save.bin");
    match File::create(current_dir) {
        Ok(mut file) => {
            log::info!("serializing");
            match bincode::serialize(data) {
                Ok(bytes) => {
                    let compressed =
                        yazi::compress(&bytes, yazi::Format::Zlib, yazi::CompressionLevel::Default)
                            .unwrap();
                    let _ = file.write_all(&compressed);
                }
                Err(err) => {
                    log::error!("{}", err)
                }
            }
        }
        Err(err) => {
            log::error!("{}", err)
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

    let mut current_dir = std::env::current_dir().unwrap();
    current_dir.push("save.bin");
    let state: ApplicationState = if let Ok(mut file) = File::open(current_dir) {
        let mut buffer = Vec::new();

        if file.read_to_end(&mut buffer).is_err() {
            ApplicationState::default()
        } else {
            let (uncompressed, _) = yazi::decompress(&buffer[..], yazi::Format::Zlib).unwrap();

            if let Ok(from_storage) = bincode::deserialize(&uncompressed[..]) {
                from_storage
            } else {
                ApplicationState::default()
            }
        }
    } else {
        ApplicationState::default()
    };

    dioxus_desktop::launch_with_props(App, AppProps { state }, Default::default());
}

#[derive(Serialize, Deserialize, Default, PartialEq)]
struct ApplicationState {
    graph: VisualGraph,
}

impl Drop for ApplicationState {
    fn drop(&mut self) {
        save_to_file(self);
    }
}

#[inline_props]
fn App(cx: Scope, state: ApplicationState) -> Element {
    use_package_manager_provider(cx);

    render! {
        FlowView {
            graph: state.graph.clone(),
        }
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
            *registry.write() = Rc::new(FloneumPackageIndex::load().await);
        });
    });
}

pub fn use_package_manager(cx: &ScopeState) -> Option<Rc<FloneumPackageIndex>> {
    use_shared_state::<Option<Rc<FloneumPackageIndex>>>(cx)
        .unwrap()
        .read()
        .clone()
}
