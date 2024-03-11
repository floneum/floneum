mod host;
mod plugin;
pub use plugin::*;
mod embedding_db;
mod llm;
mod node;
mod page;
mod proxies;

pub use llm::{listen_to_embedding_model_download_progresses, listen_to_model_download_progresses};

wasmtime::component::bindgen!({
    path: "../wit",
    async: true,
    world: "both",
});
