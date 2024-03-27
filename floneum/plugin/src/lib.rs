mod host;
mod plugin;
pub use plugin::*;
mod embedding;
mod embedding_db;
mod llm;
mod node;
mod page;
mod proxies;
mod resource;

pub use embedding::listen_to_embedding_model_download_progresses;
pub use llm::listen_to_model_download_progresses;

wasmtime::component::bindgen!({
    path: "../wit",
    async: true,
    world: "both",
});
