mod host;
// mod plugin;
// mod proxies;
mod embedding_db;
mod llm;
mod node;
mod page;
mod structure;

wasmtime::component::bindgen!({
    path: "../wit",
    async: true,
    world: "exports",
});
