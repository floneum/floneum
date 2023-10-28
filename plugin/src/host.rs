use crate::plugins::main;
use crate::plugins::main::imports::{self};
use crate::plugins::main::types::{EitherStructure, NumberParameters, ThenStructure};
use crate::Both;

use kalosm::language::Document;

use headless_chrome::Tab;
use kalosm::language::DynamicNodeId;
use kalosm::language::VectorDB;
use kalosm::language::*;
use once_cell::sync::Lazy;

use slab::Slab;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};
use wasmtime::component::Linker;
use wasmtime::component::__internal::async_trait;
use wasmtime::Config;
use wasmtime::Engine;
use wasmtime_wasi::preview2::{self, DirPerms, FilePerms, WasiView};
use wasmtime_wasi::Dir;

pub(crate) static LINKER: Lazy<Linker<State>> = Lazy::new(|| {
    let mut linker = Linker::new(&ENGINE);
    let l = &mut linker;
    Both::add_to_linker(l, |x| x).unwrap();
    preview2::command::add_to_linker(l).unwrap();
    // preview2::bindings::filesystem::types::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::filesystem::preopens::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::io::streams::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::environment::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::exit::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::stdin::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::stdout::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::stderr::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::terminal_input::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::terminal_output::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::terminal_stdin::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::terminal_stdout::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::cli::terminal_stderr::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::clocks::monotonic_clock::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::clocks::timezone::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::clocks::wall_clock::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::random::insecure::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::random::insecure_seed::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::random::random::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::sockets::network::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::sockets::instance_network::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::sockets::tcp::add_to_linker(&mut linker, |x| x).unwrap();
    // preview2::bindings::sockets::tcp_create_socket::add_to_linker(&mut linker, |x| x).unwrap();

    linker
});
pub(crate) static ENGINE: Lazy<Engine> = Lazy::new(|| {
    let mut config = Config::new();
    config.wasm_component_model(true).async_support(true);
    Engine::new(&config).unwrap()
});

pub(crate) enum StructureType {
    Num(NumberParameters),
    Literal(String),
    Or(EitherStructure),
    Then(ThenStructure),
}

#[derive(Clone, Copy)]
pub(crate) struct AnyNodeRef {
    pub(crate) node_id: DynamicNodeId,
    pub(crate) page_id: usize,
}

pub struct State {
    pub(crate) logs: Arc<RwLock<Vec<String>>>,
    pub(crate) structures: Slab<StructureType>,
    pub(crate) models: Slab<DynModel>,
    pub(crate) embedders: Slab<DynEmbedder>,
    pub(crate) embedding_dbs: Slab<VectorDB<Document>>,
    pub(crate) nodes: Slab<AnyNodeRef>,
    pub(crate) pages: Slab<Arc<Tab>>,
    pub(crate) plugin_state: HashMap<Vec<u8>, Vec<u8>>,
    pub(crate) table: preview2::Table,
    pub(crate) ctx: preview2::WasiCtx,
}
impl Default for State {
    fn default() -> Self {
        let sandbox = Path::new("./sandbox");
        std::fs::create_dir_all(sandbox).unwrap();
        let mut ctx = preview2::WasiCtxBuilder::new();
        let ctx_builder = ctx
            .inherit_stderr()
            .inherit_stdin()
            .inherit_stdio()
            .inherit_stdout()
            .preopened_dir(
                Dir::open_ambient_dir(sandbox, wasmtime_wasi::sync::ambient_authority()).unwrap(),
                DirPerms::all(),
                FilePerms::all(),
                ".",
            );
        let table = preview2::Table::new();
        let ctx = ctx_builder.build();
        State {
            plugin_state: Default::default(),
            structures: Default::default(),
            embedders: Default::default(),
            models: Default::default(),
            embedding_dbs: Default::default(),
            nodes: Default::default(),
            pages: Default::default(),
            logs: Default::default(),
            table,
            ctx,
        }
    }
}

impl WasiView for State {
    fn table(&self) -> &preview2::Table {
        &self.table
    }

    fn table_mut(&mut self) -> &mut preview2::Table {
        &mut self.table
    }

    fn ctx(&self) -> &preview2::WasiCtx {
        &self.ctx
    }

    fn ctx_mut(&mut self) -> &mut preview2::WasiCtx {
        &mut self.ctx
    }
}

impl main::types::Host for State {}

#[async_trait]
impl imports::Host for State {
    async fn log_to_user(&mut self, message: String) -> std::result::Result<(), wasmtime::Error> {
        let mut logs = self
            .logs
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock logs: {}", e)))?;
        if logs.len() >= 100 {
            logs.remove(0);
        }
        logs.push(message);
        Ok(())
    }

    async fn store(
        &mut self,
        key: Vec<u8>,
        value: Vec<u8>,
    ) -> std::result::Result<(), wasmtime::Error> {
        self.plugin_state.insert(key, value);

        Ok(())
    }

    async fn load(&mut self, key: Vec<u8>) -> std::result::Result<Vec<u8>, wasmtime::Error> {
        Ok(self.plugin_state.get(&key).cloned().unwrap_or_default())
    }

    async fn unload(&mut self, key: Vec<u8>) -> std::result::Result<(), wasmtime::Error> {
        self.plugin_state.remove(&key);
        Ok(())
    }
}
