use crate::embedding_db::VectorDBWithDocuments;
use crate::plugins::main::imports::{self};
use crate::plugins::main::{self};
use crate::Both;

use headless_chrome::Tab;
use kalosm::language::DynamicNodeId;
use kalosm::language::*;
use once_cell::sync::Lazy;

use reqwest::header::{HeaderName, HeaderValue};
use slab::Slab;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};
use wasi_common::sync::Dir;
use wasmtime::component::__internal::async_trait;
use wasmtime::component::{Linker, ResourceTable};
use wasmtime::Config;
use wasmtime::Engine;
use wasmtime_wasi::WasiCtxBuilder;
use wasmtime_wasi::{self, command, DirPerms, FilePerms, WasiCtx, WasiView};

pub(crate) static LINKER: Lazy<Linker<State>> = Lazy::new(|| {
    let mut linker = Linker::new(&ENGINE);
    let l = &mut linker;
    Both::add_to_linker(l, |x| x).unwrap();
    command::add_to_linker(l).unwrap();

    linker
});
pub(crate) static ENGINE: Lazy<Engine> = Lazy::new(|| {
    let mut config = Config::new();
    config.wasm_component_model(true).async_support(true);
    Engine::new(&config).unwrap()
});

#[derive(Clone, Copy)]
pub(crate) struct AnyNodeRef {
    pub(crate) node_id: DynamicNodeId,
    pub(crate) page_id: usize,
}

pub struct State {
    pub(crate) logs: Arc<RwLock<Vec<String>>>,
    pub(crate) models: Slab<DynModel>,
    pub(crate) embedders: Slab<DynEmbedder>,
    pub(crate) embedding_dbs: Slab<VectorDBWithDocuments>,
    pub(crate) nodes: Slab<AnyNodeRef>,
    pub(crate) pages: Slab<Arc<Tab>>,
    pub(crate) plugin_state: HashMap<Vec<u8>, Vec<u8>>,
    pub(crate) table: ResourceTable,
    pub(crate) ctx: WasiCtx,
}

impl Default for State {
    fn default() -> Self {
        let sandbox = Path::new("./sandbox");
        std::fs::create_dir_all(sandbox).unwrap();
        let mut ctx = WasiCtxBuilder::new();
        let ctx_builder = ctx
            .inherit_stderr()
            .inherit_stdin()
            .inherit_stdio()
            .inherit_stdout()
            .preopened_dir(
                Dir::open_ambient_dir(sandbox, wasi_common::sync::ambient_authority()).unwrap(),
                DirPerms::all(),
                FilePerms::all(),
                ".",
            );
        let table = ResourceTable::new();
        let ctx = ctx_builder.build();
        State {
            plugin_state: Default::default(),
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
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.table
    }

    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.ctx
    }
}

#[async_trait]
impl main::types::Host for State {
    async fn get_request(
        &mut self,
        url: String,
        headers: Vec<main::types::Header>,
    ) -> std::result::Result<String, wasmtime::Error> {
        let mut headers = headers
            .into_iter()
            .map(|header| {
                Ok((
                    HeaderName::try_from(header.key)?,
                    HeaderValue::from_str(&header.value)?,
                ))
            })
            .collect::<wasmtime::Result<Vec<_>>>()?;
        headers.push((
            HeaderName::from_static("user-agent"),
            HeaderValue::from_static("floneum"),
        ));
        let res = reqwest::Client::new()
            .get(&url)
            .headers(reqwest::header::HeaderMap::from_iter(headers))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        Ok(res)
    }
}

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
