// Ideas:
// https://github.com/1rgs/jsonformer
// https://github.com/bytecodealliance/wasmtime/issues/6074

use crate::plugins::main::imports::*;
use download::model_downloaded;
use exports::plugins::main::definitions::*;
use floneumite::PackageIndexEntry;
use futures_util::Future;
use once_cell::sync::Lazy;
use plugins::main::types::Structure;
use pollster::FutureExt;
use reqwest::header::{HeaderMap, HeaderName};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use slab::Slab;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, LockResult, RwLock, RwLockReadGuard};
use structured_parser::StructureParser;
use tokio::sync::broadcast;
use wasmtime::component::__internal::async_trait;
use wasmtime::component::{Component, Linker};
use wasmtime::Engine;
use wasmtime::Store;
use wasmtime::{Config, Error};
use wasmtime_wasi::preview2::{self, DirPerms, FilePerms, WasiView};
use wasmtime_wasi::Dir;
use wit_component::ComponentEncoder;

mod browse;
mod download;
mod embedding;
mod proxies;
mod sessions;
mod structured;
mod structured_parser;
mod vector_db;

use crate::sessions::InferenceSessions;
use crate::{exports::plugins::main::definitions::ModelType, vector_db::VectorDB};

static LINKER: Lazy<Linker<State>> = Lazy::new(|| {
    let mut linker = Linker::new(&ENGINE);
    let l = &mut linker;
    PluginWorld::add_to_linker(l, |x| x).unwrap();
    preview2::bindings::filesystem::types::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::filesystem::preopens::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::io::streams::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::cli::environment::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::cli::exit::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::cli::stdin::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::cli::stdout::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::cli::stderr::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::cli::terminal_input::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::cli::terminal_output::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::cli::terminal_stdin::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::cli::terminal_stdout::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::cli::terminal_stderr::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::clocks::monotonic_clock::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::clocks::timezone::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::clocks::wall_clock::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::random::insecure::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::random::insecure_seed::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::random::random::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::poll::poll::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::sockets::network::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::sockets::instance_network::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::sockets::tcp::add_to_linker(&mut linker, |x| x).unwrap();
    preview2::bindings::sockets::tcp_create_socket::add_to_linker(&mut linker, |x| x).unwrap();

    linker
});
static ENGINE: Lazy<Engine> = Lazy::new(|| {
    let mut config = Config::new();
    config.wasm_component_model(true).async_support(true);
    Engine::new(&config).unwrap()
});
static MULTI_PLUGIN_STATE: Lazy<RwLock<MultiPluginState>> = Lazy::new(Default::default);

#[derive(Default)]
struct MultiPluginState {
    vector_dbs: Slab<VectorDB<String>>,
    browser: browse::Browser,
}

impl MultiPluginState {
    fn vector_db_get(
        &self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
    ) -> Option<&VectorDB<String>> {
        self.vector_dbs.get(id.id as usize)
    }

    fn vector_db_get_mut(
        &mut self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
    ) -> Option<&mut VectorDB<String>> {
        self.vector_dbs.get_mut(id.id as usize)
    }

    pub fn create_db(
        &mut self,
        embedding: Vec<exports::plugins::main::definitions::Embedding>,
        documents: Vec<String>,
    ) -> exports::plugins::main::definitions::EmbeddingDbId {
        let idx = self.vector_dbs.insert(VectorDB::new(embedding, documents));

        exports::plugins::main::definitions::EmbeddingDbId { id: idx as u32 }
    }

    pub fn remove_embedding_db(&mut self, id: exports::plugins::main::definitions::EmbeddingDbId) {
        self.vector_dbs.remove(id.id as usize);
    }
}

pub struct State {
    sessions: InferenceSessions,
    structures: Slab<Structure>,
    logs: Arc<RwLock<Vec<String>>>,
    plugin_state: HashMap<Vec<u8>, Vec<u8>>,
    table: preview2::Table,
    ctx: preview2::WasiCtx,
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
        let mut table = preview2::Table::new();
        let ctx = ctx_builder.build(&mut table).unwrap();
        State {
            sessions: InferenceSessions::default(),
            structures: Slab::new(),
            plugin_state: Default::default(),
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

impl State {
    fn decode_structure(&self, id: StructureId) -> StructureParser {
        match &self.structures[id.id as usize] {
            Structure::Literal(text) => StructureParser::Literal(text.clone()),
            Structure::Str(range) => StructureParser::String {
                min_len: range.min,
                max_len: range.max,
            },
            Structure::Num(range) => StructureParser::Num {
                min: range.min,
                max: range.max,
                integer: range.integer,
            },
            Structure::Or(params) => StructureParser::Either {
                first: Box::new(self.decode_structure(params.first)),
                second: Box::new(self.decode_structure(params.second)),
            },
            Structure::Then(params) => StructureParser::Then {
                first: Box::new(self.decode_structure(params.first)),
                second: Box::new(self.decode_structure(params.second)),
            },
            Structure::Sequence(params) => StructureParser::Sequence {
                item: Box::new(self.decode_structure(params.item)),
                separator: Box::new(self.decode_structure(params.separator)),
                min_len: params.min_len,
                max_len: params.max_len,
            },
        }
    }

    pub fn get_closest(
        &self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
        embedding: exports::plugins::main::definitions::Embedding,
        n: usize,
    ) -> Option<Vec<String>> {
        MULTI_PLUGIN_STATE
            .read()
            .unwrap()
            .vector_db_get(id)
            .map(|db| db.get_closest(embedding, n))
    }

    pub fn get_within(
        &self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
        embedding: exports::plugins::main::definitions::Embedding,
        distance: f32,
    ) -> Option<Vec<String>> {
        MULTI_PLUGIN_STATE
            .read()
            .unwrap()
            .vector_db_get(id)
            .map(|db| db.get_within(embedding, distance))
    }
}

impl plugins::main::types::Host for State {}

#[async_trait]
impl Host for State {
    async fn get_request(
        &mut self,
        url: String,
        headers: Vec<Header>,
    ) -> std::result::Result<String, wasmtime::Error> {
        let client = reqwest::Client::new();
        let mut header_map = HeaderMap::new();

        header_map.append(
            reqwest::header::USER_AGENT,
            // Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0
            "Floneum/0.1.0 (Unknown; Unknown; Unknown; Unknown) Floneum/0.1.0 Floneum/0.1.0"
                .parse()
                .unwrap(),
        );

        for header in headers {
            header_map.append(HeaderName::try_from(&header.key)?, header.value.parse()?);
        }

        let response = client.get(&url).headers(header_map).send().await?;
        Ok(response.text().await?)
    }

    async fn remove_structure(
        &mut self,
        id: StructureId,
    ) -> std::result::Result<(), wasmtime::Error> {
        self.structures.remove(id.id as usize);
        Ok(())
    }

    async fn create_structure(
        &mut self,
        structure: Structure,
    ) -> std::result::Result<plugins::main::types::StructureId, wasmtime::Error> {
        let id = self.structures.insert(structure);
        Ok(plugins::main::types::StructureId { id: id as u32 })
    }

    async fn model_downloaded(
        &mut self,
        ty: exports::plugins::main::definitions::ModelType,
    ) -> std::result::Result<bool, wasmtime::Error> {
        Ok(model_downloaded(ty))
    }

    async fn load_model(
        &mut self,
        ty: exports::plugins::main::definitions::ModelType,
    ) -> std::result::Result<exports::plugins::main::definitions::ModelId, wasmtime::Error> {
        Ok(self.sessions.create(ty))
    }

    async fn unload_model(
        &mut self,
        id: exports::plugins::main::definitions::ModelId,
    ) -> std::result::Result<(), wasmtime::Error> {
        self.sessions.remove(id);
        Ok(())
    }

    async fn get_embedding(
        &mut self,
        id: exports::plugins::main::definitions::ModelId,
        text: String,
    ) -> std::result::Result<plugins::main::types::Embedding, wasmtime::Error> {
        Ok(self.sessions.get_embedding(id, &text))
    }

    async fn create_embedding_db(
        &mut self,
        embeddings: Vec<plugins::main::types::Embedding>,
        documents: Vec<String>,
    ) -> std::result::Result<exports::plugins::main::definitions::EmbeddingDbId, wasmtime::Error>
    {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .unwrap()
            .create_db(embeddings, documents))
    }

    async fn add_embedding(
        &mut self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
        embedding: plugins::main::types::Embedding,
        document: String,
    ) -> std::result::Result<(), wasmtime::Error> {
        MULTI_PLUGIN_STATE
            .write()
            .unwrap()
            .vector_db_get_mut(id)
            .ok_or(wasmtime::Error::msg("Invalid embedding db id"))?
            .add_embedding(embedding, document);
        Ok(())
    }

    async fn remove_embedding_db(
        &mut self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
    ) -> std::result::Result<(), wasmtime::Error> {
        MULTI_PLUGIN_STATE.write().unwrap().remove_embedding_db(id);
        Ok(())
    }

    async fn find_closest_documents(
        &mut self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
        search: plugins::main::types::Embedding,
        count: u32,
    ) -> std::result::Result<Vec<String>, wasmtime::Error> {
        self.get_closest(id, search, count as usize)
            .ok_or(wasmtime::Error::msg("Invalid embedding db id"))
    }

    async fn find_documents_within(
        &mut self,
        id: exports::plugins::main::definitions::EmbeddingDbId,
        search: plugins::main::types::Embedding,
        distance: f32,
    ) -> std::result::Result<Vec<String>, wasmtime::Error> {
        self.get_within(id, search, distance)
            .ok_or(wasmtime::Error::msg("Invalid embedding db id"))
    }

    async fn infer(
        &mut self,
        id: exports::plugins::main::definitions::ModelId,
        input: String,
        max_tokens: Option<u32>,
        stop_on: Option<String>,
    ) -> std::result::Result<String, wasmtime::Error> {
        Ok(self.sessions.infer(id, input, max_tokens, stop_on))
    }

    async fn infer_structured(
        &mut self,
        id: exports::plugins::main::definitions::ModelId,
        input: String,
        max_tokens: Option<u32>,
        structure: StructureId,
    ) -> std::result::Result<String, wasmtime::Error> {
        let structure = self.decode_structure(structure);
        Ok(self
            .sessions
            .infer_validate(id, input, max_tokens, structure))
    }

    async fn new_tab(
        &mut self,
        headless: bool,
    ) -> std::result::Result<crate::plugins::main::imports::TabId, wasmtime::Error> {
        MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .new_tab(headless)
    }

    async fn remove_tab(
        &mut self,
        tab: crate::plugins::main::imports::TabId,
    ) -> std::result::Result<(), wasmtime::Error> {
        MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .remove_tab(tab);
        Ok(())
    }

    async fn browse_to(
        &mut self,
        tab: crate::plugins::main::imports::TabId,
        url: String,
    ) -> std::result::Result<(), wasmtime::Error> {
        MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .goto(tab, &url)?;
        Ok(())
    }

    async fn find_in_current_page(
        &mut self,
        tab: crate::plugins::main::imports::TabId,
        query: String,
    ) -> std::result::Result<crate::plugins::main::imports::NodeId, wasmtime::Error> {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .find(tab, &query)?)
    }

    async fn get_element_text(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
    ) -> std::result::Result<String, wasmtime::Error> {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .get_text(id)?)
    }

    async fn click_element(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
    ) -> std::result::Result<(), wasmtime::Error> {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .click(id)?)
    }

    async fn type_into_element(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
        keys: String,
    ) -> std::result::Result<(), wasmtime::Error> {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .send_keys(id, &keys)?)
    }

    async fn get_element_outer_html(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
    ) -> std::result::Result<String, wasmtime::Error> {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .outer_html(id)?)
    }

    async fn screenshot_browser(
        &mut self,
        tab: crate::plugins::main::imports::TabId,
    ) -> std::result::Result<Vec<u8>, wasmtime::Error> {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .screenshot(tab)?)
    }

    async fn screenshot_element(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
    ) -> std::result::Result<Vec<u8>, wasmtime::Error> {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .screenshot_of_id(id)?)
    }

    async fn find_child_of_element(
        &mut self,
        id: crate::plugins::main::imports::NodeId,
        query: String,
    ) -> std::result::Result<crate::plugins::main::imports::NodeId, wasmtime::Error> {
        Ok(MULTI_PLUGIN_STATE
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock state: {}", e)))?
            .browser
            .find_child(id, &query)?)
    }

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

#[tracing::instrument]
pub fn load_plugin(path: &Path) -> Plugin {
    log::info!("loading plugin {path:?}");

    let module = PackageIndexEntry::new(path.into(), None, None);
    load_plugin_from_source(module)
}

pub fn load_plugin_from_source(source: PackageIndexEntry) -> Plugin {
    let md = once_cell::sync::OnceCell::new();
    if let Some(metadata) = source.meta() {
        let _ = md.set(PluginMetadata {
            name: metadata.name.clone(),
            description: metadata.description.clone(),
        });
    }

    Plugin {
        source,
        component: once_cell::sync::OnceCell::new(),
        definition: once_cell::sync::OnceCell::new(),
        metadata: md,
    }
}

#[derive(Debug, Clone)]
pub struct PluginMetadata {
    name: String,
    description: String,
}

pub struct Plugin {
    source: PackageIndexEntry,
    component: once_cell::sync::OnceCell<Component>,
    definition: once_cell::sync::OnceCell<Definition>,
    metadata: once_cell::sync::OnceCell<PluginMetadata>,
}

impl std::fmt::Debug for Plugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Plugin")
            .field("metadata", &self.metadata)
            .finish()
    }
}

impl Serialize for Plugin {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Just serialize the source
        self.source.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Plugin {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Just deserialize the source
        let source = PackageIndexEntry::deserialize(deserializer)?;

        Ok(async move { load_plugin_from_source(source) }.block_on())
    }
}

impl Plugin {
    async fn component(&self) -> anyhow::Result<&Component> {
        if let Some(component) = self.component.get() {
            return Ok(component);
        }
        let bytes = self.source.wasm_bytes().await?;
        let size = bytes.len();
        log::info!("read plugin ({:01} mb)", size as f64 / (1024. * 1024.));
        // then we transform module to component.
        // remember to get wasi_snapshot_preview1.wasm first.
        let component = ComponentEncoder::default()
            .module(bytes.as_slice())?
            .validate(true)
            .adapter(
                "wasi_snapshot_preview1",
                include_bytes!("../wasi_snapshot_preview1.wasm",),
            )
            .unwrap()
            .encode()?;
        let component = Component::from_binary(&ENGINE, &component)?;

        let _ = self.component.set(component);
        log::info!("loaded plugin ({:01} mb)", size as f64 / (1024. * 1024.));

        Ok(self.component.get().unwrap())
    }

    async fn definition(&self) -> anyhow::Result<&Definition> {
        if let Some(metadata) = self.definition.get() {
            return Ok(metadata);
        }
        // then we get the structure of the plugin.
        let mut store = Store::new(&ENGINE, State::default());
        let component = self.component().await?;
        let (world, _instance) = PluginWorld::instantiate_async(&mut store, component, &*LINKER)
            .await
            .unwrap();
        let structure = world.interface0.call_structure(&mut store).await.unwrap();

        let _ = self.definition.set(structure);

        Ok(self.definition.get().unwrap())
    }

    async fn metadata(&self) -> anyhow::Result<&PluginMetadata> {
        if let Some(metadata) = self.metadata.get() {
            return Ok(metadata);
        }
        let definition = self.definition().await?;
        let _ = self.metadata.set(PluginMetadata {
            name: definition.name.clone(),
            description: definition.description.clone(),
        });
        Ok(self.metadata.get().unwrap())
    }

    pub async fn instance(&self) -> anyhow::Result<PluginInstance> {
        // create the store of models
        let state = State::default();
        let logs = state.logs.clone();
        let mut store = Store::new(&ENGINE, state);
        let component = self.component().await?;
        let definition = self.definition().await?;
        let (world, _instance) = PluginWorld::instantiate_async(&mut store, component, &LINKER)
            .await
            .unwrap();

        let (input_sender, mut input_receiver) = broadcast::channel::<Vec<Input>>(100);
        let (output_sender, output_receiver) = broadcast::channel(100);

        tokio::spawn(async move {
            loop {
                let Ok(inputs) = input_receiver.recv().await else {
                    break;
                };
                let outputs = world.interface0.call_run(&mut store, &inputs).await;
                if output_sender.send(Arc::new(outputs)).is_err() {
                    break;
                }
            }
        });

        Ok(PluginInstance {
            source: self.source.clone(),
            sender: input_sender,
            receiver: output_receiver,
            metadata: definition.clone(),
            logs,
        })
    }

    pub async fn name(&self) -> anyhow::Result<String> {
        Ok(self.metadata().await?.name.clone())
    }

    pub async fn description(&self) -> anyhow::Result<String> {
        Ok(self.metadata().await?.description.clone())
    }
}

pub struct PluginInstance {
    source: PackageIndexEntry,
    metadata: Definition,
    logs: Arc<RwLock<Vec<String>>>,
    sender: broadcast::Sender<Vec<Input>>,
    receiver: broadcast::Receiver<Arc<Result<Vec<Output>, wasmtime::Error>>>,
}

impl std::fmt::Debug for PluginInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PluginInstance")
            .field("metadata", &self.metadata)
            .field("logs", &self.logs)
            .finish()
    }
}

impl Serialize for PluginInstance {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Just serialize the source
        self.source.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PluginInstance {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Just deserialize the source
        let source = PackageIndexEntry::deserialize(deserializer)?;
        Ok(
            async move { load_plugin_from_source(source).instance().await }
                .block_on()
                .unwrap(),
        )
    }
}

impl PluginInstance {
    pub fn run(
        &self,
        inputs: Vec<Input>,
    ) -> impl Future<Output = Option<Arc<Result<Vec<Output>, Error>>>> + 'static {
        tracing::trace!("sending inputs to plugin: {inputs:?}");
        let sender = self.sender.clone();
        let mut receiver = self.receiver.resubscribe();
        async move {
            let _ = sender.send(inputs);
            receiver.recv().await.ok()
        }
    }

    pub fn read_logs(&self) -> LockResult<RwLockReadGuard<Vec<String>>> {
        self.logs.read()
    }

    pub fn metadata(&self) -> &Definition {
        &self.metadata
    }
}

#[tokio::test]
async fn test_load_plugin() {
    // first build the plugin_demo
    // cargo build --release --target wasm32-unknown-unknown
    let command = std::process::Command::new("cargo")
        .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
        .current_dir("./plugins/format")
        .stdout(std::process::Stdio::inherit())
        .output()
        .unwrap();

    println!("{:?}", command);

    let path = "./target/wasm32-unknown-unknown/release/plugin_format.wasm";

    let plugin = load_plugin(&std::path::PathBuf::from(path));

    let instance = plugin.instance().await.unwrap();

    let inputs = vec![
        Input::Single(PrimitiveValue::Text("hello {}".to_string())),
        Input::Single(PrimitiveValue::Text("world".to_string())),
    ];
    let outputs = instance.run(inputs).await.unwrap();
    let outputs = outputs.as_deref().unwrap();
    println!("{:?}", outputs);

    assert_eq!(outputs.len(), 1);
    let first = outputs.first().unwrap();
    match first {
        Output::Single(PrimitiveValue::Text(text)) => {
            assert_eq!(text, "hello world");
        }
        _ => panic!("unexpected text output"),
    }
}

wasmtime::component::bindgen!({
    path: "../wit",
    async: true,
});
