use crate::exports::plugins::main::definitions::*;
use crate::host::State;
use crate::host::ENGINE;
use crate::host::LINKER;
use crate::Both;
use anyhow::Error;
use floneumite::PackageIndexEntry;
use pollster::FutureExt;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;
use std::future::Future;
use std::path::Path;
use std::sync::Arc;
use std::sync::LockResult;
use std::sync::RwLock;
use std::sync::RwLockReadGuard;
use tokio::sync::broadcast;
use wasmtime::component::Component;
use wasmtime::Store;
use wit_component::ComponentEncoder;

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

impl PluginMetadata {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> &str {
        &self.description
    }
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
        let (world, _instance) = Both::instantiate_async(&mut store, component, &*LINKER).await?;
        let structure = world.interface0.call_structure(&mut store).await.unwrap();

        let _ = self.definition.set(structure);

        Ok(self.definition.get().unwrap())
    }

    pub async fn metadata(&self) -> anyhow::Result<&PluginMetadata> {
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
        let (world, _instance) = Both::instantiate_async(&mut store, component, &LINKER)
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
