use crate::host::State;
use crate::plugins::main;
use crate::plugins::main::imports::{self};
use crate::plugins::main::types::{
    EitherStructure, Embedding, EmbeddingDb, EmbeddingModel, Model, Node, NumberParameters, Page,
    SequenceParameters, Structure, ThenStructure, UnsignedRange,
};
use floneumin::floneumin_language::context::document::Document;
use floneumin::floneumin_language::floneumin_sample::structured::StructuredSampler;
use floneumin::floneumin_language::floneumin_sample::structured_parser::StructureParser;
use floneumin::floneumin_language::local::{Bert, LocalSession, Mistral, Phi};
use floneumin::floneumin_language::model::{Model as _, *};
use floneumin::floneumin_language::vector_db::VectorDB;
use once_cell::sync::Lazy;
use reqwest::header::{HeaderMap, HeaderName};
use slab::Slab;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use wasmtime::component::Linker;
use wasmtime::component::__internal::async_trait;
use wasmtime::Config;
use wasmtime::Engine;
use wasmtime_wasi::preview2::{self, DirPerms, FilePerms, WasiView};
use wasmtime_wasi::Dir;

#[async_trait]
impl main::types::HostPage for State {
    async fn new(
        &mut self,
        mode: main::types::BrowserMode,
        url: String,
    ) -> wasmtime::Result<wasmtime::component::Resource<Page>> {
        todo!()
    }

    async fn find_in_current_page(
        &mut self,
        self_: wasmtime::component::Resource<Page>,
        query: String,
    ) -> wasmtime::Result<wasmtime::component::Resource<Node>> {
        todo!()
    }

    async fn screenshot_browser(
        &mut self,
        self_: wasmtime::component::Resource<Page>,
    ) -> wasmtime::Result<Vec<u8>> {
        todo!()
    }

    fn drop(&mut self, rep: wasmtime::component::Resource<Page>) -> wasmtime::Result<()> {
        todo!()
    }
}
