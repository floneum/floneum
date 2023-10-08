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
impl main::types::HostNode for State {
    async fn get_element_text(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
    ) -> wasmtime::Result<String> {
        todo!()
    }

    async fn click_element(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
    ) -> wasmtime::Result<()> {
        todo!()
    }

    async fn type_into_element(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
        keys: String,
    ) -> wasmtime::Result<()> {
        todo!()
    }

    async fn get_element_outer_html(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
    ) -> wasmtime::Result<String> {
        todo!()
    }

    async fn screenshot_element(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
    ) -> wasmtime::Result<Vec<u8>> {
        todo!()
    }

    async fn find_child_of_element(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
        query: String,
    ) -> wasmtime::Result<wasmtime::component::Resource<Node>> {
        todo!()
    }

    fn drop(&mut self, rep: wasmtime::component::Resource<Node>) -> wasmtime::Result<()> {
        todo!()
    }
}
