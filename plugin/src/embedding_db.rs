use crate::host::State;
use crate::plugins::main;
use crate::plugins::main::imports::{self};
use crate::plugins::main::types::{
    EitherStructure, Embedding, EmbeddingDb, EmbeddingModel, Model, Node, NumberParameters, Page,
    SequenceParameters, Structure, ThenStructure, UnsignedRange,
};
use crate::Exports;
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
impl main::types::HostEmbeddingDb for State {
    async fn new(
        &mut self,
        embeddings: Vec<Embedding>,
        documents: Vec<String>,
    ) -> wasmtime::Result<wasmtime::component::Resource<EmbeddingDb>> {
        let embeddings = embeddings.into_iter().map(|x| x.vector.into()).collect();
        let documents = documents
            .into_iter()
            .map(|x| Document::from_parts(String::new(), x))
            .collect();
        let db = VectorDB::new(embeddings, documents);
        let idx = self.embedding_dbs.insert(db);
        Ok(wasmtime::component::Resource::new_own(idx as u32))
    }

    async fn add_embedding(
        &mut self,
        self_: wasmtime::component::Resource<EmbeddingDb>,
        embedding: Embedding,
        document: String,
    ) -> wasmtime::Result<()> {
        self.embedding_dbs[self_.rep() as usize].add_embedding(
            embedding.vector.into(),
            Document::from_parts(String::new(), document),
        );
        Ok(())
    }

    async fn find_closest_documents(
        &mut self,
        self_: wasmtime::component::Resource<EmbeddingDb>,
        search: Embedding,
        count: u32,
    ) -> wasmtime::Result<Vec<String>> {
        let documents = self.embedding_dbs[self_.rep() as usize]
            .get_closest(search.vector.into(), count as usize);
        Ok(documents
            .into_iter()
            .map(|(_, document)| document.body().to_string())
            .collect())
    }

    async fn find_documents_within(
        &mut self,
        self_: wasmtime::component::Resource<EmbeddingDb>,
        search: Embedding,
        distance: f32,
    ) -> wasmtime::Result<Vec<String>> {
        Ok(self.embedding_dbs[self_.rep() as usize]
            .get_within(search.vector.into(), distance)
            .into_iter()
            .map(|(_, document)| document.body().to_string())
            .collect())
    }

    fn drop(&mut self, rep: wasmtime::component::Resource<EmbeddingDb>) -> wasmtime::Result<()> {
        self.embedding_dbs.remove(rep.rep() as usize);
        Ok(())
    }
}
