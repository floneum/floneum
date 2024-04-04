use kalosm::language::Tab;
use std::{
    any::{Any, TypeId},
    borrow::BorrowMut,
    collections::HashMap,
    marker::PhantomData,
};

use kalosm::language::{Node, Page};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use slab::Slab;
use std::sync::Arc;
use wasmtime::component::Type;

use crate::{
    embedding::LazyTextEmbeddingModel, embedding_db::VectorDBWithDocuments, host::AnyNodeRef,
    llm::LazyTextGenerationModel, plugins::main,
};

#[derive(Default, Clone)]
pub struct ResourceStorage {
    map: Arc<RwLock<HashMap<TypeId, Slab<Box<dyn Any + Send + Sync>>>>>,
}

impl ResourceStorage {
    pub(crate) fn insert<T: Send + Sync + 'static>(&mut self, item: T) -> Resource<T> {
        let ty_id = TypeId::of::<T>();
        let mut binding = self.map.write();
        let slab = binding.entry(ty_id).or_default();
        let id = slab.insert(Box::new(item));
        Resource {
            index: id,
            owned: true,
            phantom: PhantomData,
        }
    }

    pub(crate) fn get<T: Send + Sync + 'static>(
        &self,
        key: Resource<T>,
    ) -> Option<parking_lot::lock_api::MappedRwLockReadGuard<'_, parking_lot::RawRwLock, T>> {
        RwLockReadGuard::try_map(self.map.read(), |r| {
            r.get(&TypeId::of::<T>())
                .and_then(|slab| slab.get(key.index))
                .and_then(|any| any.downcast_ref())
        })
        .ok()
    }

    pub(crate) fn get_mut<T: Send + Sync + 'static>(
        &mut self,
        key: Resource<T>,
    ) -> Option<parking_lot::lock_api::MappedRwLockWriteGuard<'_, parking_lot::RawRwLock, T>> {
        RwLockWriteGuard::try_map(self.map.write(), |r| {
            r.get_mut(&TypeId::of::<T>())
                .and_then(|slab| slab.get_mut(key.index))
                .and_then(|any| any.downcast_mut())
        })
        .ok()
    }

    pub(crate) fn drop_key<T: 'static>(&mut self, key: Resource<T>) {
        assert!(key.owned);
        if let Some(slab) = self.map.write().get_mut(&TypeId::of::<T>()) {
            slab.remove(key.index);
        }
    }
}

pub(crate) struct Resource<T> {
    index: usize,
    owned: bool,
    phantom: PhantomData<T>,
}

impl<T> Clone for Resource<T> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            owned: self.owned,
            phantom: PhantomData,
        }
    }
}

impl<T> Copy for Resource<T> {}

impl<T> Resource<T> {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn owned(&self) -> bool {
        self.owned
    }
}

impl<T> Resource<T> {
    pub(crate) fn from_index_owned(index: usize) -> Self {
        Self {
            index,
            owned: true,
            phantom: PhantomData,
        }
    }

    pub(crate) fn from_index_borrowed(index: usize) -> Self {
        Self {
            index,
            owned: false,
            phantom: PhantomData,
        }
    }
}

impl From<main::types::EmbeddingModel> for Resource<LazyTextEmbeddingModel> {
    fn from(value: main::types::EmbeddingModel) -> Self {
        Self {
            index: value.id as usize,
            owned: value.owned,
            phantom: PhantomData,
        }
    }
}

impl From<main::types::TextGenerationModel> for Resource<LazyTextGenerationModel> {
    fn from(value: main::types::TextGenerationModel) -> Self {
        Self {
            index: value.id as usize,
            owned: value.owned,
            phantom: PhantomData,
        }
    }
}

impl From<main::types::EmbeddingDb> for Resource<VectorDBWithDocuments> {
    fn from(value: main::types::EmbeddingDb) -> Self {
        Self {
            index: value.id as usize,
            owned: value.owned,
            phantom: PhantomData,
        }
    }
}

impl From<main::types::Page> for Resource<Arc<Tab>> {
    fn from(value: main::types::Page) -> Self {
        Self {
            index: value.id as usize,
            owned: value.owned,
            phantom: PhantomData,
        }
    }
}

impl From<main::types::Node> for Resource<AnyNodeRef> {
    fn from(value: main::types::Node) -> Self {
        Self {
            index: value.id as usize,
            owned: value.owned,
            phantom: PhantomData,
        }
    }
}
