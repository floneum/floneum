use kalosm::language::Tab;
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    marker::PhantomData,
};

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use slab::Slab;
use std::sync::Arc;

use crate::{
    embedding::LazyTextEmbeddingModel, embedding_db::VectorDBWithDocuments, host::AnyNodeRef,
    llm::LazyTextGenerationModel, plugins::main,
};

type ResourceMap = Arc<RwLock<HashMap<TypeId, Slab<Box<dyn Any + Send + Sync>>>>>;

#[derive(Default, Clone)]
pub struct ResourceStorage {
    map: ResourceMap,
}

impl ResourceStorage {
    pub(crate) fn insert<T: Send + Sync + 'static>(&self, item: T) -> Resource<T> {
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
        &self,
        key: Resource<T>,
    ) -> Option<parking_lot::lock_api::MappedRwLockWriteGuard<'_, parking_lot::RawRwLock, T>> {
        RwLockWriteGuard::try_map(self.map.write(), |r| {
            r.get_mut(&TypeId::of::<T>())
                .and_then(|slab| slab.get_mut(key.index))
                .and_then(|any| any.downcast_mut())
        })
        .ok()
    }

    pub(crate) fn drop_key<T: 'static>(&self, key: Resource<T>) {
        assert!(key.owned);
        if let Some(slab) = self.map.write().get_mut(&TypeId::of::<T>()) {
            slab.remove(key.index);
        }
    }
}

/// A typed resource that is stored in [`ResourceStorage`].
pub struct Resource<T> {
    index: usize,
    owned: bool,
    phantom: PhantomData<T>,
}

impl<T> Clone for Resource<T> {
    fn clone(&self) -> Self {
        *self
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
    pub(crate) fn from_index_borrowed(index: usize) -> Self {
        Self {
            index,
            owned: false,
            phantom: PhantomData,
        }
    }
}

impl From<main::types::EmbeddingModelResource> for Resource<LazyTextEmbeddingModel> {
    fn from(value: main::types::EmbeddingModelResource) -> Self {
        Self {
            index: value.id as usize,
            owned: value.owned,
            phantom: PhantomData,
        }
    }
}

impl From<main::types::TextGenerationModelResource> for Resource<LazyTextGenerationModel> {
    fn from(value: main::types::TextGenerationModelResource) -> Self {
        Self {
            index: value.id as usize,
            owned: value.owned,
            phantom: PhantomData,
        }
    }
}

impl From<main::types::EmbeddingDbResource> for Resource<VectorDBWithDocuments> {
    fn from(value: main::types::EmbeddingDbResource) -> Self {
        Self {
            index: value.id as usize,
            owned: value.owned,
            phantom: PhantomData,
        }
    }
}

impl From<main::types::PageResource> for Resource<Arc<Tab>> {
    fn from(value: main::types::PageResource) -> Self {
        Self {
            index: value.id as usize,
            owned: value.owned,
            phantom: PhantomData,
        }
    }
}

impl From<main::types::NodeResource> for Resource<AnyNodeRef> {
    fn from(value: main::types::NodeResource) -> Self {
        Self {
            index: value.id as usize,
            owned: value.owned,
            phantom: PhantomData,
        }
    }
}
