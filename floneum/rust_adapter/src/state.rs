use std::ops::{Deref, DerefMut};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub fn load<T: DeserializeOwned>(key: &[u8]) -> Result<T, bincode::Error> {
    let bytes = crate::plugins::main::imports::load(key);
    let state = bincode::deserialize(&bytes)?;
    Ok(state)
}

pub fn store<T: Serialize>(path: &[u8], state: &T) -> Result<(), bincode::Error> {
    let bytes = bincode::serialize(state)?;
    crate::plugins::main::imports::store(path, &bytes);
    Ok(())
}

pub struct PersistentState<T: Serialize + DeserializeOwned> {
    path: Vec<u8>,
    state: T,
}

impl<T: Serialize + DeserializeOwned> PersistentState<T> {
    #[track_caller]
    pub fn new(or_init: impl FnOnce() -> T) -> Self {
        let path = std::panic::Location::caller().to_string();
        Self::new_with_key(path.into_bytes(), or_init)
    }

    pub fn new_with_key(key: Vec<u8>, or_init: impl FnOnce() -> T) -> Self {
        let state = load(&key).unwrap_or_else(|_| or_init());
        Self { path: key, state }
    }
}

impl<T: Serialize + DeserializeOwned> Deref for PersistentState<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl<T: Serialize + DeserializeOwned> DerefMut for PersistentState<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl<T: Serialize + DeserializeOwned> Drop for PersistentState<T> {
    fn drop(&mut self) {
        store(&self.path, &self.state).unwrap();
    }
}

pub struct Cache<K: Serialize + DeserializeOwned, T: Serialize + DeserializeOwned> {
    path: String,
    phantom: std::marker::PhantomData<(K, T)>,
}

impl<K: Serialize + DeserializeOwned, T: Serialize + DeserializeOwned> Default for Cache<K, T> {
    #[track_caller]
    fn default() -> Self {
        let path = std::panic::Location::caller().to_string();
        Self::new_with_key(&path)
    }
}

impl<K: Serialize + DeserializeOwned, T: Serialize + DeserializeOwned> Cache<K, T> {
    #[track_caller]
    pub fn new() -> Self {
        Default::default()
    }

    pub fn new_with_key(path: &str) -> Self {
        Self {
            path: path.to_string(),
            phantom: std::marker::PhantomData,
        }
    }

    pub fn get(&self, key: K) -> Option<T> {
        let key = bincode::serialize(&Entry {
            key: self.path.clone(),
            value: key,
        })
        .unwrap();
        load(&key).ok()
    }

    pub fn set(&self, key: K, value: T) {
        let key = bincode::serialize(&Entry {
            key: self.path.clone(),
            value: key,
        })
        .unwrap();
        store(&key, &value).unwrap();
    }

    pub fn remove(&self, key: K) {
        let key = bincode::serialize(&Entry {
            key: self.path.clone(),
            value: key,
        })
        .unwrap();
        crate::plugins::main::imports::unload(&key);
    }
}

#[derive(Serialize, Deserialize)]
struct Entry<T> {
    key: String,
    value: T,
}
