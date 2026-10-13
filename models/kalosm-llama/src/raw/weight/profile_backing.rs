//! Profiling-only source backing. The mapped file must stay unchanged while
//! the model is alive, just as for `Gguf::open`.
use std::{
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};

use fusor::{Error, Result};
use fusor_gguf::{Gguf, RawTensorBytes};
use fusor_ir::{dtype::QLayout, target::HostBytes};

struct MappedWeights(Gguf);

impl AsRef<[u8]> for MappedWeights {
    fn as_ref(&self) -> &[u8] {
        self.0.bytes()
    }
}

type CachedMap = Option<(PathBuf, Arc<MappedWeights>)>;
static MAPPED: OnceLock<Mutex<CachedMap>> = OnceLock::new();

pub(super) fn mapped_bytes(raw: &RawTensorBytes) -> Result<Option<Arc<HostBytes>>> {
    let Some(path) = std::env::var_os("PROFILE_MMAP_WEIGHTS") else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    let mut cached = MAPPED
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| Error::Io("mapped weight cache poisoned".into()))?;
    if let Some((previous, _)) = cached.as_ref() {
        if previous != &path {
            return Err(Error::Io(
                "PROFILE_MMAP_WEIGHTS changed within one process".into(),
            ));
        }
    } else {
        let owner = Arc::new(MappedWeights(Gguf::open(&path)?));
        *cached = Some((path, owner));
    }
    let owner = cached.as_ref().unwrap().1.clone();
    drop(cached);
    let gguf = &owner.0;
    let tensor = gguf
        .tensor(&raw.name)
        .ok_or_else(|| Error::Io(format!("mapped weights have no tensor {}", raw.name)))?;
    if tensor.shape != raw.shape
        || raw.layout != QLayout::Native
        || fusor_gguf::parse::ingest_qfmt(tensor.ty)? != raw.fmt
        || gguf.tensor_bytes(&raw.name)? != raw.bytes.as_ref()
    {
        return Err(Error::Io(format!(
            "mapped tensor {} differs from loaded weights",
            raw.name
        )));
    }
    let start = gguf
        .metadata()
        .tensor_data_offset
        .checked_add(tensor.offset)
        .ok_or_else(|| Error::Io("mapped tensor offset overflow".into()))?;
    let end = start
        .checked_add(tensor.bytes)
        .ok_or_else(|| Error::Io("mapped tensor length overflow".into()))?;
    let start =
        usize::try_from(start).map_err(|_| Error::Io("mapped offset exceeds usize".into()))?;
    let end = usize::try_from(end).map_err(|_| Error::Io("mapped end exceeds usize".into()))?;
    Ok(Some(Arc::new(HostBytes::shared(owner, start..end)?)))
}
