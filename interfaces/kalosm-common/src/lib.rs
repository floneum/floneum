use std::{fmt::Display, future::Future, path::PathBuf, pin::Pin, sync::OnceLock};

use candle_core::{backend::BackendStorage, utils::*, Device, Storage, Tensor, WithDType};

mod cache;
pub use cache::*;

/// Create a candle device that uses any available accelerator.
pub fn accelerated_device_if_available() -> candle_core::Result<Device> {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    if let Some(device) = DEVICE.get() {
        return Ok(device.clone());
    }
    let device = if cuda_is_available() {
        Device::new_cuda(0)?
    } else if metal_is_available() {
        Device::new_metal(0)?
    } else {
        #[cfg(all(debug_assertions, target_os = "macos", target_arch = "aarch64"))]
        {
            println!("Running on CPU, to run on GPU(metal), build with `--features metal`");
        }
        #[cfg(not(all(debug_assertions, target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build with `--features cuda`");
        }
        Device::Cpu
    };
    let _ = DEVICE.set(device.clone());
    Ok(device)
}

/// A source for a file, either from Hugging Face or a local path
#[derive(Clone, Debug)]
pub enum FileSource {
    /// A file from Hugging Face
    HuggingFace {
        /// The model id to use
        model_id: String,
        /// The revision to use
        revision: String,
        /// The file to use
        file: String,
    },
    /// A local file
    Local(PathBuf),
}

impl Display for FileSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileSource::HuggingFace {
                model_id,
                revision,
                file,
            } => write!(f, "hf://{}/{}/{}", model_id, revision, file),
            FileSource::Local(path) => write!(f, "{}", path.display()),
        }
    }
}

impl FileSource {
    /// Create a new source for a file from Hugging Face
    pub fn huggingface(
        model_id: impl ToString,
        revision: impl ToString,
        file: impl ToString,
    ) -> Self {
        Self::HuggingFace {
            model_id: model_id.to_string(),
            revision: revision.to_string(),
            file: file.to_string(),
        }
    }

    /// Create a new source for a local file
    pub fn local(path: PathBuf) -> Self {
        Self::Local(path)
    }

    /// Check if the file exists locally (if it is a local file or if it has been downloaded)
    pub fn downloaded(&self) -> bool {
        let cache = Cache::default();
        cache.exists(self)
    }
}

/// Wrap a closure in a release pool if the metal feature is enabled
pub fn maybe_autoreleasepool<T>(f: impl FnOnce() -> T) -> T {
    #[cfg(feature = "metal")]
    // Adding a manual autoreleasepool here is necessary to avoid a memory leak https://github.com/huggingface/candle/issues/2271
    {
        metal::objc::rc::autoreleasepool(f)
    }
    #[cfg(not(feature = "metal"))]
    {
        f()
    }
}

/// A future that is boxed and pinned.
pub type BoxedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Clear a Vec<T> and copy the contents of a tensor into it.
pub fn copy_tensor_into_vec<T: WithDType>(
    tensor: &Tensor,
    into: &mut Vec<T>,
) -> candle_core::Result<()> {
    into.clear();
    assert_eq!(tensor.rank(), 1);
    let mut from_cpu_storage = |cpu_storage: &candle_core::CpuStorage,
                                layout: &candle_core::Layout| {
        let data = cpu_storage.as_slice()?;
        match layout.contiguous_offsets() {
            Some((o1, o2)) => into.extend_from_slice(&data[o1..o2]),
            None => {
                for logit in tensor.strided_index().map(|i| data[i]) {
                    into.push(logit);
                }
            }
        };
        candle_core::Result::Ok(())
    };
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Cpu(storage) => from_cpu_storage(storage, layout),
        Storage::Cuda(storage) => from_cpu_storage(&storage.to_cpu_storage()?, layout),
        Storage::Metal(storage) => from_cpu_storage(&storage.to_cpu_storage()?, layout),
    }
}
