use std::{fmt::Display, path::PathBuf, sync::OnceLock};

use candle_core::{utils::*, Device};

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
    pub fn huggingface(model_id: String, revision: String, file: String) -> Self {
        Self::HuggingFace {
            model_id,
            revision,
            file,
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
