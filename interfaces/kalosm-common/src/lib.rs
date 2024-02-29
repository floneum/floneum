use std::path::PathBuf;

use candle_core::{utils::*, Device};
use hf_hub::{Repo, RepoType};

/// Create a candle device that uses any available accelerator.
pub fn accelerated_device_if_available() -> candle_core::Result<Device> {
    if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(debug_assertions, target_os = "macos", target_arch = "aarch64"))]
        {
            println!("Running on CPU, to run on GPU(metal), build with `--features metal`");
        }
        #[cfg(not(all(debug_assertions, target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

/// A source for a file, either from Hugging Face or a local path
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

    /// Get the path to the file
    pub fn path(&self) -> anyhow::Result<std::path::PathBuf> {
        match self {
            Self::HuggingFace {
                model_id,
                revision,
                file,
            } => {
                let api = hf_hub::api::sync::Api::new()?;
                let repo = Repo::with_revision(
                    model_id.to_string(),
                    RepoType::Model,
                    revision.to_string(),
                );
                let api = api.repo(repo);
                let model_path = api.get(file)?;
                Ok(model_path)
            }
            Self::Local(path) => Ok(path.clone()),
        }
    }
}
