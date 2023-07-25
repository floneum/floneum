// Clone a repository from any URL or Path to a given target directory

use anyhow::anyhow;
use directories::BaseDirs;

mod package;
pub use package::PackageStructure;

mod index;
pub use index::{FloneumPackageIndex, PackageIndexEntry};

pub use crate::package::Config;

#[tracing::instrument]
fn packages_path() -> anyhow::Result<std::path::PathBuf> {
    let base_dirs = BaseDirs::new().ok_or_else(|| anyhow!("No home directory found"))?;
    let path = base_dirs.data_dir().join("floneum").join("packages");
    std::fs::create_dir_all(&path)?;
    Ok(path)
}
