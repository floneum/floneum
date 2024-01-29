use anyhow::anyhow;
use directories::BaseDirs;
use once_cell::sync::Lazy;

mod package;
pub use package::PackageStructure;

mod index;
pub use index::{FloneumPackageIndex, PackageIndexEntry};

pub use crate::package::Config;

/// The path to the floneum packages directory.
#[tracing::instrument]
pub fn packages_path() -> anyhow::Result<std::path::PathBuf> {
    let base_dirs = BaseDirs::new().ok_or_else(|| anyhow!("No home directory found"))?;
    let path = base_dirs.data_dir().join("floneum").join("packages");
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

static OCTOCRAB: Lazy<octocrab::Octocrab> = Lazy::new(|| match std::env::var("GITHUB_TOKEN") {
    Ok(token) => octocrab::OctocrabBuilder::new()
        .personal_token(token)
        .build()
        .unwrap_or_else(|err| {
            tracing::error!("Failed to create octocrab instance: {}", err);
            unauthenticated_octocrab()
        }),
    Err(_) => unauthenticated_octocrab(),
});

fn unauthenticated_octocrab() -> octocrab::Octocrab {
    tracing::warn!("No GITHUB_TOKEN found, using unauthenticated requests. If you are hitting the rate limit, you can set a GITHUB_TOKEN to increase the rate limit.");
    octocrab::OctocrabBuilder::new().build().unwrap()
}
