// Clone a repository from any URL or Path to a given target directory

use std::path::Path;

use anyhow::{anyhow, Ok};
use directories::BaseDirs;

mod package;
pub use package::PackageStructure;

pub use crate::package::Config;

#[derive(Default)]
pub struct FloneumPackageIndex {
    entries: Vec<Package>,
}

impl FloneumPackageIndex {
    pub async fn fetch() -> anyhow::Result<Self> {
        let path = packages_path()?;
        if path.exists() {
            // remove the old packages
            // TODO: use git fetch to update the packages
            std::fs::remove_dir_all(&path)?;
        }
        let entries = download_package_index(&path).await?;

        Ok(Self { entries })
    }

    pub fn entries(&self) -> &[Package] {
        &self.entries
    }
}

#[derive(Debug)]
pub struct Package {
    path: std::path::PathBuf,
    structure: package::PackageStructure,
}

impl Package {
    pub fn name(&self) -> &str {
        &self.structure.name
    }

    pub fn description(&self) -> &str {
        &self.structure.description
    }

    pub fn version(&self) -> &str {
        &self.structure.package_version
    }

    pub fn binding_version(&self) -> &str {
        &self.structure.binding_version
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub fn wasm_path(&self) -> std::path::PathBuf {
        self.path.join("package.wasm")
    }

    pub fn load_wasm(&self) -> anyhow::Result<Vec<u8>> {
        let wasm_path = self.path.join("package.wasm");
        Ok(std::fs::read(wasm_path)?)
    }
}

#[tracing::instrument]
fn packages_path() -> anyhow::Result<std::path::PathBuf> {
    let base_dirs = BaseDirs::new().ok_or_else(|| anyhow!("No home directory found"))?;
    Ok(base_dirs.data_dir().join("floneum").join("packages"))
}

#[tracing::instrument]
async fn download_package_index(path: &Path) -> anyhow::Result<Vec<Package>> {
    if path.exists() {
        // remove the old packages
        // TODO: use git fetch to update the packages
        std::fs::remove_dir_all(&path)?;
    }
    let instance = octocrab::instance();
    let page = instance
        .search()
        .repositories("topic:floneum")
        .sort("stars")
        .order("desc")
        .send()
        .await?;
    let mut combined_packages = Vec::new();
    for item in page.items {
        if let Some(author) = &item.owner {
            let repo_handle = instance.repos(author.login.clone(), item.name.clone());
            let commits = match repo_handle.list_commits().send().await {
                std::result::Result::Ok(commits) => commits,
                Err(err) => {
                    log::error!("Error loading repo commits: {}", err);
                    continue;
                }
            };
            if let Some(last_commit) = commits.items.first() {
                log::trace!("found repo: user: {} repo: {}", author.login, item.name);
                let file = repo_handle
                    .raw_file(last_commit.sha.clone(), "dist/floneum.toml")
                    .await?;
                let body = file.into_body();
                let bytes = hyper::body::to_bytes(body).await;
                if let core::result::Result::Ok(as_str) = std::str::from_utf8(&bytes.unwrap()) {
                    if let std::result::Result::Ok(package) = toml::from_str::<Config>(as_str) {
                        log::trace!("found package: {:#?}", package);
                        for package in package.packages() {
                            let repo_path = format!("dist/{}/package.wasm", package.name);
                            let repo_handle =
                                instance.repos(author.login.clone(), item.name.clone());
                            let file = repo_handle
                                .raw_file(last_commit.sha.clone(), &repo_path)
                                .await?;
                            let body = file.into_body();
                            if let std::result::Result::Ok(bytes) =
                                hyper::body::to_bytes(body).await
                            {
                                let package_path = path.join(&package.name);
                                std::fs::create_dir_all(&package_path)?;
                                let wasm_path = package_path.join("package.wasm");
                                std::fs::write(wasm_path, bytes)?;
                                let package = Package {
                                    path: package_path,
                                    structure: package.clone(),
                                };
                                combined_packages.push(package);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(combined_packages)
}

#[tokio::test]
async fn get_plugins() {
    let path = packages_path().unwrap();
    let packages = download_package_index(&path).await.unwrap();
    for package in packages {
        println!("{:#?}", package);
    }
}
