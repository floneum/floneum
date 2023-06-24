// Clone a repository from any URL or Path to a given target directory

use anyhow::{anyhow, Ok};
use directories::BaseDirs;

mod package;

pub struct Index {
    entries: Vec<Package>,
}

impl Index {
    pub fn new() -> anyhow::Result<Self> {
        let path = packages_path()?;
        if !path.exists(){
            download_package_index()?;
        }
        let entries = std::fs::read_dir(path)?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.is_dir() {
                    Some(path)
                } else {
                    None
                }
            })
            .filter_map(|path| Package::try_new(path).ok())
            .collect();

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
    fn try_new(path: std::path::PathBuf) -> anyhow::Result<Self> {
        let toml_path = path.join("floneum.toml");
        let structure = toml::from_str(&std::fs::read_to_string(toml_path)?)?;
        let path = path.join("package.wasm");
        Ok(Self { path, structure })
    }

    pub fn name(&self) -> &str {
        &self.structure.name
    }

    pub fn description(&self) -> &str {
        &self.structure.description
    }

    pub fn version(&self) -> &str {
        &self.structure.version
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub fn load_wasm(&self) -> anyhow::Result<Vec<u8>> {
        let wasm_path = self.path.join("package.wasm");
        Ok(std::fs::read(wasm_path)?)
    }
}

fn packages_path() -> anyhow::Result<std::path::PathBuf> {
    let base_dirs = BaseDirs::new().ok_or_else(|| anyhow!("No home directory found"))?;
    Ok(base_dirs.data_dir().join("floneum").join("packages"))
}

fn download_package_index() -> anyhow::Result<()> {
    let repo_url = "https://github.com/floneum/floneum-packages";

    let dst = packages_path()?;

    gix::interrupt::init_handler(|| {})?;
    std::fs::create_dir_all(&dst)?;
    let url = gix::url::parse(repo_url.into())?;

    let mut prepare_clone = gix::prepare_clone(url, &dst)?;

    let (mut prepare_checkout, _) = prepare_clone
        .fetch_then_checkout(gix::progress::Discard, &gix::interrupt::IS_INTERRUPTED)?;

    prepare_checkout.main_worktree(gix::progress::Discard, &gix::interrupt::IS_INTERRUPTED)?;

    Ok(())
}
