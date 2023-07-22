use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Config {
    packages: Vec<PackageStructure>,
}

impl Config {
    pub fn push(&mut self, package: PackageStructure) {
        self.packages.push(package);
    }

    pub fn packages(&self) -> &[PackageStructure] {
        &self.packages
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PackageStructure {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) authors: Vec<String>,
    #[serde(default)]
    pub(crate) description: String,
    #[serde(default = "default_version")]
    pub(crate) package_version: String,
    #[serde(default = "default_binding_version")]
    pub(crate) binding_version: String,
}

fn default_version() -> String {
    "0.1".to_string()
}

fn default_binding_version() -> String {
    "*".to_string()
}

impl PackageStructure {
    pub fn new(name: &str, version: &str, description: &str, binding_version: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            package_version: version.to_string(),
            binding_version: binding_version.to_string(),
            authors: Vec::new(),
        }
    }

    pub fn with_authors(self, authors: Vec<String>) -> Self {
        Self { authors, ..self }
    }
}
