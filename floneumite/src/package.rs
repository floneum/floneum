use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct PackageStructure {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) package_version: String,
    pub(crate) binding_version: String,
}

impl PackageStructure {
    pub fn new(name: &str, version: &str, description: &str, binding_version: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            package_version: version.to_string(),
            binding_version: binding_version.to_string(),
        }
    }
}
