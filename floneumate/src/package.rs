use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct PackageStructure {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) version: String,
}

impl PackageStructure {
    pub fn new(name: &str, version: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            version: version.to_string(),
        }
    }
}
