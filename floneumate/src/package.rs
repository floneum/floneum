use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct PackageStructure {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) version: String,
}
