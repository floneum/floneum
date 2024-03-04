use core::fmt;
use std::{
    fmt::{Display, Formatter},
    str::FromStr,
};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Config {
    packages: Vec<PackageStructure>,
}

impl Config {
    pub fn new(packages: Vec<PackageStructure>) -> Self {
        Self { packages }
    }

    pub fn push(&mut self, package: PackageStructure) {
        self.packages.push(package);
    }

    pub fn packages(&self) -> &[PackageStructure] {
        &self.packages
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Category {
    Utility,
    Logic,
    Data,
    AI,
    IO,
    #[default]
    Other,
}

impl Category {
    pub const ALL: [Category; 6] = [
        Category::Utility,
        Category::Logic,
        Category::Data,
        Category::AI,
        Category::IO,
        Category::Other,
    ];
}

impl FromStr for Category {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "utility" => Ok(Category::Utility),
            "logic" => Ok(Category::Logic),
            "data" => Ok(Category::Data),
            "ai" => Ok(Category::AI),
            "io" => Ok(Category::IO),
            _ => Ok(Category::Other),
        }
    }
}

impl Display for Category {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Category::Utility => write!(f, "Utility"),
            Category::Logic => write!(f, "Logic"),
            Category::Data => write!(f, "Data"),
            Category::AI => write!(f, "AI"),
            Category::IO => write!(f, "IO"),
            Category::Other => write!(f, "Other"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct PackageStructure {
    pub name: String,
    #[serde(default)]
    pub authors: Vec<String>,
    #[serde(default)]
    pub category: Category,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_version")]
    pub package_version: String,
    #[serde(default = "current_binding_version")]
    pub binding_version: String,
}

fn default_version() -> String {
    "0.1".to_string()
}

fn current_binding_version() -> String {
    crate::CURRENT_BINDING_VERSION.to_string()
}

impl PackageStructure {
    pub fn new(
        name: &str,
        version: &str,
        category: Category,
        description: &str,
        binding_version: &str,
    ) -> Self {
        Self {
            name: name.to_string(),
            category,
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
