pub use ego_tree::NodeId as StaticNodeId;
pub use headless_chrome::protocol::cdp::DOM::NodeId as DynamicNodeId;
use scraper::{ElementRef, Selector};

use super::Node;

/// A node in either a static or dynamic page.
pub enum AnyNode<'a> {
    /// A node in a static page.
    Static(scraper::ElementRef<'a>),
    /// A node in a dynamic (headless) page.
    Dynamic(Node<'a>),
}

impl AnyNode<'_> {
    /// Get the node reference.
    pub fn node_ref(&self) -> NodeRef {
        match self {
            Self::Static(node) => NodeRef::Static(node.id()),
            Self::Dynamic(node) => node.id(),
        }
    }

    /// Get the name of the element.
    pub fn name(&self) -> &str {
        match self {
            Self::Static(node) => node.value().name(),
            Self::Dynamic(node) => node.name(),
        }
    }

    /// Get the attributes of the element.
    pub fn attributes(&self) -> anyhow::Result<Vec<(String, String)>> {
        match self {
            Self::Static(node) => Ok(node
                .value()
                .attrs()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()),
            Self::Dynamic(node) => node.attributes(),
        }
    }

    /// Get the text content of the node.
    pub fn text(&self) -> anyhow::Result<String> {
        match self {
            Self::Static(node) => Ok(node.text().collect::<Vec<_>>().join("")),
            Self::Dynamic(node) => Ok(node.get_text()?),
        }
    }

    /// Click the node if it in a headless browser.
    pub fn click(&self) -> anyhow::Result<()> {
        match self {
            Self::Static(_) => Err(anyhow::anyhow!("Cannot click static node")),
            Self::Dynamic(node) => {
                node.click()?;
                Ok(())
            }
        }
    }

    /// Type into the node if it is in a headless browser.
    pub fn type_into(&self, keys: &str) -> anyhow::Result<()> {
        match self {
            Self::Static(_) => Err(anyhow::anyhow!("Cannot type into static node")),
            Self::Dynamic(node) => {
                node.send_keys(keys)?;
                Ok(())
            }
        }
    }

    /// Get the outer HTML of the node.
    pub fn outer_html(&self) -> anyhow::Result<String> {
        match self {
            Self::Static(node) => Ok(node.html()),
            Self::Dynamic(node) => Ok(node.outer_html()?),
        }
    }

    /// Find the first child of the node that matches the given selector
    pub fn find_child(&self, selector: &str) -> anyhow::Result<Self> {
        match self {
            Self::Static(node) => {
                let query = Selector::parse(selector)
                    .map_err(|e| anyhow::anyhow!("Invalid query: {}", e))?;
                Ok(Self::Static(
                    node.select(&query)
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("No child found"))?,
                ))
            }
            Self::Dynamic(node) => Ok(Self::Dynamic(node.find_child(selector)?)),
        }
    }

    /// return all the children of the current node
    pub fn children(&self) -> anyhow::Result<Vec<NodeRef>> {
        match self {
            Self::Static(node) => Ok(node
                .children()
                .filter_map(|child| ElementRef::wrap(child).map(|node| NodeRef::Static(node.id())))
                .collect()),
            Self::Dynamic(node) => node.children(),
        }
    }
}

/// An ID of a node in either a static or dynamic page.
#[derive(Debug, Clone, Copy)]
pub enum NodeRef {
    /// A reference to a node in a static page.
    Static(StaticNodeId),
    /// A reference to a node in a dynamic (headless) page.
    Dynamic(DynamicNodeId),
}
