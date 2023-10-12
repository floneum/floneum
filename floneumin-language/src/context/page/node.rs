pub use ego_tree::NodeId as StaticNodeId;
pub use headless_chrome::protocol::cdp::DOM::NodeId as DynamicNodeId;
use scraper::Selector;

/// A node in either a static or dynamic page.
pub enum AnyNode<'a> {
    /// A node in a static page.
    Static(scraper::ElementRef<'a>),
    /// A node in a dynamic (headless) page.
    Dynamic(headless_chrome::Element<'a>),
}

impl<'a> AnyNode<'a> {
    /// Get the node reference.
    pub fn node_ref(&self) -> NodeRef {
        match self {
            Self::Static(node) => NodeRef::Static(node.id()),
            Self::Dynamic(node) => NodeRef::Dynamic(node.node_id),
        }
    }

    /// Get the text content of the node.
    pub fn text(&self) -> anyhow::Result<String> {
        match self {
            Self::Static(node) => Ok(node.text().collect::<Vec<_>>().join("")),
            Self::Dynamic(node) => Ok(node.get_inner_text()?),
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
                node.type_into(keys)?;
                Ok(())
            }
        }
    }

    /// Get the outer HTML of the node.
    pub fn outer_html(&self) -> anyhow::Result<String> {
        match self {
            Self::Static(node) => Ok(node.html()),
            Self::Dynamic(node) => Ok(node.get_content()?),
        }
    }

    /// Screen shot the node if it is in a headless browser.
    pub fn screenshot(&self) -> anyhow::Result<Vec<u8>> {
        match self {
            Self::Static(_) => Err(anyhow::anyhow!("Cannot take screenshot of static node")),
            Self::Dynamic(node) => Ok(node.capture_screenshot(
                headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
            )?),
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
            Self::Dynamic(node) => Ok(Self::Dynamic(node.find_element(selector)?)),
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
