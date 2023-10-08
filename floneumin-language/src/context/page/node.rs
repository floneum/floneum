pub use ego_tree::NodeId as StaticNodeId;
pub use headless_chrome::protocol::cdp::DOM::NodeId as DynamicNodeId;
use scraper::Selector;

pub enum Node<'a> {
    Static(scraper::ElementRef<'a>),
    Dynamic(headless_chrome::Element<'a>),
}

impl<'a> Node<'a> {
    pub fn node_ref(&self) -> NodeRef {
        match self {
            Self::Static(node) => NodeRef::Static(node.id()),
            Self::Dynamic(node) => NodeRef::Dynamic(node.node_id),
        }
    }

    pub fn text(&self) -> anyhow::Result<String> {
        match self {
            Self::Static(node) => Ok(node.text().collect::<Vec<_>>().join("")),
            Self::Dynamic(node) => Ok(node.get_inner_text()?),
        }
    }

    pub fn click(&self) -> anyhow::Result<()> {
        match self {
            Self::Static(_) => Err(anyhow::anyhow!("Cannot click static node")),
            Self::Dynamic(node) => {
                node.click()?;
                Ok(())
            }
        }
    }

    pub fn type_into(&self, keys: &str) -> anyhow::Result<()> {
        match self {
            Self::Static(_) => Err(anyhow::anyhow!("Cannot type into static node")),
            Self::Dynamic(node) => {
                node.type_into(keys)?;
                Ok(())
            }
        }
    }

    pub fn outer_html(&self) -> anyhow::Result<String> {
        match self {
            Self::Static(node) => Ok(node.html()),
            Self::Dynamic(node) => Ok(node.get_content()?),
        }
    }

    pub fn screenshot(&self) -> anyhow::Result<Vec<u8>> {
        match self {
            Self::Static(_) => Err(anyhow::anyhow!("Cannot take screenshot of static node")),
            Self::Dynamic(node) => Ok(node.capture_screenshot(
                headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
            )?),
        }
    }

    pub fn find_child(&self, query: &str) -> anyhow::Result<Self> {
        match self {
            Self::Static(node) => {
                let query =
                    Selector::parse(query).map_err(|e| anyhow::anyhow!("Invalid query: {}", e))?;
                Ok(Self::Static(
                    node.select(&query)
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("No child found"))?,
                ))
            }
            Self::Dynamic(node) => Ok(Self::Dynamic(node.find_element(query)?)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum NodeRef {
    Static(StaticNodeId),
    Dynamic(DynamicNodeId),
}
