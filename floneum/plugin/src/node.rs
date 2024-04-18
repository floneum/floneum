use crate::host::AnyNodeRef;
use crate::resource::{Resource, ResourceStorage};

use crate::plugins::main::types::Node;

impl ResourceStorage {
    pub fn with_node<O>(
        &self,
        node: Node,
        f: impl FnOnce(headless_chrome::Element) -> anyhow::Result<O>,
    ) -> anyhow::Result<O> {
        let index = node.into();
        let node = self.get(index).ok_or(anyhow::anyhow!(
            "Node not found; may have been already dropped"
        ))?;
        let tab = Resource::from_index_borrowed(node.page_id);
        let tab = self.get(tab).ok_or(anyhow::anyhow!(
            "Page not found; may have been already dropped"
        ))?;
        f(headless_chrome::Element::new(&tab, node.node_id)?)
    }

    pub(crate) async fn impl_get_element_text(&self, self_: Node) -> wasmtime::Result<String> {
        self.with_node(self_, |node| node.get_inner_text())
    }

    pub(crate) async fn impl_click_element(&self, self_: Node) -> wasmtime::Result<()> {
        self.with_node(self_, |node| {
            node.click()?;
            Ok(())
        })?;
        Ok(())
    }

    pub(crate) async fn impl_type_into_element(
        &self,
        self_: Node,
        keys: String,
    ) -> wasmtime::Result<()> {
        self.with_node(self_, |node| {
            node.type_into(&keys)?;
            Ok(())
        })
    }

    pub(crate) async fn impl_get_element_outer_html(
        &self,
        self_: Node,
    ) -> wasmtime::Result<String> {
        self.with_node(self_, |node| node.get_content())
    }

    pub(crate) async fn impl_screenshot_element(&self, self_: Node) -> wasmtime::Result<Vec<u8>> {
        self.with_node(self_, |node| {
            node.capture_screenshot(
                headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
            )
        })
    }

    pub(crate) async fn impl_find_child_of_element(
        &self,
        self_: Node,
        query: String,
    ) -> wasmtime::Result<Node> {
        let (node_id, page_id) = {
            let index = self_.into();
            let node = self.get(index).ok_or(anyhow::anyhow!("Node not found"))?;
            let page_id = node.page_id;
            let node_id = node.node_id;
            (node_id, page_id)
        };
        let node_id = {
            let page = Resource::from_index_borrowed(page_id);
            let tab = self.get(page).ok_or(anyhow::anyhow!("Page not found"))?;
            let node = headless_chrome::Element::new(&tab, node_id)?;
            let child = node.find_element(&query)?;
            child.node_id
        };
        let child = self.insert(AnyNodeRef { page_id, node_id });
        Ok(Node {
            id: child.index() as u64,
            owned: true,
        })
    }

    pub(crate) fn impl_drop_node(&self, rep: Node) -> wasmtime::Result<()> {
        let index = rep.into();
        self.drop_key(index);
        Ok(())
    }
}
