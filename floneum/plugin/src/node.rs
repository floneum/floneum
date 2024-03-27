use crate::host::{AnyNodeRef, State};
use crate::resource::Resource;
use crate::resource::ResourceStorage;

use crate::plugins::main::types::Node;

impl State {
    pub fn with_node(
        &self,
        node: Node,
        f: impl FnOnce(headless_chrome::Element) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        let index = node.into();
        let node = self.resources.get(index).ok_or(anyhow::anyhow!(
            "Node not found; may have been already dropped"
        ))?;
        let tab = Resource::from_index_borrowed(node.page_id);
        let tab = self.resources.get(tab).ok_or(anyhow::anyhow!(
            "Page not found; may have been already dropped"
        ))?;
        f(headless_chrome::Element::new(&tab, node.node_id)?)?;
        Ok(())
    }
}

impl State {
    pub(crate) async fn impl_get_element_text(&mut self, self_: Node) -> wasmtime::Result<String> {
        self.with_node(self_, |node| {node.get_inner_text();Ok(())})?;
        Ok(())
    }

    pub(crate) async fn impl_click_element(&mut self, self_: Node) -> wasmtime::Result<()> {
        self.with_node(self_, |node| {
            node.click()?;
            Ok(())
        })?;
        Ok(())
    }

    pub(crate) async fn impl_type_into_element(
        &mut self,
        self_: Node,
        keys: String,
    ) -> wasmtime::Result<()> {
        self.with_node(self_, |node| {
            node.type_into(&keys)?;
            Ok(())
        })?;
        Ok(())
    }

    pub(crate) async fn impl_get_element_outer_html(
        &mut self,
        self_: Node,
    ) -> wasmtime::Result<String> {
        self.with_node(self_, |node| {node.get_content();Ok(())})?;
    }

    pub(crate) async fn impl_screenshot_element(
        &mut self,
        self_: Node,
    ) -> wasmtime::Result<Vec<u8>> {
        self.with_node(self_, |node| {
            node.capture_screenshot(
                headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
            )
        })?;
    }

    pub(crate) async fn impl_find_child_of_element(
        &mut self,
        self_: Node,
        query: String,
    ) -> wasmtime::Result<Node> {
        let index = self_.into();
        let node = self
            .resources
            .get(index)
            .ok_or(anyhow::anyhow!("Node not found"))?;
        let page_id = node.page_id;
        let page = Resource::from_index_borrowed(page_id);
        let tab = self
            .resources
            .get(page)
            .ok_or(anyhow::anyhow!("Page not found"))?;
        let node = headless_chrome::Element::new(&*tab, node.node_id)?;
        let child = node.find_element(&query)?;
        let child = self.resources.insert(AnyNodeRef {
            page_id,
            node_id: child.node_id,
        });
        Ok(Node {
            id: child.index() as u64,
            owned: true,
        })
    }

    pub(crate) fn impl_drop_node(&mut self, rep: Node) -> wasmtime::Result<()> {
        let index = rep.into();
        self.resources.drop_key(index);
        Ok(())
    }
}
