use crate::host::{AnyNodeRef, State};

use crate::plugins::main::types::Node;

impl State {
    pub fn get_node(&self, node: Node) -> anyhow::Result<headless_chrome::Element> {
        let node = self
            .nodes
            .get(node.id as usize)
            .ok_or(anyhow::anyhow!("Node not found"))?;
        let tab = self
            .pages
            .get(node.page_id)
            .ok_or(anyhow::anyhow!("Page not found"))?;
        headless_chrome::Element::new(tab, node.node_id)
    }
}

impl State {
    pub(crate) async fn impl_get_element_text(&mut self, self_: Node) -> wasmtime::Result<String> {
        let node = self.get_node(self_)?;
        node.get_inner_text()
    }

    pub(crate) async fn impl_click_element(&mut self, self_: Node) -> wasmtime::Result<()> {
        let node = self.get_node(self_)?;
        node.click()?;
        Ok(())
    }

    pub(crate) async fn impl_type_into_element(
        &mut self,
        self_: Node,
        keys: String,
    ) -> wasmtime::Result<()> {
        let node = self.get_node(self_)?;
        node.type_into(&keys)?;
        Ok(())
    }

    pub(crate) async fn impl_get_element_outer_html(
        &mut self,
        self_: Node,
    ) -> wasmtime::Result<String> {
        let node = self.get_node(self_)?;
        Ok(node.get_content()?)
    }

    pub(crate) async fn impl_screenshot_element(
        &mut self,
        self_: Node,
    ) -> wasmtime::Result<Vec<u8>> {
        let node = self.get_node(self_)?;
        Ok(node.capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
        )?)
    }

    pub(crate) async fn impl_find_child_of_element(
        &mut self,
        self_: Node,
        query: String,
    ) -> wasmtime::Result<Node> {
        let node = self
            .nodes
            .get(self_.id as usize)
            .ok_or(anyhow::anyhow!("Node not found"))?;
        let page_id = node.page_id;
        let tab = self
            .pages
            .get(node.page_id)
            .ok_or(anyhow::anyhow!("Page not found"))?;
        let node = headless_chrome::Element::new(tab, node.node_id)?;
        let child = node.find_element(&query)?;
        let child = self.nodes.insert(AnyNodeRef {
            page_id,
            node_id: child.node_id,
        });
        Ok(Node {
            id: child as u64,
            owned: true,
        })
    }

    pub(crate) fn impl_drop_node(&mut self, rep: Node) -> wasmtime::Result<()> {
        self.nodes.remove(rep.id as usize);
        Ok(())
    }
}
