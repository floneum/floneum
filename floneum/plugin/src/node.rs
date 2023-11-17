use crate::host::{AnyNodeRef, State};
use crate::plugins::main;

use crate::plugins::main::types::Node;

use wasmtime::component::__internal::async_trait;

impl State {
    pub fn get_node(
        &self,
        node: wasmtime::component::Resource<Node>,
    ) -> anyhow::Result<headless_chrome::Element<'_>> {
        let node = self
            .nodes
            .get(node.rep() as usize)
            .ok_or(anyhow::anyhow!("Node not found"))?;
        let tab = self
            .pages
            .get(node.page_id)
            .ok_or(anyhow::anyhow!("Page not found"))?;
        headless_chrome::Element::new(tab, node.node_id)
    }
}

#[async_trait]
impl main::types::HostNode for State {
    async fn get_element_text(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
    ) -> wasmtime::Result<String> {
        let node = self.get_node(self_)?;
        Ok(node.get_inner_text()?)
    }

    async fn click_element(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
    ) -> wasmtime::Result<()> {
        let node = self.get_node(self_)?;
        node.click()?;
        Ok(())
    }

    async fn type_into_element(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
        keys: String,
    ) -> wasmtime::Result<()> {
        let node = self.get_node(self_)?;
        node.type_into(&keys)?;
        Ok(())
    }

    async fn get_element_outer_html(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
    ) -> wasmtime::Result<String> {
        let node = self.get_node(self_)?;
        Ok(node.get_content()?)
    }

    async fn screenshot_element(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
    ) -> wasmtime::Result<Vec<u8>> {
        let node = self.get_node(self_)?;
        Ok(node.capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
        )?)
    }

    async fn find_child_of_element(
        &mut self,
        self_: wasmtime::component::Resource<Node>,
        query: String,
    ) -> wasmtime::Result<wasmtime::component::Resource<Node>> {
        let node = self
            .nodes
            .get(self_.rep() as usize)
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
        Ok(wasmtime::component::Resource::new_own(child as u32))
    }

    fn drop(&mut self, rep: wasmtime::component::Resource<Node>) -> wasmtime::Result<()> {
        self.nodes.remove(rep.rep() as usize);
        Ok(())
    }
}
