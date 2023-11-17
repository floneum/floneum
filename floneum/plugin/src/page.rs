use crate::host::{AnyNodeRef, State};
use crate::plugins::main;

use crate::plugins::main::types::{Node, Page};

use kalosm::language::Tab;
use wasmtime::component::__internal::async_trait;

#[async_trait]
impl main::types::HostPage for State {
    async fn new(
        &mut self,
        mode: main::types::BrowserMode,
        url: String,
    ) -> wasmtime::Result<wasmtime::component::Resource<Page>> {
        let page = Tab::new(
            url.parse()?,
            matches!(mode, main::types::BrowserMode::Headless),
        )?;
        let page_id = self.pages.insert(page.inner());
        Ok(wasmtime::component::Resource::new_own(page_id as u32))
    }

    async fn find_in_current_page(
        &mut self,
        self_: wasmtime::component::Resource<Page>,
        query: String,
    ) -> wasmtime::Result<wasmtime::component::Resource<Node>> {
        let page = self
            .pages
            .get(self_.rep() as usize)
            .ok_or(anyhow::anyhow!("Page not found"))?;
        let node = page.find_element(&query)?;
        let node_id = node.node_id;
        let node = AnyNodeRef {
            node_id: node_id as u32,
            page_id: self_.rep() as usize,
        };
        let node_id = self.nodes.insert(node);
        Ok(wasmtime::component::Resource::new_own(node_id as u32))
    }

    async fn screenshot_browser(
        &mut self,
        self_: wasmtime::component::Resource<Page>,
    ) -> wasmtime::Result<Vec<u8>> {
        let page = self
            .pages
            .get(self_.rep() as usize)
            .ok_or(anyhow::anyhow!("Page not found"))?;
        let bytes = page.capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
            None,
            None,
            false,
        )?;
        Ok(bytes)
    }

    async fn html(&mut self, self_: wasmtime::component::Resource<Page>) -> wasmtime::Result<String> {
        let page = self
            .pages
            .get(self_.rep() as usize)
            .ok_or(anyhow::anyhow!("Page not found"))?;
        Ok(page.get_content()?)
    }

    fn drop(&mut self, rep: wasmtime::component::Resource<Page>) -> wasmtime::Result<()> {
        self.pages.remove(rep.rep() as usize);
        Ok(())
    }
}
