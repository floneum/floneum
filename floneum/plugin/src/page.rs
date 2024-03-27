use crate::host::{AnyNodeRef, State};
use crate::plugins::main;

use crate::plugins::main::types::{Node, Page};

use kalosm::language::Tab;

impl State {
    pub(crate) fn impl_create_page(
        &mut self,
        mode: main::types::BrowserMode,
        url: String,
    ) -> wasmtime::Result<Page> {
        let page = Tab::new(
            url.parse()?,
            matches!(mode, main::types::BrowserMode::Headless),
        )?;
        let page_id = self.resources.insert(page.inner());
        Ok(Page {
            id: page_id.index() as u64,
            owned: true,
        })
    }

    pub(crate) async fn impl_find_in_current_page(
        &mut self,
        self_: Page,
        query: String,
    ) -> wasmtime::Result<Node> {
        let index = self_.into();
        let page = self
            .resources
            .get(index)
            .ok_or(anyhow::anyhow!("Page not found"))?;
        let node = page.find_element(&query)?;
        let node_id = node.node_id;
        let node = AnyNodeRef {
            node_id: node_id as u32,
            page_id: self_.id as usize,
        };
        let node_id = self.resources.insert(node);
        Ok(Node {
            id: node_id.index() as u64,
            owned: true,
        })
    }

    pub(crate) async fn impl_screenshot_browser(
        &mut self,
        self_: Page,
    ) -> wasmtime::Result<Vec<u8>> {
        let index = self_.into();
        let page = self
            .resources
            .get(index)
            .ok_or(anyhow::anyhow!("Page not found"))?;
        let bytes = page.capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
            None,
            None,
            false,
        )?;
        Ok(bytes)
    }

    pub(crate) async fn impl_page_html(&mut self, self_: Page) -> wasmtime::Result<String> {
        let index = self_.into();
        let page = self
            .resources
            .get(index)
            .ok_or(anyhow::anyhow!("Page not found"))?;
        Ok(page.get_content()?)
    }

    pub(crate) fn impl_drop_page(&mut self, rep: Page) -> wasmtime::Result<()> {
        let index = rep.into();
        self.resources.drop_key(index);
        Ok(())
    }
}
