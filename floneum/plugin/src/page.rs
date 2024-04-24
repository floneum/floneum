use crate::host::AnyNodeRef;
use crate::plugins::main;

use crate::plugins::main::types::{NodeResource, PageResource};
use crate::resource::ResourceStorage;

use kalosm::language::Tab;

impl ResourceStorage {
    pub(crate) fn impl_create_page(
        &self,
        mode: main::types::BrowserMode,
        url: String,
    ) -> wasmtime::Result<PageResource> {
        let page = Tab::new(
            url.parse()?,
            matches!(mode, main::types::BrowserMode::Headless),
        )?;
        let page_id = self.insert(page.inner());
        Ok(PageResource {
            id: page_id.index() as u64,
            owned: true,
        })
    }

    pub(crate) async fn impl_find_in_current_page(
        &self,
        self_: PageResource,
        query: String,
    ) -> wasmtime::Result<NodeResource> {
        let node_id = {
            let index = self_.into();
            let page = self.get(index).ok_or(anyhow::anyhow!("Page not found"))?;
            let node = page.find(&query)?;
            node.into_inner().node_id
        };
        let node = AnyNodeRef {
            node_id: node_id as u32,
            page_id: self_.id as usize,
        };
        let node_id = self.insert(node);
        Ok(NodeResource {
            id: node_id.index() as u64,
            owned: true,
        })
    }

    pub(crate) async fn impl_screenshot_browser(&self, self_: PageResource) -> wasmtime::Result<Vec<u8>> {
        let index = self_.into();
        let page = self.get(index).ok_or(anyhow::anyhow!("Page not found"))?;
        let bytes = page.screenshot()?;
        Ok(bytes.into_bytes())
    }

    pub(crate) async fn impl_page_html(&self, self_: PageResource) -> wasmtime::Result<String> {
        let index = self_.into();
        let page = self.get(index).ok_or(anyhow::anyhow!("Page not found"))?;
        page.html().map(|html| html.html())
    }

    pub(crate) fn impl_drop_page(&self, rep: PageResource) -> wasmtime::Result<()> {
        let index = rep.into();
        self.drop_key(index);
        Ok(())
    }
}
