use crate::plugins::main::imports::{NodeId, TabId};
use headless_chrome::{Browser as HeadlessBrowser, Element, LaunchOptions};
use once_cell::sync::Lazy;
use slab::Slab;
use std::sync::Arc;

pub struct Browser {
    headless_client: Lazy<Result<HeadlessBrowser, String>>,
    headfull_client: Lazy<Result<HeadlessBrowser, String>>,
    tabs: Slab<Arc<headless_chrome::Tab>>,
}

impl std::fmt::Debug for Browser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Browser").finish()
    }
}

impl Default for Browser {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl Browser {
    pub fn new() -> wasmtime::Result<Self> {
        Ok(Self {
            tabs: Slab::new(),
            headless_client: Lazy::new(|| {
                let browser = HeadlessBrowser::new(
                    LaunchOptions::default_builder()
                        .headless(true)
                        .build()
                        .expect("Could not find chrome-executable"),
                )
                .map_err(|err| err.to_string())?;
                Ok(browser)
            }),
            headfull_client: Lazy::new(|| {
                let browser = HeadlessBrowser::new(
                    LaunchOptions::default_builder()
                        .headless(false)
                        .build()
                        .expect("Could not find chrome-executable"),
                )
                .map_err(|err| err.to_string())?;
                Ok(browser)
            }),
        })
    }

    #[tracing::instrument]
    pub fn new_tab(&mut self, headless: bool) -> Result<TabId, wasmtime::Error> {
        let client = if headless {
            &self.headless_client
        } else {
            &self.headfull_client
        };
        let browser = client.as_ref().map_err(|err| anyhow::anyhow!("{}", err))?;
        let tab = browser.new_tab()?;
        let id = self.tabs.insert(tab);
        Ok(TabId { id: id as u32 })
    }

    #[tracing::instrument]
    pub fn remove_tab(&mut self, tab: TabId) {
        let tab = self.tabs.remove(tab.id as usize);
        tab.close(true).unwrap();
    }

    #[tracing::instrument]
    pub fn get_tab(&self, tab: TabId) -> Result<Arc<headless_chrome::Tab>, wasmtime::Error> {
        let tab = self
            .tabs
            .get(tab.id as usize)
            .ok_or_else(|| anyhow::anyhow!("Tab not found"))?;
        Ok(tab.clone())
    }

    #[tracing::instrument]
    fn get_node(&self, node: NodeId) -> Result<Element, wasmtime::Error> {
        Element::new(
            self.tabs
                .get(node.tab.id as usize)
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Tab not found"))?,
            node.id,
        )
    }

    #[tracing::instrument]
    pub fn goto(&mut self, tab: TabId, url: &str) -> Result<(), wasmtime::Error> {
        pub fn goto_inner(
            browser: &mut Browser,
            tab: TabId,
            url: &str,
        ) -> Result<(), wasmtime::Error> {
            browser
                .get_tab(tab)?
                .navigate_to(url)?
                .wait_until_navigated()?;
            Ok(())
        }
        match goto_inner(self, tab, url) {
            Ok(()) => Ok(()),
            Err(err) => {
                log::error!("Error: {:?}", err);
                Err(err)
            }
        }
    }

    #[tracing::instrument]
    pub fn find(&mut self, tab: TabId, selector: &str) -> Result<NodeId, wasmtime::Error> {
        let element = self.get_tab(tab)?.wait_for_element(selector)?.node_id;

        Ok(NodeId {
            tab: tab,
            id: element,
        })
    }

    #[tracing::instrument]
    pub fn get_text(&mut self, id: NodeId) -> Result<String, wasmtime::Error> {
        let element = self.get_node(id)?;
        let text = element.get_inner_text()?;
        Ok(text)
    }

    #[tracing::instrument]
    pub fn click(&mut self, id: NodeId) -> Result<(), wasmtime::Error> {
        let element = self.get_node(id)?;
        element.click()?;
        Ok(())
    }

    #[tracing::instrument]
    pub fn send_keys(&mut self, id: NodeId, keys: &str) -> Result<(), wasmtime::Error> {
        let element = self.get_node(id)?;
        element.type_into(keys)?;
        Ok(())
    }

    #[tracing::instrument]
    pub fn outer_html(&mut self, id: NodeId) -> Result<String, wasmtime::Error> {
        let element = self.get_node(id)?;
        let html = element.get_content()?;
        Ok(html)
    }

    #[tracing::instrument]
    pub fn screenshot(&mut self, tab: TabId) -> Result<Vec<u8>, wasmtime::Error> {
        let bytes = self.get_tab(tab)?.capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
            None,
            None,
            false,
        )?;
        Ok(bytes)
    }

    #[tracing::instrument]
    pub fn screenshot_of_id(&mut self, id: NodeId) -> Result<Vec<u8>, wasmtime::Error> {
        let element = self.get_node(id)?;
        let bytes = element.capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
        )?;
        Ok(bytes)
    }

    #[tracing::instrument]
    pub fn find_child(&mut self, id: NodeId, selector: &str) -> Result<NodeId, wasmtime::Error> {
        let element = self.get_node(id)?;
        let child = element.find_element(selector)?;
        Ok(NodeId {
            tab: id.tab,
            id: child.node_id,
        })
    }
}

#[test]
fn browse() {
    let mut browser = Browser::new().unwrap();
    let tab = browser.new_tab(true).unwrap();
    browser
        .goto(tab, "https://www.rust-lang.org/learn")
        .unwrap();
    let id = browser.find(tab, "h1").unwrap();
    let text = browser.get_text(id).unwrap();
    assert_eq!(text, "Learn Rust");
}
