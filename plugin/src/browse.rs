use crate::{NodeId, TabId};
use headless_chrome::{Browser as HeadlessBrowser, Element, LaunchOptions};
use once_cell::unsync::Lazy;
use slab::Slab;
use std::sync::Arc;

pub struct Browser {
    headless_client: Lazy<Result<HeadlessBrowser, String>>,
    headfull_client: Lazy<Result<HeadlessBrowser, String>>,
    tabs: Slab<Arc<headless_chrome::Tab>>,
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

    pub fn get_tab(&self, tab: TabId) -> Result<Arc<headless_chrome::Tab>, wasmtime::Error> {
        let tab = self
            .tabs
            .get(tab.id as usize)
            .ok_or_else(|| anyhow::anyhow!("Tab not found"))?;
        Ok(tab.clone())
    }

    fn get_node(&self, tab: TabId, node: NodeId) -> Result<Element, wasmtime::Error> {
        Element::new(
            self.tabs
                .get(tab.id as usize)
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Tab not found"))?,
            node.id,
        )
    }

    pub fn goto(&mut self, tab: TabId, url: &str) -> Result<(), wasmtime::Error> {
        self.get_tab(tab)?
            .navigate_to(url)?
            .wait_until_navigated()?;
        Ok(())
    }

    pub fn find(&mut self, tab: TabId, selector: &str) -> Result<NodeId, wasmtime::Error> {
        let element = self.get_tab(tab)?.wait_for_element(selector)?.node_id;

        Ok(NodeId { id: element })
    }

    pub fn get_text(&mut self, tab: TabId, id: NodeId) -> Result<String, wasmtime::Error> {
        let element = self.get_node(tab, id)?;
        let text = element.get_inner_text()?;
        Ok(text)
    }

    pub fn click(&mut self, tab: TabId, id: NodeId) -> Result<(), wasmtime::Error> {
        let element = self.get_node(tab, id)?;
        element.click()?;
        Ok(())
    }

    pub fn send_keys(&mut self, tab: TabId, id: NodeId, keys: &str) -> Result<(), wasmtime::Error> {
        let element = self.get_node(tab, id)?;
        element.type_into(keys)?;
        Ok(())
    }

    pub fn outer_html(&mut self, tab: TabId, id: NodeId) -> Result<String, wasmtime::Error> {
        let element = self.get_node(tab, id)?;
        let html = element.get_content()?;
        Ok(html)
    }

    pub fn screenshot(&mut self, tab: TabId) -> Result<Vec<u8>, wasmtime::Error> {
        let bytes = self.get_tab(tab)?.capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
            None,
            None,
            false,
        )?;
        Ok(bytes)
    }

    pub fn screenshot_of_id(&mut self, tab: TabId, id: NodeId) -> Result<Vec<u8>, wasmtime::Error> {
        let element = self.get_node(tab, id)?;
        let bytes = element.capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
        )?;
        Ok(bytes)
    }

    pub fn find_child(
        &mut self,
        tab: TabId,
        id: NodeId,
        selector: &str,
    ) -> Result<NodeId, wasmtime::Error> {
        let element = self.get_node(tab, id)?;
        let child = element.find_element(selector)?;
        Ok(NodeId { id: child.node_id })
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
    let text = browser.get_text(tab, id).unwrap();
    assert_eq!(text, "Learn Rust");
}
