use crate::NodeId;
use headless_chrome::{Browser as HeadlessBrowser, Element, LaunchOptions};
use std::sync::Arc;

pub struct Browser {
    #[allow(unused)]
    client: HeadlessBrowser,
    tab: Arc<headless_chrome::Tab>,
}

impl Browser {
    pub fn new() -> wasmtime::Result<Self> {
        let browser = HeadlessBrowser::new(
            LaunchOptions::default_builder()
                .build()
                .expect("Could not find chrome-executable"),
        )?;
        let tab = browser.new_tab()?;

        Ok(Self {
            tab,
            client: browser,
        })
    }

    fn get_node(&self, node: NodeId) -> Result<Element, wasmtime::Error> {
        Element::new(&self.tab, node.id)
    }

    pub fn goto(&mut self, url: &str) -> Result<(), wasmtime::Error> {
        self.tab.navigate_to(url)?.wait_until_navigated()?;
        Ok(())
    }

    pub fn find(&mut self, selector: &str) -> Result<NodeId, wasmtime::Error> {
        let element = self.tab.wait_for_element(selector)?.node_id;

        Ok(NodeId { id: element })
    }

    pub fn get_text(&mut self, id: NodeId) -> Result<String, wasmtime::Error> {
        let element = self.get_node(id)?;
        let text = element.get_inner_text()?;
        Ok(text)
    }

    pub fn click(&mut self, id: NodeId) -> Result<(), wasmtime::Error> {
        let element = self.get_node(id)?;
        element.click()?;
        Ok(())
    }

    pub fn send_keys(&mut self, id: NodeId, keys: &str) -> Result<(), wasmtime::Error> {
        let element = self.get_node(id)?;
        element.type_into(keys)?;
        Ok(())
    }

    pub fn outer_html(&mut self, id: NodeId) -> Result<String, wasmtime::Error> {
        let element = self.get_node(id)?;
        let html = element.get_content()?;
        Ok(html)
    }

    pub fn screenshot(&mut self) -> Result<Vec<u8>, wasmtime::Error> {
        let bytes = self.tab.capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
            None,
            None,
            false,
        )?;
        Ok(bytes)
    }

    pub fn screenshot_of_id(&mut self, id: NodeId) -> Result<Vec<u8>, wasmtime::Error> {
        let element = self.get_node(id)?;
        let bytes = element.capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
        )?;
        Ok(bytes)
    }

    pub fn find_child(&mut self, id: NodeId, selector: &str) -> Result<NodeId, wasmtime::Error> {
        let element = self.get_node(id)?;
        let child = element.find_element(selector)?;
        Ok(NodeId { id: child.node_id })
    }
}

#[test]
fn browse() {
    let mut browser = Browser::new().unwrap();
    browser.goto("https://www.rust-lang.org/learn").unwrap();
    let id = browser.find("h1").unwrap();
    let text = browser.get_text(id).unwrap();
    assert_eq!(text, "Learn Rust");
}
