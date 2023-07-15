use fantoccini::elements::Element;
use fantoccini::error::NewSessionError;
use fantoccini::{Client, ClientBuilder, Locator};
use slab::Slab;
use std::time::Duration;
use crate::ElementId;

fn start_process() -> std::process::Child {
    #[cfg(target_os = "macos")]
    const CMD: &str = r#"/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=4444"#;
    #[cfg(target_os = "linux")]
    const CMD: &str = "chromium-browser --remote-debugging-port=4444";
    #[cfg(target_os = "windows")]
    const CMD: &str = "chrome.exe --remote-debugging-port=4444";

    let child = std::process::Command::new("sh")
        .arg("-c")
        .arg(CMD)
        .spawn()
        .expect("failed to start chrome");
    child
}

pub struct Browser {
    client: Client,
    child: std::process::Child,
    elements: Slab<Element>,
}

impl Drop for Browser {
    fn drop(&mut self) {
        self.child.kill().unwrap();
    }
}

impl Browser {
    pub async fn new() -> Result<Self, NewSessionError> {
        let client = ClientBuilder::native()
            .connect("http://localhost:4444")
            .await?;
        let child = start_process();
        Ok(Self {
            client,
            child,
            elements: Slab::new(),
        })
    }

    pub async fn goto(&mut self, url: &str) -> Result<(), wasmtime::Error> {
        self.client.goto(url).await?;
        self.client
            .wait()
            .at_most(Duration::from_secs(5))
            .for_url(url::Url::parse(url)?)
            .await?;
        Ok(())
    }

    pub async fn find(&mut self, selector: &str) -> Result<ElementId, wasmtime::Error> {
        let element: Element = self
            .client
            .wait()
            .at_most(Duration::from_secs(5))
            .every(Duration::from_millis(100))
            .for_element(Locator::Css(selector))
            .await?;

        Ok(ElementId{
            id: self.elements.insert(element) as u32})
    }

    pub async fn get_text(&mut self, id: ElementId) -> Result<String, wasmtime::Error> {
        let element = self.elements.get(id.id as usize).unwrap();
        let text = element.text().await?;
        Ok(text)
    }
    pub async fn click(&mut self, id: ElementId) -> Result<(), wasmtime::Error> {
        let element = self.elements.get(id.id as usize).unwrap();
        element.click().await?;
        Ok(())
    }

    pub async fn send_keys(
        &mut self,
        id: ElementId,
        keys: &str,
    ) -> Result<(), wasmtime::Error> {
        let element = self.elements.get(id.id as usize).unwrap();
        element.send_keys(keys).await?;
        Ok(())
    }

    pub async fn inner_html(
        &mut self,
        id: ElementId,
    ) -> Result<String, wasmtime::Error> {
        let element = self.elements.get(id.id as usize).unwrap();
        let html = element.html(true).await?;
        Ok(html)
    }

    pub async fn outer_html(
        &mut self,
        id: ElementId,
    ) -> Result<String, wasmtime::Error> {
        let element = self.elements.get(id.id as usize).unwrap();
        let html = element.html(false).await?;
        Ok(html)
    }

    pub async fn screenshot(&mut self) -> Result<Vec<u8>, wasmtime::Error> {
        let bytes = self.client.screenshot().await?;
        Ok(bytes)
    }

    pub async fn screenshot_of_id(
        &mut self,
        id: ElementId,
    ) -> Result<Vec<u8>, wasmtime::Error> {
        let element = self.elements.get(id.id as usize).unwrap();
        let bytes = element.screenshot().await?;
        Ok(bytes)
    }

    pub async fn find_child(
        &mut self,
        id: ElementId,
        selector: &str,
    ) -> Result<ElementId, wasmtime::Error> {
        let element = self.elements.get(id.id as usize).unwrap();
        let child = element.find(Locator::Css(selector)).await?;
        Ok(ElementId{
            id:self.elements.insert(child) as u32})
    }

    pub fn drop_element(&mut self, id: ElementId) {
        self.elements.remove(id.id as usize);
    }
}

#[tokio::test]
async fn browse(){
    use crate::browse::Browser;

    let mut browser = Browser::new().await.unwrap();
    let url = "https://www.google.com";
    browser.goto(url).await.unwrap();
    let id = browser.find("input[name=q]").await.unwrap();
    browser.send_keys(id, "rust").await.unwrap();
    browser.click(id).await.unwrap();
    let id = browser.find("#search").await.unwrap();
    
}