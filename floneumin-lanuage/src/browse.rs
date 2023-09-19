use headless_chrome::{Browser as HeadlessBrowser, Element, LaunchOptions};
use once_cell::sync::Lazy;
use std::sync::Arc;

pub struct Browser {
    headless_client: Lazy<Result<HeadlessBrowser, String>>,
    headfull_client: Lazy<Result<HeadlessBrowser, String>>,
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
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
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
    pub fn new_tab(&mut self, headless: bool) -> Result<Tab, anyhow::Error> {
        let client = if headless {
            &self.headless_client
        } else {
            &self.headfull_client
        };
        let browser = client.as_ref().map_err(|err| anyhow::anyhow!("{}", err))?;
        let tab = browser.new_tab()?;

        Ok(Tab { inner: tab })
    }
}

struct Tab {
    inner: Arc<headless_chrome::Tab>,
}

impl Tab {
    #[tracing::instrument]
    pub fn goto(&self, url: &str) -> Result<(), anyhow::Error> {
        self.inner.navigate_to(url)?.wait_until_navigated()?;
        Ok(())
    }

    #[tracing::instrument]
    pub fn find(&self, selector: &str) -> Result<Node, anyhow::Error> {
        let element = self.inner.wait_for_element(selector)?;

        Ok(Node { inner: element })
    }

    #[tracing::instrument]
    pub fn screenshot(&self) -> Result<Vec<u8>, anyhow::Error> {
        let bytes = self.inner.capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
            None,
            None,
            false,
        )?;
        Ok(bytes)
    }
}

impl Drop for Tab {
    #[tracing::instrument]
    fn drop(&mut self) {
        self.inner.close(true).unwrap();
    }
}

impl std::fmt::Debug for Tab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tab").finish()
    }
}

#[derive(Debug)]
pub struct Node<'a> {
    inner: Element<'a>,
}

impl<'a> Node<'a> {
    #[tracing::instrument]
    pub fn get_text(&self) -> Result<String, anyhow::Error> {
        let text = self.inner.get_inner_text()?;
        Ok(text)
    }

    #[tracing::instrument]
    pub fn click(&self) -> Result<(), anyhow::Error> {
        self.inner.click()?;
        Ok(())
    }

    #[tracing::instrument]
    pub fn send_keys(&self, keys: &str) -> Result<(), anyhow::Error> {
        self.inner.type_into(keys)?;
        Ok(())
    }

    #[tracing::instrument]
    pub fn outer_html(&self) -> Result<String, anyhow::Error> {
        let html = self.inner.get_content()?;
        Ok(html)
    }

    #[tracing::instrument]
    pub fn screenshot_of_id(&self) -> Result<Vec<u8>, anyhow::Error> {
        let bytes = self.inner.capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
        )?;
        Ok(bytes)
    }

    #[tracing::instrument]
    pub fn find_child(&self, selector: &str) -> Result<Self, anyhow::Error> {
        let child = self.inner.find_element(selector)?;
        Ok(Self { inner: child })
    }
}
