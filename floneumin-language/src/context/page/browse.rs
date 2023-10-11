use headless_chrome::{Browser as HeadlessBrowser, Element, LaunchOptions};
use image::DynamicImage;
use once_cell::sync::Lazy;
use scraper::Html;
use std::sync::Arc;
use url::Url;

use super::extract_article;
use crate::context::document::Document;

static BROWSER: Browser = Browser::new();

/// A browser that can be used to interact with web pages.
pub(crate) struct Browser {
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
        Self::new()
    }
}

impl Browser {
    /// Create a new browser.
    pub const fn new() -> Self {
        Self {
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
        }
    }

    /// Create a new tab.
    #[tracing::instrument]
    pub fn new_tab(&self, headless: bool) -> Result<Tab, anyhow::Error> {
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

/// A tab that can be used to interact with a web page.
#[derive(Clone)]
pub struct Tab {
    pub(crate) inner: Arc<headless_chrome::Tab>,
}

impl Tab {
    /// Create a new tab.
    pub fn new(url: Url, headless: bool) -> Result<Self, anyhow::Error> {
        let tab = BROWSER.new_tab(headless)?;
        tab.goto(url.as_ref())?;
        Ok(tab)
    }

    /// Get the inner headless chrome tab.
    pub fn inner(&self) -> Arc<headless_chrome::Tab> {
        self.inner.clone()
    }

    /// Go to the given URL.
    #[tracing::instrument]
    pub fn goto(&self, url: &str) -> Result<(), anyhow::Error> {
        self.inner.navigate_to(url)?.wait_until_navigated()?;
        Ok(())
    }

    /// Find the first element matching the given selector.
    #[tracing::instrument]
    pub fn find(&self, selector: &str) -> Result<Node, anyhow::Error> {
        let element = self.inner.wait_for_element(selector)?;

        Ok(Node { inner: element })
    }

    /// Screen shot the current page.
    #[tracing::instrument]
    pub fn screenshot(&self) -> Result<DynamicImage, anyhow::Error> {
        let bytes = self.inner.capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
            None,
            None,
            false,
        )?;
        let image = image::load_from_memory(&bytes)?;
        Ok(image)
    }

    /// Get the URL of the current page.
    pub fn url(&self) -> Url {
        self.inner.get_url().parse().unwrap()
    }

    /// Extract the article from the current page.
    pub fn article(&self) -> anyhow::Result<Document> {
        let html = self.inner.get_content()?;
        extract_article(&html)
    }

    /// Get the title of the current page.
    pub fn title(&self) -> Option<String> {
        self.inner.get_title().ok()
    }

    /// Get the HTML of the current page.
    pub fn html(&self) -> anyhow::Result<Html> {
        Ok(Html::parse_document(&self.inner.get_content()?))
    }
}

impl Drop for Tab {
    #[tracing::instrument]
    fn drop(&mut self) {
        let _ = self.inner.close(true);
    }
}

impl std::fmt::Debug for Tab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tab").finish()
    }
}

/// A node in a [`Tab`]
#[derive(Debug)]
pub struct Node<'a> {
    inner: Element<'a>,
}

impl<'a> Node<'a> {
    /// Get the text of the node.
    #[tracing::instrument]
    pub fn get_text(&self) -> Result<String, anyhow::Error> {
        let text = self.inner.get_inner_text()?;
        Ok(text)
    }

    /// Click the node.
    #[tracing::instrument]
    pub fn click(&self) -> Result<(), anyhow::Error> {
        self.inner.click()?;
        Ok(())
    }

    /// Type the given keys into the node.
    #[tracing::instrument]
    pub fn send_keys(&self, keys: &str) -> Result<(), anyhow::Error> {
        self.inner.type_into(keys)?;
        Ok(())
    }

    /// Get the outer HTML of the node.
    #[tracing::instrument]
    pub fn outer_html(&self) -> Result<String, anyhow::Error> {
        let html = self.inner.get_content()?;
        Ok(html)
    }

    /// Screen shot the node.
    #[tracing::instrument]
    pub fn screenshot(&self) -> Result<DynamicImage, anyhow::Error> {
        let bytes = self.inner.capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Jpeg,
        )?;
        let image = image::load_from_memory(&bytes)?;
        Ok(image)
    }

    /// Find the first child matching the given selector.
    #[tracing::instrument]
    pub fn find_child(&self, selector: &str) -> Result<Self, anyhow::Error> {
        let child = self.inner.find_element(selector)?;
        Ok(Self { inner: child })
    }
}
