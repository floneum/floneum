pub use headless_chrome::protocol::cdp::CSS::CSSComputedStyleProperty;
use headless_chrome::{Browser as HeadlessBrowser, Element, LaunchOptions};
use image::DynamicImage;
use once_cell::sync::Lazy;
use scraper::Html;
use serde::de::DeserializeOwned;
use std::sync::Arc;
use url::Url;

use super::{extract_article, NodeRef};
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

    /// Get a node from the current page.
    pub fn node(&self, node_ref: NodeRef) -> anyhow::Result<Node<'_>> {
        if let NodeRef::Dynamic(node_id) = node_ref {
            Ok(Element::new(&self.inner, node_id)?.into())
        } else {
            anyhow::bail!("NodeRef is not a DynamicNodeRef")
        }
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

impl<'a> From<Element<'a>> for Node<'a> {
    fn from(inner: Element<'a>) -> Self {
        Self { inner }
    }
}

impl Node<'_> {
    /// Get the id of the node.
    pub fn id(&self) -> NodeRef {
        NodeRef::Dynamic(self.inner.node_id)
    }

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

    /// Get the outer HTML of the node filtering out any nodes that are not visible.
    #[tracing::instrument]
    pub fn outer_html_visible(&self) -> Result<String, anyhow::Error> {
        self.call_js_fn(
            r#"
            function(...args) {
                function node_to_html(node) {
                    // If this is a text node, return the text content
                    if (node instanceof Text) {
                        return node.textContent;
                    }

                    if (node instanceof HTMLElement && node.checkVisibility()) {
                        let tag = node.tagName.toLowerCase();
                        let text = "<" + tag;
                        for (const attr of node.attributes) {
                            text += " " + attr.name + "=\"" + attr.value + "\"";
                        }
                        text += ">";
                        if (node.hasChildNodes()) {
                            for (const child of node.childNodes) {
                                text += node_to_html(child);
                            }
                        }
                        text += "</" + tag + ">";
                        return text;
                    }
                    return "";
                }

                return node_to_html(this);
            }
        "#,
            vec![],
            false,
        )
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

    /// Get the name of the element.
    #[tracing::instrument]
    pub fn name(&self) -> &str {
        &self.inner.tag_name
    }

    /// Get the attributes of the element.
    #[tracing::instrument]
    pub fn attributes(&self) -> Result<Vec<(String, String)>, anyhow::Error> {
        let Some(attributes) = self.inner.get_attributes()? else {
            return Ok(Vec::new());
        };
        Ok(attributes
            .into_iter()
            .filter_map(|attribute| {
                let value = self.inner.get_attribute_value(&attribute).ok()??;
                Some((attribute, value))
            })
            .collect())
    }

    /// Try to get all the computed style of the current node. This will return an error if the node is not a dynamic node.
    pub fn computed_style(&self) -> anyhow::Result<Vec<CSSComputedStyleProperty>> {
        self.inner.get_computed_styles()
    }

    /// Try to find out if the current node is visible.
    ///
    /// On static pages, this will always return true. On dynamic pages, this will return true if the node is not hidden.
    pub fn is_visible(&self) -> anyhow::Result<bool> {
        self.call_js_fn(
            "function(...args) { return this.checkVisibility(); }",
            vec![],
            false,
        )
    }

    /// Call a function on the node.
    pub fn call_js_fn<R: DeserializeOwned>(
        &self,
        function: &str,
        args: Vec<serde_json::Value>,
        async_function: bool,
    ) -> anyhow::Result<R> {
        let result = self.inner.call_js_fn(function, args, async_function)?;

        match result.value {
            Some(value) => Ok(serde_json::from_value(value)?),
            None => Err(anyhow::anyhow!("No value returned from function")),
        }
    }

    /// Hover over the node.
    #[tracing::instrument]
    pub fn hover(&self) -> anyhow::Result<()> {
        self.inner.move_mouse_over()?;
        Ok(())
    }

    /// Find the children of the current node.
    pub fn children(&self) -> anyhow::Result<Vec<NodeRef>> {
        let node_info = self.inner.get_description()?;
        let children = node_info.children.unwrap_or_default();
        let children = children.iter().map(|child| {
            let child_id = child.node_id;
            NodeRef::Dynamic(child_id)
        });
        Ok(children.collect())
    }
}
