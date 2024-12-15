use super::browse::Tab;
use super::{super::document::Document, NodeRef};
use super::{extract_article, AnyNode, ExtractDocumentError};
use crate::context::page::crawl::Crawler;
pub use crate::context::page::crawl::CrawlingCallback;
use image::DynamicImage;
use once_cell::sync::OnceCell;
use scraper::{Html, Selector};
use tokio::time::Instant;
use url::Url;

/// A page that is either static or dynamic.
///
/// # Example
///
/// ```rust, no_run
/// use kalosm_language::prelude::*;
///
/// #[tokio::main]
/// async fn main() {
///     let page = Page::new(
///         Url::parse("https://www.nytimes.com/live/2023/09/21/world/zelensky-russia-ukraine-news").unwrap(),
///         BrowserMode::Static,
///     ).unwrap();
///     let document = page.article().await.unwrap();
///     println!("Title: {}", document.title());
///     println!("Body: {}", document.body());
/// }
/// ```
#[derive(Debug, Clone)]
pub enum Page {
    /// A page of static HTML.
    Static(StaticPage),
    /// A page in a headless browser.
    Dynamic(Tab),
}

impl Page {
    /// Create a new page at the given URL.
    pub fn new(url: Url, mode: BrowserMode) -> anyhow::Result<Self> {
        match mode {
            BrowserMode::Static => Ok(Self::Static(StaticPage::new(url)?)),
            BrowserMode::Headless => Ok(Self::Dynamic(Tab::new(url, true)?)),
            BrowserMode::Headfull => Ok(Self::Dynamic(Tab::new(url, false)?)),
        }
    }

    /// Get the node with the given ID.
    pub async fn get_node(&self, node_ref: NodeRef) -> anyhow::Result<AnyNode<'_>> {
        match (self, node_ref) {
            (Self::Static(page), NodeRef::Static(node_id)) => {
                let html = page.html_ref().await?;
                Ok(AnyNode::Static(
                    scraper::ElementRef::wrap(html.tree.get(node_id).ok_or_else(|| {
                        anyhow::anyhow!("Could not find node with id: {:?}", node_id)
                    })?)
                    .ok_or_else(|| anyhow::anyhow!("Could not find node with id: {:?}", node_id))?,
                ))
            }
            (Self::Dynamic(page), NodeRef::Dynamic(node_id)) => Ok(AnyNode::Dynamic(
                headless_chrome::Element::new(&page.inner, node_id)?.into(),
            )),
            _ => Err(anyhow::anyhow!("Invalid node reference")),
        }
    }

    /// Find all elements matching the given selector.
    pub async fn select_elements(&self, selector: &str) -> anyhow::Result<Vec<AnyNode<'_>>> {
        match self {
            Self::Static(page) => {
                let selector = Selector::parse(selector).map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok(page
                    .html_ref()
                    .await?
                    .select(&selector)
                    .map(AnyNode::Static)
                    .collect())
            }
            Self::Dynamic(page) => Ok(page
                .inner
                .wait_for_elements(selector)?
                .into_iter()
                .map(|node| AnyNode::Dynamic(node.into()))
                .collect()),
        }
    }

    pub(crate) fn new_wait_until(
        url: Url,
        mode: BrowserMode,
        wait_until: Instant,
    ) -> anyhow::Result<Self> {
        match mode {
            BrowserMode::Static => Ok(Self::Static(StaticPage::new_wait_until(url, wait_until)?)),
            BrowserMode::Headless => Ok(Self::Dynamic(Tab::new(url, true)?)),
            BrowserMode::Headfull => Ok(Self::Dynamic(Tab::new(url, false)?)),
        }
    }

    /// Take a screenshot of the page if it is in a headless browser.
    pub fn screenshot(&self) -> anyhow::Result<DynamicImage> {
        match self {
            Self::Static(_) => Err(anyhow::anyhow!("Cannot take screenshot of static page")),
            Self::Dynamic(page) => page.screenshot(),
        }
    }

    /// Get the URL of the page.
    pub fn url(&self) -> Url {
        match self {
            Self::Static(page) => page.url().clone(),
            Self::Dynamic(page) => page.url().clone(),
        }
    }

    /// Extract the article from the page.
    pub async fn article(&self) -> anyhow::Result<Document> {
        match self {
            Self::Static(page) => Ok(page.article().await?),
            Self::Dynamic(page) => page.article(),
        }
    }

    /// Get the title of the page.
    pub async fn title(&self) -> Option<String> {
        match self {
            Self::Static(page) => page.title().await,
            Self::Dynamic(page) => page.title(),
        }
    }

    /// Get the HTML of the page.
    pub async fn html(&self) -> anyhow::Result<Html> {
        match self {
            Self::Static(page) => page.html().await,
            Self::Dynamic(page) => page.html(),
        }
    }

    /// Get all the links from the page.
    pub async fn links(&self) -> anyhow::Result<Vec<Url>> {
        let mut links: Vec<_> = self
            .html()
            .await?
            .select(&Selector::parse("a").unwrap())
            .filter_map(|e| {
                let href = e.value().attr("href")?;
                let url = self.url().join(href).ok()?;
                Some(url)
            })
            .collect();

        links.sort();
        links.dedup();

        Ok(links)
    }

    /// Start crawling from this page.
    pub async fn crawl(start: Url, mode: BrowserMode, visit: impl CrawlingCallback) {
        Crawler::new(mode, visit).crawl(start).await
    }
}

/// The mode of the browser.
#[derive(Debug, Clone, Copy)]
pub enum BrowserMode {
    /// A static browser that just downloads the HTML.
    Static,
    /// A headless browser.
    Headless,
    /// A browser with a visible GUI.
    Headfull,
}

/// A static page that lazily fetches the HTML from a page.
#[derive(Debug, Clone)]
pub struct StaticPage {
    wait_until: Instant,
    url: Url,
    html: OnceCell<Html>,
}

impl StaticPage {
    /// Create a new static page at the given URL.
    pub fn new(url: Url) -> anyhow::Result<Self> {
        Ok(Self {
            wait_until: Instant::now(),
            url: url.clone(),
            html: OnceCell::new(),
        })
    }

    fn new_wait_until(url: Url, wait_until: Instant) -> anyhow::Result<Self> {
        Ok(Self {
            wait_until,
            url: url.clone(),
            html: OnceCell::new(),
        })
    }

    /// Get the URL of the page.
    pub fn url(&self) -> Url {
        self.url.clone()
    }

    /// Get the HTML of the page.
    pub async fn html_ref(&self) -> Result<&Html, reqwest::Error> {
        match self.html.get() {
            Some(html) => Ok(html),
            None => {
                tokio::time::sleep_until(self.wait_until).await;
                let html = reqwest::get(self.url.clone()).await?.text().await?;
                let html = Html::parse_document(&html);
                self.html.set(html).unwrap();
                Ok(self.html.get().unwrap())
            }
        }
    }

    /// Get the HTML of the page.
    pub async fn html(&self) -> anyhow::Result<Html> {
        Ok(self.html_ref().await?.clone())
    }

    /// Extract the article from the page.
    pub async fn article(&self) -> Result<Document, ExtractDocumentError> {
        extract_article(&self.html_ref().await?.html())
    }

    /// Get the title of the page.
    pub async fn title(&self) -> Option<String> {
        let selector = Selector::parse("title").ok()?;
        self.html_ref()
            .await
            .ok()?
            .select(&selector)
            .next()
            .map(|e| e.inner_html())
    }
}
