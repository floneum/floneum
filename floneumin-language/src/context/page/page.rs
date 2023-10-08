use super::browse::Tab;
use super::{super::document::Document, NodeRef};
use super::{extract_article, Node};
pub use crate::context::page::crawl::CrawlFeedback;
use crate::context::page::crawl::Crawler;
pub use crate::context::page::crawl::CrawlingCallback;
use image::DynamicImage;
use once_cell::sync::OnceCell;
use scraper::{Html, Selector};
use tokio::time::Instant;
use url::Url;

#[derive(Debug, Clone)]
pub enum Page {
    Static(StaticPage),
    Dynamic(Tab),
}

impl Page {
    pub fn new(url: Url, mode: BrowserMode) -> anyhow::Result<Self> {
        match mode {
            BrowserMode::Static => Ok(Self::Static(StaticPage::new(url)?)),
            BrowserMode::Headless => Ok(Self::Dynamic(Tab::new(url, true)?)),
            BrowserMode::Headfull => Ok(Self::Dynamic(Tab::new(url, false)?)),
        }
    }

    pub async fn get_node(&self, node_ref: NodeRef) -> anyhow::Result<Node<'_>> {
        match (self, node_ref) {
            (Self::Static(page), NodeRef::Static(node_id)) => {
                let html = page.html_ref().await?;
                Ok(Node::Static(
                    scraper::ElementRef::wrap(html.tree.get(node_id).ok_or_else(|| {
                        anyhow::anyhow!("Could not find node with id: {:?}", node_id)
                    })?)
                    .ok_or_else(|| anyhow::anyhow!("Could not find node with id: {:?}", node_id))?,
                ))
            }
            (Self::Dynamic(page), NodeRef::Dynamic(node_id)) => Ok(Node::Dynamic(
                headless_chrome::Element::new(&page.inner, node_id)?,
            )),
            _ => Err(anyhow::anyhow!("Invalid node reference")),
        }
    }

    pub async fn select_elements(&self, selector: &str) -> anyhow::Result<Vec<Node<'_>>> {
        match self {
            Self::Static(page) => {
                let selector = Selector::parse(selector).map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok(page
                    .html_ref()
                    .await?
                    .select(&selector)
                    .map(Node::Static)
                    .collect())
            }
            Self::Dynamic(page) => Ok(page
                .inner
                .wait_for_elements(selector)?
                .into_iter()
                .map(Node::Dynamic)
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

    pub fn screenshot(&self) -> anyhow::Result<DynamicImage> {
        match self {
            Self::Static(_) => Err(anyhow::anyhow!("Cannot take screenshot of static page")),
            Self::Dynamic(page) => page.screenshot(),
        }
    }

    pub fn url(&self) -> Url {
        match self {
            Self::Static(page) => page.url().clone(),
            Self::Dynamic(page) => page.url().clone(),
        }
    }

    pub async fn article(&self) -> anyhow::Result<Document> {
        match self {
            Self::Static(page) => page.article().await,
            Self::Dynamic(page) => page.article(),
        }
    }

    pub async fn title(&self) -> Option<String> {
        match self {
            Self::Static(page) => page.title().await,
            Self::Dynamic(page) => page.title(),
        }
    }

    pub async fn html(&self) -> anyhow::Result<Html> {
        match self {
            Self::Static(page) => page.html().await,
            Self::Dynamic(page) => page.html(),
        }
    }

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

    pub async fn crawl(
        start: Url,
        mode: BrowserMode,
        visit: impl CrawlingCallback,
    ) -> anyhow::Result<()> {
        Crawler::new(mode, visit).crawl(start).await
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BrowserMode {
    Static,
    Headless,
    Headfull,
}

#[derive(Debug, Clone)]
pub struct StaticPage {
    wait_until: Instant,
    url: Url,
    html: OnceCell<Html>,
}

impl StaticPage {
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

    pub fn url(&self) -> Url {
        self.url.clone()
    }

    pub async fn html_ref(&self) -> anyhow::Result<&Html> {
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

    pub async fn html(&self) -> anyhow::Result<Html> {
        Ok(self.html_ref().await?.clone())
    }

    pub async fn article(&self) -> anyhow::Result<Document> {
        extract_article(&self.html_ref().await?.html())
    }

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
