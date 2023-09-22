use self::browse::Tab;
use super::document::Document;
pub use crate::context::page::crawl::CrawlFeedback;
use crate::context::page::crawl::Crawler;
pub use crate::context::page::crawl::CrawlingCallback;
use once_cell::sync::OnceCell;
use scraper::{Html, Selector};
use tokio::time::Instant;
use url::Url;

pub mod browse;
mod crawl;

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

    fn new_wait_until(url: Url, mode: BrowserMode, wait_until: Instant) -> anyhow::Result<Self> {
        match mode {
            BrowserMode::Static => Ok(Self::Static(StaticPage::new_wait_until(url, wait_until)?)),
            BrowserMode::Headless => Ok(Self::Dynamic(Tab::new(url, true)?)),
            BrowserMode::Headfull => Ok(Self::Dynamic(Tab::new(url, false)?)),
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

pub async fn get_article(url: Url) -> Result<Document, anyhow::Error> {
    let html = reqwest::get(url.clone()).await?.text().await?;
    extract_article(&html)
}

pub fn extract_article(html: &str) -> anyhow::Result<Document> {
    let cleaned =
        readability::extractor::extract(&mut html.as_bytes(), &Url::parse("https://example.com")?)
            .unwrap();
    Ok(Document::from_parts(cleaned.title, cleaned.text))
}
