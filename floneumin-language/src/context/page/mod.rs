use self::browse::Tab;
use super::document::Document;
use async_recursion::async_recursion;
use once_cell::unsync::OnceCell;
use scraper::{Html, Selector};
use std::future::Future;
use std::pin::Pin;
use url::Url;

pub mod browse;

#[derive(Debug, Clone)]
pub enum Page {
    Static(StaticPage),
    Dynamic(Tab),
}

impl Page {
    pub async fn new(url: Url, headless: bool, headfull: bool) -> anyhow::Result<Self> {
        if headless {
            Ok(Self::Dynamic(Tab::new(url, !headfull)?))
        } else {
            Ok(Self::Static(StaticPage::new(url).await?))
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

    #[async_recursion(?Send)]
    async fn crawl_inner<'a>(
        &self,
        visit: &mut (impl FnMut(Self) -> Pin<Box<dyn Future<Output = bool>>> + 'async_recursion),
        headless: bool,
        headfull: bool,
    ) -> anyhow::Result<()> {
        if !visit(self.clone()).await {
            return Ok(());
        }
        let links = self.links().await?;
        for link in links {
            let tab = Self::new(link, headless, headfull).await?;
            tab.crawl_inner(visit, headless, headfull).await?;
        }
        Ok(())
    }

    pub async fn crawl(
        &self,
        mut visit: impl FnMut(Self) -> Pin<Box<dyn Future<Output = bool>>>,
        headless: bool,
        headfull: bool,
    ) -> anyhow::Result<()> {
        self.crawl_inner(&mut visit, headless, headfull).await
    }
}

#[derive(Debug, Clone)]
pub struct StaticPage {
    url: Url,
    html: OnceCell<Html>,
}

impl StaticPage {
    pub async fn new(url: Url) -> anyhow::Result<Self> {
        Ok(Self {
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
