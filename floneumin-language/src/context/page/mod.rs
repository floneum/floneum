use self::browse::Tab;
use super::document::Document;
use async_recursion::async_recursion;
use scraper::{Html, Selector};
use url::Url;

pub mod browse;

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

    pub fn article(&self) -> anyhow::Result<Document> {
        match self {
            Self::Static(page) => page.article(),
            Self::Dynamic(page) => page.article(),
        }
    }

    pub fn title(&self) -> Option<String> {
        match self {
            Self::Static(page) => page.title(),
            Self::Dynamic(page) => page.title(),
        }
    }

    pub fn html(&self) -> anyhow::Result<Html> {
        match self {
            Self::Static(page) => Ok(page.html()),
            Self::Dynamic(page) => page.html(),
        }
    }

    pub fn links(&self) -> anyhow::Result<Vec<Url>> {
        let mut links: Vec<_> = self
            .html()?
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
    async fn crawl_inner(
        &self,
        visit: &mut (impl FnMut(&Self) -> bool + 'async_recursion),
        headless: bool,
        headfull: bool,
    ) -> anyhow::Result<()> {
        if !visit(self) {
            return Ok(());
        }
        let links = self.links()?;
        for link in links {
            let tab = Self::new(link, headless, headfull).await?;
            tab.crawl_inner(visit, headless, headfull).await?;
        }
        Ok(())
    }

    pub async fn crawl(
        &self,
        mut visit: impl FnMut(&Self) -> bool,
        headless: bool,
        headfull: bool,
    ) -> anyhow::Result<()> {
        self.crawl_inner(&mut visit, headless, headfull).await
    }
}

pub struct StaticPage {
    url: Url,
    html: Html,
}

impl StaticPage {
    pub async fn new(url: Url) -> anyhow::Result<Self> {
        let html = reqwest::get(url.clone()).await?.text().await?;
        let parsed = Html::parse_document(&html);
        Ok(Self { url, html: parsed })
    }

    pub fn url(&self) -> Url {
        self.url.clone()
    }

    pub fn html(&self) -> Html {
        self.html.clone()
    }

    pub fn article(&self) -> anyhow::Result<Document> {
        extract_article(&self.html.html())
    }

    pub fn title(&self) -> Option<String> {
        let selector = Selector::parse("title").ok()?;
        self.html.select(&selector).next().map(|e| e.inner_html())
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
