use crate::context::page::BrowserMode;
use crate::context::page::Page;
use core::task::Context;
use dashmap::DashMap;
use once_cell::sync::OnceCell;
use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::task::Poll;
use std::task::Waker;
use tokio::time::Duration;
use tokio::time::Instant;
use url::Url;

const COOLDOWN: Duration = Duration::from_secs(5);

/// Feedback that can be given to the crawler after visiting a page.
pub enum CrawlFeedback {
    /// Continue crawling from this page.
    Continue,
    /// Dont follow any links on this page.
    DontFollow,
    /// Stop the entire crawler
    Stop,
}

/// Trait for a callback that is called when a page is visited.
///
/// # Example
///
/// ```rust, no_run
/// use kalosm_language::BrowserMode;
/// use kalosm_language::CrawlFeedback;
/// use kalosm_language::Page;
/// use kalosm_language::Url;
/// use std::future::Future;
/// use std::pin::Pin;
/// use std::sync::atomic::AtomicUsize;
/// use std::sync::atomic::Ordering;
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() {
///     let count = Arc::new(AtomicUsize::new(0));
///     let real_visited = Arc::new(AtomicUsize::new(0));
///     Page::crawl(
///         Url::parse("https://www.nytimes.com/live/2023/09/21/world/zelensky-russia-ukraine-news")
///             .unwrap(),
///         BrowserMode::Static,
///         move |page: Page| {
///             let count = count.clone();
///             let real_visited = real_visited.clone();
///             Box::pin(async move {
///                 real_visited.fetch_add(1, Ordering::SeqCst);
///                 let current_count = count.load(Ordering::SeqCst);
///                 if current_count > 1000 {
///                     return CrawlFeedback::Stop;
///                 }
///
///                 let Ok(page) = page.article().await else {
///                     return CrawlFeedback::DontFollow;
///                 };
///
///                 let body = page.body();
///
///                 if body.len() < 100 {
///                     return CrawlFeedback::DontFollow;
///                 }
///
///                 println!("Title: {}", page.title());
///                 println!("Article:\n{}", body);
///
///                 count.fetch_add(1, Ordering::SeqCst);
///
///                 CrawlFeedback::Continue
///             }) as Pin<Box<dyn Future<Output = CrawlFeedback>>>
///         },
///     )
///     .await
///     .unwrap();
/// }
/// ```
pub trait CrawlingCallback: Send + Sync + 'static {
    /// The function that is called when a page is visited.
    fn visit(&self, page: Page) -> Pin<Box<dyn Future<Output = CrawlFeedback>>>;
}

impl<T: Fn(Page) -> Pin<Box<dyn Future<Output = CrawlFeedback>>> + Send + Sync + 'static>
    CrawlingCallback for T
{
    fn visit(&self, page: Page) -> Pin<Box<dyn Future<Output = CrawlFeedback>>> {
        (self)(page)
    }
}

struct ActiveLinks {
    active: AtomicUsize,
    waker: OnceCell<Waker>,
}

impl ActiveLinks {
    fn new() -> Self {
        Self {
            active: AtomicUsize::new(0),
            waker: Default::default(),
        }
    }

    fn add(&self) {
        self.active.fetch_add(1, Ordering::SeqCst);
    }

    fn remove(&self) {
        let new = self.active.fetch_sub(1, Ordering::SeqCst) - 1;
        if new == 0 {
            if let Some(waker) = self.waker.get() {
                waker.wake_by_ref();
            }
        }
    }

    fn abort(&self) {
        self.active.store(0, Ordering::SeqCst);
        if let Some(waker) = self.waker.get() {
            waker.wake_by_ref();
        }
    }

    async fn wait(&self) {
        self.await;
    }
}

impl std::future::Future for &ActiveLinks {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let active = self.active.load(Ordering::SeqCst);
        if active == 0 {
            return Poll::Ready(());
        }

        let _ = self.waker.try_insert(cx.waker().clone());

        Poll::Pending
    }
}

pub(crate) struct Crawler<T> {
    active: Arc<ActiveLinks>,
    visit: Arc<T>,
    mode: BrowserMode,
    queued: Arc<DashMap<url::Origin, DomainQueue<T>>>,
    aborted: Arc<AtomicBool>,
}

impl<T> Clone for Crawler<T> {
    fn clone(&self) -> Self {
        Self {
            active: self.active.clone(),
            visit: self.visit.clone(),
            mode: self.mode,
            queued: self.queued.clone(),
            aborted: self.aborted.clone(),
        }
    }
}

impl<T: CrawlingCallback> Crawler<T> {
    pub fn new(mode: BrowserMode, visit: T) -> Self {
        Self {
            active: Arc::new(ActiveLinks::new()),
            mode,
            queued: Default::default(),
            visit: Arc::new(visit),
            aborted: Default::default(),
        }
    }

    pub fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::SeqCst)
    }

    pub fn abort(self) {
        self.aborted.store(true, Ordering::SeqCst);
        for queue in self.queued.iter() {
            queue.abort();
        }
        self.active.abort();
    }

    pub async fn crawl(&mut self, url: Url) -> anyhow::Result<()> {
        if self.is_aborted() {
            return Ok(());
        }

        self.add_urls(vec![url]);

        self.active.wait().await;

        Ok(())
    }

    fn add_urls(&self, urls: Vec<Url>) {
        if self.is_aborted() {
            return;
        }

        for url in urls {
            let origin = url.origin();
            if let Some(mut queue) = self.queued.get_mut(&origin) {
                queue.push(url);
                continue;
            }

            let mut queue = DomainQueue::new(self.clone());
            queue.push(url);
            self.queued.insert(origin, queue);
        }
    }
}

struct DomainQueue<T> {
    visited: HashSet<Url>,
    queue: tokio::sync::mpsc::UnboundedSender<Url>,
    crawler: Crawler<T>,
    task: tokio::task::JoinHandle<()>,
}

impl<T: CrawlingCallback> DomainQueue<T> {
    fn new(crawler: Crawler<T>) -> Self {
        let (queue, mut rx) = tokio::sync::mpsc::unbounded_channel();

        let pool = get_local_pool();
        let task = {
            let crawler = crawler.clone();
            pool.spawn_pinned(move || async move {
                while let Some(url) = rx.recv().await {
                    let mode = crawler.mode;
                    let wait_until = Instant::now() + COOLDOWN;
                    let page = Page::new_wait_until(url, mode, wait_until).unwrap();

                    let visit = crawler.visit.visit(page.clone());

                    let feedback = visit.await;

                    match feedback {
                        CrawlFeedback::Continue => match page.links().await {
                            Ok(new_urls) => {
                                crawler.add_urls(new_urls);
                            }
                            Err(err) => tracing::error!("Error getting links: {}", err),
                        },
                        CrawlFeedback::DontFollow => {}
                        CrawlFeedback::Stop => {
                            crawler.abort();
                            return;
                        }
                    }
                    crawler.active.remove();
                }
            })
        };

        Self {
            task,
            queue,
            visited: HashSet::new(),
            crawler,
        }
    }

    fn abort(&self) {
        self.task.abort();
    }

    fn push(&mut self, url: Url) {
        if self.visited.contains(&url) {
            return;
        }

        self.crawler.active.add();

        self.visited.insert(url.clone());

        let _ = self.queue.send(url);
    }
}

fn get_local_pool() -> tokio_util::task::LocalPoolHandle {
    static LOCAL_POOL: OnceCell<tokio_util::task::LocalPoolHandle> = OnceCell::new();
    LOCAL_POOL
        .get_or_init(|| {
            tokio_util::task::LocalPoolHandle::new(
                std::thread::available_parallelism()
                    .map(Into::into)
                    .unwrap_or(1),
            )
        })
        .clone()
}
