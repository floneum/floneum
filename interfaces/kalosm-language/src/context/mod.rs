//! Context for language models to consume.

mod document;
pub use document::*;
mod io;
pub use io::*;
#[cfg(feature = "scrape")]
mod page;
#[cfg(feature = "scrape")]
pub use page::*;
mod rss;
pub use self::rss::*;
mod search;
pub use search::*;

pub use url::Url;
