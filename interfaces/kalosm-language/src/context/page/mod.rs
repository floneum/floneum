use super::{document::Document, IntoDocument};
use url::Url;

mod browse;
pub use browse::*;
mod crawl;
pub use crawl::*;
mod node;
pub use node::*;
#[allow(clippy::module_inception)]
mod page;
pub use page::*;
