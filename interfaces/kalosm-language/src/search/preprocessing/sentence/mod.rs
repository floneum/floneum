use crate::prelude::{Chunk, Chunker, Document, Embedder};
use srx::SRX;
use std::cell::OnceCell;
use std::future::Future;
use std::rc::Rc;
use std::str::FromStr;

/// The default sentence chunker. Unlike [`SentenceChunker`], this is Send + Sync
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultSentenceChunker;

impl Chunker for DefaultSentenceChunker {
    type Error<E: Send + Sync + 'static> = E;

    /// Chunk a document into embedded snippets.
    fn chunk<E: Embedder + Send>(
        &self,
        document: &Document,
        embedder: &E,
    ) -> impl Future<Output = Result<Vec<Chunk<E::VectorSpace>>, Self::Error<E::Error>>> + Send
    {
        let default = SentenceChunker::default();
        // Split the document into sentences. We first just collect the sentences as strings and byte ranges
        let mut initial_chunks = Vec::new();
        let body = document.body();
        let ranges = default.split_sentences(document.body());
        for chunk in &ranges {
            initial_chunks.push(body[chunk.clone()].to_string());
        }

        embed_chunk(embedder, initial_chunks, ranges)
    }
}

/// A [`Chunker`] that splits a string into sentences with a given [SRX](https://www.unicode.org/uli/pas/srx/srx20.html) rules.
///
/// This uses the [srx](https://crates.io/crates/srx) crate to parse and apply the rules.
#[derive(Debug, Clone)]
pub struct SentenceChunker {
    srx: Rc<SRX>,
}

impl SentenceChunker {
    /// Create a new sentence chunker from a xml rules string
    pub fn new(rules: &str) -> Self {
        Self {
            srx: SRX::from_str(rules)
                .expect("the rules file is valid")
                .into(),
        }
    }

    /// Create a new sentence chunker from anything that implements [`std::io::Read`](std::io::Read) in the srx rules format
    pub fn load(reader: impl std::io::Read) -> Result<Self, srx::Error> {
        Ok(Self {
            srx: SRX::from_reader(reader)?.into(),
        })
    }

    /// Split the body of a document into a list of ranges with sentences
    pub fn split_sentences(&self, string: &str) -> Vec<std::ops::Range<usize>> {
        // Try to autodetect the language of the document
        let language = whatlang::detect_lang(string)
            .map(|lang_code| lang_code.code())
            .unwrap_or("en");

        // Then get the language specific rules to split the document into sentences
        let rules = self.srx.language_rules(language);

        rules.split_ranges(string)
    }
}

impl Default for SentenceChunker {
    fn default() -> Self {
        // The rules are expensive to parse (~1 second), so we cache them in a static once cell
        thread_local! {
            static DEFAULT_RULES: OnceCell<Rc<SRX>> = const { OnceCell::new() };
        }

        let rules = DEFAULT_RULES.with(|default| {
            default
                .get_or_init(|| {
                    // Defaults to the language tool ruleset: https://github.com/languagetool-org/languagetool/blob/master/languagetool-core/src/main/resources/org/languagetool/resource/segment.srx
                    let rules = SRX::from_str(include_str!("./assets/segment.srx"))
                        .expect("the rules file is valid");
                    Rc::new(rules)
                })
                .clone()
        });

        Self { srx: rules }
    }
}

/// A strategy for chunking a document into smaller pieces.
impl Chunker for SentenceChunker {
    type Error<E: Send + Sync + 'static> = E;

    /// Chunk a document into embedded snippets.
    fn chunk<E: Embedder + Send>(
        &self,
        document: &Document,
        embedder: &E,
    ) -> impl Future<Output = Result<Vec<Chunk<E::VectorSpace>>, Self::Error<E::Error>>> + Send
    {
        // Split the document into sentences. We first just collect the sentences as strings and byte ranges
        let mut initial_chunks = Vec::new();
        let body = document.body();
        let ranges = self.split_sentences(document.body());
        for chunk in &ranges {
            initial_chunks.push(body[chunk.clone()].to_string());
        }

        embed_chunk(embedder, initial_chunks, ranges)
    }
}

async fn embed_chunk<E: Embedder + Send>(
    embedder: &E,
    initial_chunks: Vec<String>,
    ranges: Vec<std::ops::Range<usize>>,
) -> Result<Vec<Chunk<E::VectorSpace>>, E::Error> {
    // Next embed them all in one big batch
    let embeddings = embedder.embed_vec(initial_chunks).await?;

    // Now merge the embeddings and ranges into chunks
    let mut chunks = Vec::new();
    for (embedding, chunk) in embeddings.into_iter().zip(ranges) {
        let chunk = Chunk {
            byte_range: chunk,
            embeddings: vec![embedding],
        };
        chunks.push(chunk);
    }

    Ok(chunks)
}
