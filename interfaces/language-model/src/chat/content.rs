use std::{
    fmt::Arguments,
    ops::AddAssign,
    path::{Path, PathBuf},
};

/// The contents of a chat message. The message can contain chunks of interleaved text and media.
///
/// ```rust, no_run
/// # use std::path::PathBuf;
/// # use kalosm::language::*;
/// // You can push text and media chunks to the message content.
/// let mut contents = MessageContent::new();
/// contents.push("Hello, world!");
/// contents.push(MediaSource::url("https://example.com/image.png"));
/// contents
///     .try_push(PathBuf::from("path/to/file.png"))
///     .unwrap();
///
/// // Or create a message content from a tuple of chunks.
/// let contents = MessageContent::from((
///     "Hello, world!",
///     MediaSource::url("https://example.com/image.png"),
/// ));
///
/// // Or use the `+=` operator to add chunks to the message content.
/// let mut contents = MessageContent::from("Hello, world!");
/// contents += MediaSource::url("https://example.com/image.png");
/// contents += MediaSource::try_from(PathBuf::from("path/to/file.png")).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageContent {
    chunks: Vec<ContentChunk>,
}

impl MessageContent {
    /// Create a new empty [`MessageContent`].
    pub fn new() -> Self {
        MessageContent { chunks: Vec::new() }
    }

    /// Push a chunk to the end of this message.
    pub fn push(&mut self, chunk: impl Into<ContentChunk>) {
        self.chunks.push(chunk.into());
    }

    /// Try to push a chunk to the end of this message.
    /// Returns an error if the chunk could not be converted to a [`ContentChunk`].
    pub fn try_push<T: TryInto<ContentChunk>>(&mut self, chunk: T) -> Result<(), T::Error> {
        let chunk = chunk.try_into()?;
        self.chunks.push(chunk);
        Ok(())
    }

    /// Get the chunks of this message.
    pub fn chunks(&self) -> &[ContentChunk] {
        &self.chunks
    }
}

impl<T: Into<ContentChunk>> From<T> for MessageContent {
    fn from(chunk: T) -> Self {
        MessageContent {
            chunks: vec![chunk.into()],
        }
    }
}

macro_rules! impl_from_tuple {
    ($($name:ident),+) => {
        #[allow(non_snake_case)]
        impl<$($name: Into<ContentChunk>),+> From<($($name,)+)> for MessageContent {
            fn from(($($name,)+): ($($name,)+)) -> Self {
                let chunks = vec![
                    $(
                        $name.into(),
                    )+
                ];
                MessageContent { chunks }
            }
        }
    };
}

impl_from_tuple!(A);
impl_from_tuple!(A, B);
impl_from_tuple!(A, B, C);
impl_from_tuple!(A, B, C, D);
impl_from_tuple!(A, B, C, D, E);
impl_from_tuple!(A, B, C, D, E, F);
impl_from_tuple!(A, B, C, D, E, F, G);
impl_from_tuple!(A, B, C, D, E, F, G, H);
impl_from_tuple!(A, B, C, D, E, F, G, H, I);
impl_from_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_from_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_from_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_from_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_from_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);

impl<T> AddAssign<T> for MessageContent
where
    T: Into<ContentChunk>,
{
    fn add_assign(&mut self, rhs: T) {
        self.chunks.push(rhs.into());
    }
}

/// A chunk of content in a chat message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentChunk {
    /// A text chunk.
    Text(String),
    /// A media chunk.
    Media(MediaChunk),
}

impl From<String> for ContentChunk {
    fn from(text: String) -> Self {
        ContentChunk::Text(text)
    }
}

impl From<&str> for ContentChunk {
    fn from(text: &str) -> Self {
        ContentChunk::Text(text.to_string())
    }
}

impl From<Arguments<'_>> for ContentChunk {
    fn from(args: Arguments<'_>) -> Self {
        ContentChunk::Text(args.to_string())
    }
}

impl From<MediaSource> for ContentChunk {
    fn from(source: MediaSource) -> Self {
        ContentChunk::Media(MediaChunk {
            media_type: MediaType::Image,
            source,
        })
    }
}

impl TryFrom<PathBuf> for ContentChunk {
    type Error = std::io::Error;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        Ok(ContentChunk::Media(MediaChunk {
            media_type: MediaType::Image,
            source: MediaSource::try_from(path)?,
        }))
    }
}

/// A chunk of media content that can be used with a LLM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaChunk {
    media_type: MediaType,
    source: MediaSource,
}

impl MediaChunk {
    /// Create a new [`MediaChunk`] from a [`MediaSource`] and a [`MediaType`].
    pub fn new(source: MediaSource, media_type: MediaType) -> Self {
        MediaChunk { media_type, source }
    }

    /// Get the media type of this chunk.
    pub fn media_type(&self) -> MediaType {
        self.media_type
    }

    /// Get the source of this chunk.
    pub fn source(&self) -> &MediaSource {
        &self.source
    }
}

/// The type of a [`MediaChunk`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaType {
    /// An image media type (e.g. PNG, JPEG).
    Image,
    /// A video media type (e.g. MP4, AVI).
    Video,
}

/// The source of some media content that can be used with a LLM.
///
/// ```rust, no_run
/// # use std::path::PathBuf;
/// # use kalosm::language::*;
/// // You can create a `MediaSource` from a URL,
/// let source = MediaSource::url("https://example.com/image.png");
/// // byte array,
/// let source = MediaSource::bytes(vec![1, 2, 3]);
/// // or a file path.
/// let source = MediaSource::try_from(PathBuf::from("path/to/file.png")).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaSource {
    variant: MediaSourceVariant,
}

impl MediaSource {
    /// Create a new `MediaSource` from a URL.
    pub fn url(url: impl ToString) -> Self {
        MediaSource {
            variant: MediaSourceVariant::Url(url.to_string()),
        }
    }

    /// Create a new `MediaSource` from a byte array.
    pub fn bytes(bytes: impl Into<Box<[u8]>>) -> Self {
        MediaSource {
            variant: MediaSourceVariant::Bytes(bytes.into()),
        }
    }

    /// Try to create a new `MediaSource` from a file path.
    /// Returns an error if the file could not be read.
    pub fn file(path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let bytes = std::fs::read(path)?;
        Ok(MediaSource {
            variant: MediaSourceVariant::Bytes(bytes.into_boxed_slice()),
        })
    }
}

impl From<String> for MediaSource {
    fn from(url: String) -> Self {
        MediaSource {
            variant: MediaSourceVariant::Url(url),
        }
    }
}

impl From<&str> for MediaSource {
    fn from(url: &str) -> Self {
        MediaSource {
            variant: MediaSourceVariant::Url(url.to_string()),
        }
    }
}

impl TryFrom<PathBuf> for MediaSource {
    type Error = std::io::Error;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        let bytes = std::fs::read(path)?;
        Ok(MediaSource {
            variant: MediaSourceVariant::Bytes(bytes.into_boxed_slice()),
        })
    }
}

impl From<Vec<u8>> for MediaSource {
    fn from(bytes: Vec<u8>) -> Self {
        MediaSource {
            variant: MediaSourceVariant::Bytes(bytes.into_boxed_slice()),
        }
    }
}

impl From<&[u8]> for MediaSource {
    fn from(bytes: &[u8]) -> Self {
        MediaSource {
            variant: MediaSourceVariant::Bytes(bytes.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MediaSourceVariant {
    Url(String),
    Bytes(Box<[u8]>),
}

#[cfg(feature = "remote")]
impl MediaChunk {
    /// Create a url from the media chunk by either using the url or encoding the bytes as base64.
    pub fn as_url(&self) -> String {
        use base64::{prelude::BASE64_STANDARD_NO_PAD, Engine};
        
        match (&self.source.variant, &self.media_type) {
            (MediaSourceVariant::Url(url), _) => url.to_string(),
            (MediaSourceVariant::Bytes(bytes), MediaType::Image) => format!(
                "data:image/png;base64,{}",
                BASE64_STANDARD_NO_PAD.encode(bytes)
            ),
            (MediaSourceVariant::Bytes(bytes), MediaType::Video) => format!(
                "data:video/mp4;base64,{}",
                BASE64_STANDARD_NO_PAD.encode(bytes)
            ),
        }
    }
}
