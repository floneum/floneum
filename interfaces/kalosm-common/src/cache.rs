use kalosm_model_types::{FileLoadingProgress, FileSource};
use std::path::PathBuf;

#[cfg(feature = "tokio")]
use tokio::fs::{File, OpenOptions};
#[cfg(feature = "tokio")]
use tokio::io::AsyncWriteExt;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CacheError {
    #[cfg(feature = "tokio")]
    #[error("Hugging Face API error: {0}")]
    HuggingFaceApi(#[from] hf_hub::api::tokio::ApiError),
    #[cfg(feature = "tokio")]
    #[error("Unable to get file metadata for {0}: {1}")]
    UnableToGetFileMetadata(PathBuf, #[source] tokio::io::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Unexpected status code: {0}")]
    UnexpectedStatusCode(reqwest::StatusCode),
}

#[derive(Debug, Clone)]
pub struct Cache {
    location: PathBuf,
    /// The huggingface token to use (defaults to the token set with `huggingface-cli login`)
    huggingface_token: Option<String>,
}

impl Cache {
    /// Create a new cache with a specific location
    pub fn new(location: PathBuf) -> Self {
        Self {
            location,
            huggingface_token: None,
        }
    }

    /// Set the Hugging Face token to use for downloading (defaults to the token set with `huggingface-cli login`, and then the environment variable `HF_TOKEN`)
    pub fn with_huggingface_token(mut self, token: Option<String>) -> Self {
        self.huggingface_token = token;
        self
    }

    /// Check if the file exists locally (if it is a local file or if it has been downloaded)
    pub fn exists(&self, source: &FileSource) -> bool {
        match source {
            FileSource::HuggingFace {
                model_id,
                revision,
                file,
                ..
            } => {
                let path = self.location.join(model_id).join(revision);
                let complete_download = path.join(file);
                complete_download.exists()
            }
            FileSource::Local(path) => path.exists(),
        }
    }

    /// Get the bytes from the cache, downloading it if necessary
    pub async fn get_bytes(
        &self,
        source: &FileSource,
        #[allow(unused_mut)] mut progress: impl FnMut(FileLoadingProgress),
    ) -> Result<Vec<u8>, CacheError> {
        #[cfg(feature = "tokio")]
        {
            let path = self.get(source, progress).await;
            tokio::fs::read(path?).await.map_err(CacheError::from)
        }
        #[cfg(not(feature = "tokio"))]
        {
            use futures_util::StreamExt;
            use reqwest::StatusCode;
            use std::str::FromStr;
            match source {
                FileSource::HuggingFace {
                    model_id,
                    revision,
                    file,
                } => {
                    let token = self.huggingface_token.clone().or_else(huggingface_token);
                    let url =
                        format!("https://huggingface.co/{model_id}/resolve/{revision}/{file}");
                    let client = reqwest::Client::new();
                    let head = client
                        .head(&url)
                        .with_authorization_header(token.clone())
                        .send()
                        .await
                        .map_err(CacheError::from)?;
                    let length = head
                        .headers()
                        .get(reqwest::header::CONTENT_LENGTH)
                        .and_then(|length| {
                            length.to_str().ok().and_then(|s| u64::from_str(s).ok())
                        });
                    if let Some(length) = length {
                        progress(FileLoadingProgress {
                            progress: 0,
                            cached_size: 0,
                            size: length,
                            start_time: None,
                        });
                    }
                    let request = client.get(url).with_authorization_header(token);
                    let response = request.send().await?;

                    let status = response.status();
                    if !(status == StatusCode::OK || status == StatusCode::PARTIAL_CONTENT) {
                        return Err(CacheError::UnexpectedStatusCode(status));
                    }

                    let mut current_progress = 0;

                    let mut bytes = Vec::new();
                    let mut stream = response.bytes_stream();
                    while let Some(chunk) = stream.next().await {
                        let chunk = chunk?;
                        bytes.extend_from_slice(&chunk);
                        tracing::trace!("wrote chunk of size {}", chunk.len());
                        current_progress += chunk.len() as u64;
                        if let Some(length) = length {
                            progress(FileLoadingProgress {
                                progress: current_progress,
                                cached_size: 0,
                                size: length,
                                start_time: None,
                            });
                        }
                    }

                    Ok(bytes)
                }
                _ => Err(CacheError::Io(std::io::Error::other(
                    "Local file access not supported without the 'tokio' feature",
                ))),
            }
        }
    }

    /// Get the file from the cache, downloading it if necessary
    #[cfg(feature = "tokio")]
    pub async fn get(
        &self,
        source: &FileSource,
        progress: impl FnMut(FileLoadingProgress),
    ) -> Result<PathBuf, CacheError> {
        use hf_hub::{Repo, RepoType};
        match source {
            FileSource::HuggingFace {
                model_id,
                revision,
                file,
            } => {
                let token = self.huggingface_token.clone().or_else(huggingface_token);

                let repo = Repo::with_revision(
                    model_id.to_string(),
                    RepoType::Model,
                    revision.to_string(),
                );
                let api = hf_hub::api::tokio::Api::new()?.repo(repo);
                let url = api.url(file);
                let client = reqwest::Client::new();
                tracing::trace!("Fetching metadata for {file} from {url}");
                let response = client
                    .head(&url)
                    .with_authorization_header(token.clone())
                    .send()
                    .await;

                let path = self.location.join(model_id).join(revision);
                let complete_download = path.join(file);

                // Quick check without lock - if file exists and is up-to-date, return it
                if is_file_current(&complete_download, &response).await {
                    return Ok(complete_download);
                }

                // Need to download - acquire lock to prevent race conditions
                let lock_path = path.join(format!("{file}.lock"));
                tokio::fs::create_dir_all(&path).await?;

                // Acquire exclusive lock using blocking task to avoid blocking async runtime
                let lock_path_clone = lock_path.clone();
                let _lock_file = tokio::task::spawn_blocking(move || {
                    let file = std::fs::File::create(&lock_path_clone)?;
                    file.lock()?;
                    Ok::<_, std::io::Error>(file)
                })
                .await
                .map_err(|e| CacheError::Io(std::io::Error::other(e)))??;

                // Double-check if file was downloaded while we were waiting for lock
                if is_file_current(&complete_download, &response).await {
                    let _ = tokio::fs::remove_file(&lock_path).await;
                    return Ok(complete_download);
                }

                let incomplete_download = path.join(format!("{file}.partial"));

                tracing::trace!("Downloading into {:?}", incomplete_download);

                let download_result = download_into(
                    url,
                    &incomplete_download,
                    response?,
                    client,
                    token,
                    progress,
                )
                .await;

                if let Err(e) = download_result {
                    let _ = tokio::fs::remove_file(&lock_path).await;
                    return Err(e);
                }

                // Rename the file to remove the .partial extension
                let rename_result =
                    tokio::fs::rename(&incomplete_download, &complete_download).await;

                // Release lock and clean up lock file
                let _ = tokio::fs::remove_file(&lock_path).await;

                rename_result?;

                Ok(complete_download)
            }
            FileSource::Local(path) => Ok(path.clone()),
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for Cache {
    fn default() -> Self {
        Self {
            location: {
                #[cfg(feature = "tokio")]
                {
                    // Try various locations in order of preference
                    dirs::data_dir()
                        .or_else(dirs::cache_dir)
                        .or_else(|| std::env::var("HOME").ok().map(std::path::PathBuf::from))
                        .or_else(|| std::env::current_dir().ok())
                        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
                        .join("kalosm")
                        .join("cache")
                }
                #[cfg(not(feature = "tokio"))]
                {
                    Default::default()
                }
            },
            huggingface_token: None,
        }
    }
}

/// Check if the local file exists and is up-to-date compared to the server's Last-Modified header.
/// Returns true if the file can be used as-is, false if it needs to be downloaded.
#[cfg(feature = "tokio")]
async fn is_file_current(
    path: &std::path::Path,
    response: &Result<reqwest::Response, reqwest::Error>,
) -> bool {
    if !path.exists() {
        return false;
    }

    let Ok(metadata) = tokio::fs::metadata(path).await else {
        return false;
    };

    let Ok(file_last_modified) = metadata.modified() else {
        return false;
    };

    // If the server says the file hasn't been modified since we downloaded it, we can use the local file
    if let Some(last_updated) = response
        .as_ref()
        .ok()
        .and_then(|response| response.headers().get(reqwest::header::LAST_MODIFIED))
        .and_then(|last_updated| last_updated.to_str().ok())
        .and_then(|s| httpdate::parse_http_date(s).ok())
    {
        last_updated <= file_last_modified
    } else {
        // If we're offline or the server doesn't provide Last-Modified, use the local file
        true
    }
}

#[cfg(feature = "tokio")]
async fn download_into<U: reqwest::IntoUrl>(
    url: U,
    file: &PathBuf,
    head: reqwest::Response,
    client: reqwest::Client,
    token: Option<String>,
    mut progress: impl FnMut(FileLoadingProgress),
) -> Result<(), CacheError> {
    use reqwest::header::{HeaderValue, CONTENT_LENGTH, RANGE};
    use reqwest::StatusCode;
    use std::str::FromStr;

    let length = head
        .headers()
        .get(CONTENT_LENGTH)
        .ok_or("response doesn't include the content length")
        .unwrap();
    let length = length.to_str().ok().and_then(|s| u64::from_str(s).ok());

    let (start, mut output_file) = if let Ok(metadata) = tokio::fs::metadata(file).await {
        let start = metadata.len();
        let output_file = OpenOptions::new().append(true).open(file).await.unwrap();
        (start, output_file)
    } else {
        tokio::fs::create_dir_all(file.parent().unwrap()).await?;
        (0, File::create(file).await.unwrap())
    };

    if let Some(length) = length {
        progress(FileLoadingProgress {
            progress: start,
            cached_size: start,
            size: length,
            start_time: Some(std::time::Instant::now()),
        });
    }

    if Some(start) == length {
        tracing::trace!("File {} already downloaded", file.display());
        progress(FileLoadingProgress {
            progress: start,
            cached_size: start,
            size: length.unwrap_or(0),
            start_time: Some(std::time::Instant::now()),
        });
        return Ok(());
    }

    let range = length
        .and_then(|length| HeaderValue::from_str(&format!("bytes={}-{}", start, length - 1)).ok());

    tracing::trace!("Fetching range {:?}", range);
    let mut request = client.get(url).with_authorization_header(token);
    if let Some(range) = range {
        request = request.header(RANGE, range);
    }
    let mut response = request.send().await?;

    let status = response.status();
    if !(status == StatusCode::OK || status == StatusCode::PARTIAL_CONTENT) {
        return Err(CacheError::UnexpectedStatusCode(status));
    }

    let mut current_progress = start;

    while let Some(chunk) = response.chunk().await? {
        output_file.write_all(&chunk).await?;
        tracing::trace!("wrote chunk of size {}", chunk.len());
        current_progress += chunk.len() as u64;
        if let Some(length) = length {
            progress(FileLoadingProgress {
                progress: current_progress,
                cached_size: start,
                size: length,
                start_time: Some(std::time::Instant::now()),
            });
        }
    }

    tracing::trace!("Download of {} complete", file.display());

    Ok(())
}

trait RequestBuilderExt {
    fn with_authorization_header(self, token: Option<String>) -> Self;
}

impl RequestBuilderExt for reqwest::RequestBuilder {
    fn with_authorization_header(self, token: Option<String>) -> Self {
        if let Some(token) = token {
            self.header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"))
        } else {
            self
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn downloads_work() {
    let url = "https://httpbin.org/range/102400?duration=2";
    let file = PathBuf::from("download.bin");
    let progress = |p| {
        println!("Progress: {p:?}");
    };
    let client = reqwest::Client::new();
    let response = client.head(url).send().await.unwrap();
    download_into(url, &file, response, client, None, progress)
        .await
        .unwrap();
    assert!(file.exists());
    tokio::fs::remove_file(file).await.unwrap();
}

fn huggingface_token() -> Option<String> {
    cfg!(not(target_arch = "wasm32"))
        .then(|| {
            let cache = hf_hub::Cache::default();
            cache.token().or_else(|| std::env::var("HF_TOKEN").ok())
        })
        .flatten()
}
