use hf_hub::{Repo, RepoType};
use httpdate::parse_http_date;
use kalosm_model_types::{FileLoadingProgress, FileSource};
use reqwest::{
    header::{HeaderValue, CONTENT_LENGTH, LAST_MODIFIED, RANGE},
    IntoUrl,
};
use reqwest::{Response, StatusCode};
use std::path::PathBuf;
use std::str::FromStr;
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;

#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("Hugging Face API error: {0}")]
    HuggingFaceApi(#[from] hf_hub::api::sync::ApiError),
    #[error("Unable to get file metadata for {0}: {1}")]
    UnableToGetFileMetadata(PathBuf, #[source] tokio::io::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Unexpected status code: {0}")]
    UnexpectedStatusCode(StatusCode),
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

    /// Get the file from the cache, downloading it if necessary
    pub async fn get(
        &self,
        source: &FileSource,
        progress: impl FnMut(FileLoadingProgress),
    ) -> Result<PathBuf, CacheError> {
        match source {
            FileSource::HuggingFace {
                model_id,
                revision,
                file,
            } => {
                let token = self.huggingface_token.clone().or_else(huggingface_token);

                let path = self.location.join(model_id).join(revision);
                let complete_download = path.join(file);

                let repo = Repo::with_revision(
                    model_id.to_string(),
                    RepoType::Model,
                    revision.to_string(),
                );
                let api = hf_hub::api::sync::Api::new()?.repo(repo);
                let url = api.url(file);
                let client = reqwest::Client::new();
                tracing::trace!("Fetching metadata for {file} from {url}");
                let response = client
                    .head(&url)
                    .with_authorization_header(token.clone())
                    .send()
                    .await;

                if complete_download.exists() {
                    let metadata = tokio::fs::metadata(&complete_download).await.map_err(|e| {
                        CacheError::UnableToGetFileMetadata(complete_download.clone(), e)
                    })?;
                    let file_last_modified = metadata.modified()?;
                    // If the server says the file hasn't been modified since we downloaded it, we can use the local file
                    if let Some(last_updated) = response
                        .as_ref()
                        .ok()
                        .and_then(|response| response.headers().get(LAST_MODIFIED))
                        .and_then(|last_updated| last_updated.to_str().ok())
                        .and_then(|s| parse_http_date(s).ok())
                    {
                        if last_updated <= file_last_modified {
                            return Ok(complete_download);
                        }
                    } else {
                        // Or if we are offline, we can use the local file
                        return Ok(complete_download);
                    }
                }
                let incomplete_download = path.join(format!("{file}.partial"));

                tracing::trace!("Downloading into {:?}", incomplete_download);

                download_into(
                    url,
                    &incomplete_download,
                    response?,
                    client,
                    token,
                    progress,
                )
                .await?;

                // Rename the file to remove the .partial extension
                tokio::fs::rename(&incomplete_download, &complete_download).await?;

                Ok(complete_download)
            }
            FileSource::Local(path) => Ok(path.clone()),
        }
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            location: dirs::data_dir().unwrap().join("kalosm").join("cache"),
            huggingface_token: None,
        }
    }
}

async fn download_into<U: IntoUrl>(
    url: U,
    file: &PathBuf,
    head: Response,
    client: reqwest::Client,
    token: Option<String>,
    mut progress: impl FnMut(FileLoadingProgress),
) -> Result<(), CacheError> {
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
            start_time: std::time::Instant::now(),
        });
    }

    if Some(start) == length {
        tracing::trace!("File {} already downloaded", file.display());
        progress(FileLoadingProgress {
            progress: start,
            cached_size: start,
            size: length.unwrap_or(0),
            start_time: std::time::Instant::now(),
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
                start_time: std::time::Instant::now(),
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
        println!("Progress: {:?}", p);
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
    let cache = hf_hub::Cache::default();
    cache.token().or_else(|| std::env::var("HF_TOKEN").ok())
}
