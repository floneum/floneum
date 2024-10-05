use anyhow::bail;
use hf_hub::{Repo, RepoType};
use httpdate::parse_http_date;
use reqwest::header::{HeaderValue, CONTENT_LENGTH, LAST_MODIFIED, RANGE};
use reqwest::{Response, StatusCode, Url};
use std::path::PathBuf;
use std::str::FromStr;
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;

use crate::FileSource;

/// The progress starting a model
#[derive(Clone, Debug)]
pub enum ModelLoadingProgress {
    /// The model is downloading
    Downloading {
        /// The source of the download. This is not a path or URL, but a description of the source
        source: String,
        progress: FileLoadingProgress,
    },
    /// The model is loading
    Loading {
        /// The progress of the loading, from 0 to 1
        progress: f32,
    },
}

/// The progress of a file download
#[derive(Clone, Debug)]
pub struct FileLoadingProgress {
    /// The time stamp the download started
    start_time: std::time::Instant,
    /// The size of the cached part of the download in bytes
    cached_size: u64,
    /// The size of the download in bytes
    size: u64,
    /// The progress of the download in bytes, from 0 to size
    progress: u64,
}

impl ModelLoadingProgress {
    /// Create a new downloading progress
    pub fn downloading(source: String, file_loading_progress: FileLoadingProgress) -> Self {
        Self::Downloading {
            source,
            progress: file_loading_progress,
        }
    }

    /// Create a new downloading progress
    pub fn downloading_progress(
        source: String,
    ) -> impl FnMut(FileLoadingProgress) -> Self + Send + Sync {
        move |progress| ModelLoadingProgress::downloading(source.clone(), progress)
    }

    /// Create a new loading progress
    pub fn loading(progress: f32) -> Self {
        Self::Loading { progress }
    }

    /// Return the percent complete
    pub fn progress(&self) -> f32 {
        match self {
            Self::Downloading {
                progress:
                    FileLoadingProgress {
                        progress,
                        size,
                        cached_size,
                        ..
                    },
                ..
            } => (*progress - *cached_size) as f32 / *size as f32,
            Self::Loading { progress } => *progress,
        }
    }

    /// Try to estimate the time remaining for a download
    pub fn estimate_time_remaining(&self) -> Option<std::time::Duration> {
        match self {
            Self::Downloading {
                progress: FileLoadingProgress { start_time, .. },
                ..
            } => {
                let elapsed = start_time.elapsed();
                let progress = self.progress();
                let remaining = (1. - progress) * elapsed.as_secs_f32();
                Some(std::time::Duration::from_secs_f32(remaining))
            }
            _ => None,
        }
    }

    /// A default loading progress bar
    pub fn multi_bar_loading_indicator() -> impl FnMut(ModelLoadingProgress) + Send + Sync + 'static
    {
        use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
        use std::collections::HashMap;
        let m = MultiProgress::new();
        let sty = ProgressStyle::with_template(
            "{spinner:.green} {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] ({decimal_bytes_per_sec}, ETA {eta})",
        )
        .unwrap();
        let mut progress_bars = HashMap::new();

        move |progress| match progress {
            Self::Downloading {
                source,
                progress:
                    FileLoadingProgress {
                        progress,
                        size,
                        cached_size,
                        ..
                    },
                ..
            } => {
                let progress_bar = progress_bars.entry(source.clone()).or_insert_with(|| {
                    let pb = m.add(ProgressBar::new(size));
                    pb.set_message(format!("Downloading {source}"));
                    pb.set_style(sty.clone());
                    pb.set_position(cached_size);
                    pb
                });

                progress_bar.set_position(progress);
            }
            ModelLoadingProgress::Loading { progress } => {
                for pb in progress_bars.values_mut() {
                    pb.finish();
                }
                let progress = progress * 100.;
                m.println(format!("Loading {progress:.2}%")).unwrap();
            }
        }
    }
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
    ) -> anyhow::Result<PathBuf> {
        match source {
            FileSource::HuggingFace {
                model_id,
                revision,
                file,
            } => {
                let token = self.huggingface_token.clone().or_else(huggingface_token);

                let path = self.location.join(model_id).join(revision);
                let complete_download = path.join(file);

                let api = hf_hub::api::sync::Api::new()?;
                let repo = Repo::with_revision(
                    model_id.to_string(),
                    RepoType::Model,
                    revision.to_string(),
                );
                let api = api.repo(repo);
                let url = api.url(file);
                let url = Url::from_str(&url)?;
                let client = reqwest::Client::new();
                let response = client
                    .head(url.clone())
                    .with_authorization_header(token.clone())
                    .send()
                    .await;

                if complete_download.exists() {
                    let metadata = tokio::fs::metadata(&complete_download).await?;
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
                let incomplete_download = path.join(format!("{}.partial", file));

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

impl FileSource {
    /// Check if the file exists locally (if it is a local file or if it has been downloaded)
    pub async fn download(
        &self,
        progress: impl FnMut(FileLoadingProgress),
    ) -> anyhow::Result<PathBuf> {
        let cache = Cache::default();
        cache.get(self, progress).await
    }
}

async fn download_into(
    url: Url,
    file: &PathBuf,
    head: Response,
    client: reqwest::Client,
    token: Option<String>,
    mut progress: impl FnMut(FileLoadingProgress),
) -> anyhow::Result<()> {
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
        bail!("Unexpected status code: {:?}", status);
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

#[tokio::test]
async fn downloads_work() {
    let url = "https://httpbin.org/range/102400?duration=2";
    let file = PathBuf::from("download.bin");
    let progress = |p| {
        println!("Progress: {:?}", p);
    };
    let client = reqwest::Client::new();
    let response = client.head(url).send().await.unwrap();
    download_into(
        Url::from_str(url).unwrap(),
        &file,
        response,
        client,
        None,
        progress,
    )
    .await
    .unwrap();
    assert!(file.exists());
    tokio::fs::remove_file(file).await.unwrap();
}

fn huggingface_token() -> Option<String> {
    let cache = hf_hub::Cache::default();
    cache.token().or_else(|| std::env::var("HF_TOKEN").ok())
}
