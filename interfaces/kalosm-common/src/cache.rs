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

#[derive(Debug)]
/// The progress starting a model
#[derive(Clone)]
pub enum ModelLoadingProgress {
    /// The model is downloading
    Downloading {
        /// The source of the download. This is not a path or URL, but a description of the source
        source: String,
        /// The time stamp the download started
        start_time: std::time::Instant,
        /// The progress of the download, from 0 to 1
        progress: f32,
    },
    /// The model is loading
    Loading {
        /// The progress of the loading, from 0 to 1
        progress: f32,
    },
}

impl ModelLoadingProgress {
    /// Create a new downloading progress
    pub fn downloading(source: String, progress: f32, start_time: std::time::Instant) -> Self {
        Self::Downloading {
            source,
            progress,
            start_time,
        }
    }

    /// Create a new downloading progress
    pub fn downloading_progress(source: String) -> impl FnMut(f32) -> Self + Send + Sync {
        let start = std::time::Instant::now();
        move |progress| ModelLoadingProgress::downloading(source.clone(), progress, start)
    }

    /// Create a new loading progress
    pub fn loading(progress: f32) -> Self {
        Self::Loading { progress }
    }

    /// Try to estimate the time remaining for a download
    pub fn estimate_time_remaining(&self) -> Option<std::time::Duration> {
        match self {
            Self::Downloading {
                start_time,
                progress,
                ..
            } => {
                let elapsed = start_time.elapsed();
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
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})",
        )
        .unwrap();
        let mut progress_bars = HashMap::new();

        move |progress| match progress {
            ModelLoadingProgress::Downloading {
                source, progress, ..
            } => {
                let n = 100;
                let progress = progress * n as f32;

                let progress_bar = progress_bars.entry(source.clone()).or_insert_with(|| {
                    let pb = m.add(ProgressBar::new(n));
                    pb.set_message(format!("Downloading {source}"));
                    pb.set_style(sty.clone());
                    pb
                });

                progress_bar.set_position(progress as u64);
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

pub struct Cache {
    location: PathBuf,
}

impl Cache {
    /// Create a new cache with a specific location
    pub fn new(location: PathBuf) -> Self {
        Self { location }
    }

    /// Check if the file exists locally (if it is a local file or if it has been downloaded)
    pub fn exists(&self, source: &FileSource) -> bool {
        match source {
            FileSource::HuggingFace {
                model_id,
                revision,
                file,
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
        progress: impl FnMut(f32),
    ) -> anyhow::Result<PathBuf> {
        match source {
            FileSource::HuggingFace {
                model_id,
                revision,
                file,
            } => {
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
                let response = client.head(url.clone()).send().await;

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

                download_into(url, &incomplete_download, response?, progress).await?;

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
            location: dirs::cache_dir().unwrap().join("kalosm").join("cache"),
        }
    }
}

impl FileSource {
    /// Check if the file exists locally (if it is a local file or if it has been downloaded)
    pub async fn download(&self, progress: impl FnMut(f32)) -> anyhow::Result<PathBuf> {
        let cache = Cache::default();
        cache.get(self, progress).await
    }
}

async fn download_into(
    url: Url,
    file: &PathBuf,
    head: Response,
    mut progress: impl FnMut(f32),
) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
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
        progress(start as f32 / length as f32);
    }

    if Some(start) == length {
        tracing::trace!("File {} already downloaded", file.display());
        progress(1.0);
        return Ok(());
    }

    let range = length
        .and_then(|length| HeaderValue::from_str(&format!("bytes={}-{}", start, length - 1)).ok());

    tracing::trace!("Fetching range {:?}", range);
    let mut request = client.get(url);
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
            progress(current_progress as f32 / length as f32);
        }
    }

    tracing::trace!("Download of {} complete", file.display());
    progress(1.0);

    Ok(())
}

#[tokio::test]
async fn downloads_work() {
    let url = "https://httpbin.org/range/102400?duration=2";
    let file = PathBuf::from("download.bin");
    let progress = |p| {
        println!("Progress: {}", p);
    };
    let client = reqwest::Client::new();
    let response = client.head(url).send().await.unwrap();
    download_into(Url::from_str(url).unwrap(), &file, response, progress)
        .await
        .unwrap();
    assert!(file.exists());
    tokio::fs::remove_file(file).await.unwrap();
}
