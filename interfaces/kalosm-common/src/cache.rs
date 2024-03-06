use anyhow::bail;
use hf_hub::{Repo, RepoType};
use reqwest::header::{HeaderValue, CONTENT_LENGTH, RANGE};
use reqwest::{StatusCode, Url};
use std::path::PathBuf;
use std::str::FromStr;
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;

use crate::FileSource;

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
        progress: impl FnMut(f32) + Send + Sync,
    ) -> anyhow::Result<PathBuf> {
        match source {
            FileSource::HuggingFace {
                model_id,
                revision,
                file,
            } => {
                let path = self.location.join(model_id).join(revision);
                let complete_download = path.join(file);
                if complete_download.exists() {
                    return Ok(complete_download);
                }
                let incomplete_download = path.join(format!("{}.partial", file));

                let api = hf_hub::api::sync::Api::new()?;
                let repo = Repo::with_revision(
                    model_id.to_string(),
                    RepoType::Model,
                    revision.to_string(),
                );
                let api = api.repo(repo);
                let url = api.url(file);
                let url = Url::from_str(&url)?;

                tracing::trace!("Downloading into {:?}", incomplete_download);

                download_into(url, &incomplete_download, progress).await?;

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
        }
    }
}

impl FileSource {
    /// Check if the file exists locally (if it is a local file or if it has been downloaded)
    pub async fn download(
        &self,
        progress: impl FnMut(f32) + Send + Sync,
    ) -> anyhow::Result<PathBuf> {
        let cache = Cache::default();
        cache.get(self, progress).await
    }
}

async fn download_into(
    url: Url,
    file: &PathBuf,
    mut progress: impl FnMut(f32) + Send + Sync,
) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let response = client.head(url.clone()).send().await.unwrap();
    let length = response
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
    download_into(Url::from_str(url).unwrap(), &file, progress)
        .await
        .unwrap();
    assert!(file.exists());
    tokio::fs::remove_file(file).await.unwrap();
}
