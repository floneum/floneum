//! Common types for Kalosm models

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
    pub start_time: std::time::Instant,
    /// The size of the cached part of the download in bytes
    pub cached_size: u64,
    /// The size of the download in bytes
    pub size: u64,
    /// The progress of the download in bytes, from 0 to size
    pub progress: u64,
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

    #[cfg(feature = "loading-progress-bar")]
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
