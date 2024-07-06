//! Handles chunking audio with a voice audio detection model
use std::{
    pin::Pin,
    task::{Context, Poll},
};

use futures_core::{ready, Stream};
use rodio::buffer::SamplesBuffer;
use voice_activity_detector::VoiceActivityDetector;

use crate::{AsyncSource, ResampledAsyncSource};

/// An extension trait for audio streams that adds a voice activity detection information
pub trait VoiceActivityDetectorExt: AsyncSource {
    /// Transform the audio stream to a stream of [`SamplesBuffer`]s with voice activity detection information
    fn voice_activity_stream(self) -> VoiceActivityDetectorStream<ResampledAsyncSource<Self>>
    where
        Self: Sized + Unpin,
    {
        let (source, closest) = resample_to_nearest_supported_rate(self);
        let vad = closest.vad();

        VoiceActivityDetectorStream::new(source, vad, closest.chunk_sizes[0])
    }
}

impl<S: AsyncSource> VoiceActivityDetectorExt for S {}

/// A stream of [`SamplesBuffer`]s with voice activity detection information
pub struct VoiceActivityDetectorStream<S: AsyncSource + Unpin> {
    source: S,
    buffer: Vec<f32>,
    chunk_size: usize,
    vad: VoiceActivityDetector,
}

impl<S: AsyncSource + Unpin> VoiceActivityDetectorStream<S> {
    fn new(source: S, vad: VoiceActivityDetector, chunk_size: usize) -> Self {
        Self {
            source,
            buffer: Vec::with_capacity(chunk_size),
            chunk_size,
            vad,
        }
    }
}

/// The output of a [`VoiceActivityDetectorStream`]
pub struct VoiceActivityDetectorOutput {
    /// The probability of voice activity (between 0 and 1)
    pub probability: f32,
    /// The audio sample associated with the voice activity probability
    pub samples: SamplesBuffer<f32>,
}

impl<S: AsyncSource + Unpin> Stream for VoiceActivityDetectorStream<S> {
    type Item = VoiceActivityDetectorOutput;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let sample_rate = this.source.sample_rate();
        let stream = this.source.as_stream();
        let mut stream = std::pin::pin!(stream);
        while this.buffer.len() < this.chunk_size {
            let sample = ready!(stream.as_mut().poll_next(cx));
            if let Some(sample) = sample {
                this.buffer.push(sample);
            } else {
                return Poll::Ready(None);
            }
        }
        let data = this.buffer.drain(..).collect::<Vec<_>>();
        let vad = this.vad.predict(data.iter().copied());
        let samples = SamplesBuffer::new(1, sample_rate, data);
        Poll::Ready(Some(VoiceActivityDetectorOutput {
            probability: vad,
            samples,
        }))
    }
}

/// Resample the audio to the closest supported sample rate
fn resample_to_nearest_supported_rate<S: AsyncSource + Unpin>(
    source: S,
) -> (ResampledAsyncSource<S>, SupportedSampleRate) {
    let sample_rate = source.sample_rate();
    let closet = SupportedSampleRate::closest(sample_rate);
    let resampled = source.resample(closet.sample_rate);
    (resampled, closet)
}

// let chunk = vec![0i16; 512];
// let mut vad = VoiceActivityDetector::builder()
//     .sample_rate(8000)
//     .chunk_size(512usize)
//     .build()?;
// let probability = vad.predict(chunk);
// println!("probability: {}", probability);

#[derive(Clone, Copy)]
struct SupportedSampleRate {
    sample_rate: u32,
    chunk_sizes: [usize; 3],
}

impl SupportedSampleRate {
    fn closest(sample_rate: u32) -> Self {
        *SUPPORTED_SAMPLE_RATES
            .iter()
            .min_by_key(|sr| (sr.sample_rate as i64 - sample_rate as i64).abs())
            .unwrap()
    }

    fn vad(&self) -> VoiceActivityDetector {
        VoiceActivityDetector::builder()
            .sample_rate(self.sample_rate)
            .chunk_size(self.chunk_sizes[0])
            .build()
            .unwrap()
    }
}

const SUPPORTED_SAMPLE_RATES: [SupportedSampleRate; 2] = [
    SupportedSampleRate {
        sample_rate: 8000,
        chunk_sizes: [256, 512, 768],
    },
    SupportedSampleRate {
        sample_rate: 16000,
        chunk_sizes: [512, 768, 1024],
    },
];
