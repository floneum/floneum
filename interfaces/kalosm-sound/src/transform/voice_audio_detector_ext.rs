use futures_core::ready;
use rodio::buffer::SamplesBuffer;
use std::{collections::VecDeque, task::Poll, time::Duration};

/// The output of a [`VoiceActivityDetectorStream`]
pub struct VoiceActivityDetectorOutput {
    /// The probability of voice activity (between 0 and 1)
    pub probability: f32,
    /// The audio sample associated with the voice activity probability
    pub samples: rodio::buffer::SamplesBuffer<f32>,
}

/// An extension trait for audio streams with voice activity detection information
pub trait VoiceActivityStreamExt: futures_core::Stream<Item = VoiceActivityDetectorOutput> {
    /// Only keep audio chunks that have a probability of voice activity above the given threshold
    fn filter_voice_activity(self, threshold: f32) -> VoiceActivityFilterStream<Self>
    where
        Self: Sized + Unpin,
    {
        VoiceActivityFilterStream::new(self, threshold)
    }

    /// Rechunk the audio into chunks of audio with a rolling average over the given duration more than the given threshold
    fn rechunk_voice_activity(
        self,
        chunk_size: Duration,
        threshold: f32,
    ) -> VoiceActivityRechunkerStream<Self>
    where
        Self: Sized + Unpin,
    {
        VoiceActivityRechunkerStream::new(self, chunk_size, threshold)
    }
}

impl<S: futures_core::Stream<Item = VoiceActivityDetectorOutput>> VoiceActivityStreamExt for S {}

/// A stream of audio chunks that have a voice activity probability above a given threshold
pub struct VoiceActivityFilterStream<S> {
    source: S,
    threshold: f32,
}

impl<S> VoiceActivityFilterStream<S> {
    fn new(source: S, threshold: f32) -> Self {
        Self { source, threshold }
    }
}

impl<S: futures_core::Stream<Item = VoiceActivityDetectorOutput> + Unpin> futures_core::Stream
    for VoiceActivityFilterStream<S>
{
    type Item = SamplesBuffer<f32>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let mut source = std::pin::pin!(&mut this.source);
        loop {
            let next = ready!(source.as_mut().poll_next(cx));
            if let Some(next) = next {
                if next.probability > this.threshold {
                    return Poll::Ready(Some(next.samples));
                }
            } else {
                return Poll::Ready(None);
            }
        }
    }
}

/// A stream of audio chunks with a voice activity probability rolling average above a given threshold
pub struct VoiceActivityRechunkerStream<S> {
    source: S,
    threshold: f32,
    chunk_size: Duration,
    in_voice_run: bool,
    buffer: Vec<SamplesBuffer<f32>>,
    channels: u16,
    sample_rate: u32,
    voice_probabilities_window: VecDeque<(f32, Duration)>,
    duration_in_window: Duration,
    sum: f32,
}

impl<S> VoiceActivityRechunkerStream<S> {
    fn new(source: S, chunk_size: Duration, threshold: f32) -> Self {
        Self {
            source,
            threshold,
            chunk_size,
            in_voice_run: false,
            buffer: Vec::new(),
            channels: 1,
            sample_rate: 0,
            voice_probabilities_window: VecDeque::new(),
            duration_in_window: Duration::ZERO,
            sum: 0.0,
        }
    }

    fn add_sample(&mut self, probability: f32, len: Duration) {
        // Add the samples to the rolling average
        self.voice_probabilities_window
            .push_front((probability, len));
        self.sum += probability;
        self.duration_in_window += len;
        // If the buffer is full, remove the first probability from the rolling average
        while self.duration_in_window >= self.chunk_size {
            self.pop_last_sample();
        }
    }

    fn pop_last_sample(&mut self) {
        let (probability, len) = self.voice_probabilities_window.pop_back().unwrap();
        self.sum -= probability;
        self.duration_in_window -= len;
    }

    fn rolling_average(&self) -> f32 {
        self.sum / self.voice_probabilities_window.len() as f32
    }

    fn finish_voice_run(&mut self) -> SamplesBuffer<f32> {
        let samples = SamplesBuffer::new(
            self.channels,
            self.sample_rate,
            std::mem::take(&mut self.buffer)
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );
        self.in_voice_run = false;
        samples
    }
}

impl<S: futures_core::Stream<Item = VoiceActivityDetectorOutput> + Unpin> futures_core::Stream
    for VoiceActivityRechunkerStream<S>
{
    type Item = SamplesBuffer<f32>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            let source = std::pin::pin!(&mut this.source);
            let next = ready!(source.poll_next(cx));
            if let Some(next) = next {
                // Set the sample rate from the stream
                this.sample_rate = rodio::Source::sample_rate(&next.samples);
                this.add_sample(
                    next.probability,
                    rodio::Source::total_duration(&next.samples).unwrap(),
                );

                // Add the samples to the buffer
                this.buffer.push(next.samples);
                // If we are inside a chunk that looks like voice, set the in voice run flag
                if this.rolling_average() > this.threshold {
                    this.in_voice_run = true;
                }
                // Otherwise, if we just left a chunk that looks like voice, add the buffer to the output
                else if this.in_voice_run {
                    let samples = this.finish_voice_run();
                    return Poll::Ready(Some(samples));
                }
                // Or if not, remove the first sample from the buffer while the probability is very low
                else {
                    this.buffer.remove(0);
                    while this.rolling_average() < this.threshold / 2. && this.buffer.len() > 1 {
                        this.buffer.remove(0);
                        this.pop_last_sample();
                    }
                }
            } else {
                // Finish off the current voice run if there is one
                if this.in_voice_run {
                    let samples = this.finish_voice_run();
                    return Poll::Ready(Some(samples));
                }
                // Otherwise, return None and finish the stream
                return Poll::Ready(None);
            }
        }
    }
}
