use futures_core::ready;
use rodio::buffer::SamplesBuffer;
use std::{collections::VecDeque, task::Poll, time::Duration};

/// The output of a [`crate::VoiceActivityDetectorStream`]
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
    fn rechunk_voice_activity(self) -> VoiceActivityRechunkerStream<Self>
    where
        Self: Sized + Unpin,
    {
        let start_threshold = 0.6;
        let start_window = Duration::from_millis(100);
        let end_threshold = 0.2;
        let end_window = Duration::from_millis(2000);
        let time_before_speech = Duration::from_millis(500);
        VoiceActivityRechunkerStream::new(
            self,
            start_threshold,
            start_window,
            end_threshold,
            end_window,
            time_before_speech,
        )
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
    start_threshold: f32,
    start_window: Duration,
    end_threshold: f32,
    end_window: Duration,
    include_duration_before: Duration,
    duration_before_window: Duration,
    in_voice_run: bool,
    buffer: VecDeque<SamplesBuffer<f32>>,
    channels: u16,
    sample_rate: u32,
    voice_probabilities_window: VecDeque<(f32, Duration)>,
    duration_in_window: Duration,
    sum: f32,
}

impl<S> VoiceActivityRechunkerStream<S> {
    /// Set the threshold for the start of a voice activity run
    pub fn with_start_threshold(mut self, start_threshold: f32) -> Self {
        self.start_threshold = start_threshold;
        self
    }

    /// Set the window for the start of a voice activity run
    pub fn with_start_window(mut self, start_window: Duration) -> Self {
        self.start_window = start_window;
        self
    }

    /// Set the threshold for the end of a voice activity run
    pub fn with_end_threshold(mut self, end_threshold: f32) -> Self {
        self.end_threshold = end_threshold;
        self
    }

    /// Set the window for the end of a voice activity run
    pub fn with_end_window(mut self, end_window: Duration) -> Self {
        self.end_window = end_window;
        self
    }

    /// Set the time before the speech run starts to include in the output
    pub fn with_time_before_speech(mut self, time_before_speech: Duration) -> Self {
        self.include_duration_before = time_before_speech;
        self
    }
}

impl<S> VoiceActivityRechunkerStream<S> {
    fn new(
        source: S,
        start_threshold: f32,
        start_window: Duration,
        end_threshold: f32,
        end_window: Duration,
        include_duration_before: Duration,
    ) -> Self {
        Self {
            source,
            start_threshold,
            start_window,
            end_threshold,
            end_window,
            include_duration_before,
            duration_before_window: Duration::ZERO,
            in_voice_run: false,
            buffer: VecDeque::new(),
            channels: 1,
            sample_rate: 0,
            voice_probabilities_window: VecDeque::new(),
            duration_in_window: Duration::ZERO,
            sum: 0.0,
        }
    }

    fn add_sample(&mut self, probability: f32, len: Duration, window: Duration) {
        // Add the samples to the rolling average
        self.voice_probabilities_window
            .push_front((probability, len));
        self.sum += probability;
        self.duration_in_window += len;
        // If the buffer is full, remove the first probability from the rolling average
        while self.duration_in_window > window {
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
        self.sum = 0.0;
        self.duration_in_window = Duration::ZERO;
        self.voice_probabilities_window.clear();
        self.in_voice_run = false;
        self.duration_before_window = Duration::ZERO;
        self.buffer.clear();
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
                let sample_duration = rodio::Source::total_duration(&next.samples)
                    .expect("samples must have a duration");
                let window = if this.in_voice_run {
                    this.end_window
                } else {
                    this.start_window
                };
                this.add_sample(next.probability, sample_duration, window);
                // If we are inside a chunk that looks like voice, set the in voice run flag
                if this.rolling_average() > this.start_threshold {
                    this.in_voice_run = true;
                }
                // Add the samples to the buffer
                this.buffer.push_back(next.samples);
                // If this is inside a voice run, add the sample to the buffer
                if this.in_voice_run {
                    // Otherwise, if we just left a chunk that looks like voice, add the buffer to the output
                    if this.rolling_average() < this.end_threshold {
                        let samples = this.finish_voice_run();
                        return Poll::Ready(Some(samples));
                    }
                } else {
                    // Otherwise, add it to the pre-voice buffer
                    this.duration_before_window += sample_duration;
                    // If the pre-voice buffer is full, remove the first sample from it
                    while this.duration_before_window >= this.include_duration_before {
                        let sample = this.buffer.pop_front().unwrap();
                        this.duration_before_window -= rodio::Source::total_duration(&sample)
                            .expect("samples must have a duration");
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
