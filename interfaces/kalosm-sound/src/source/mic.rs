use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SizedSample,
};
use dasp::sample::ToSample;
use futures_channel::mpsc;
use futures_core::Stream;
use futures_util::StreamExt;
use rodio::buffer::SamplesBuffer;
use std::{pin::Pin, sync::Arc};

use tokio::time::Instant;

use crate::AsyncSource;

/// A microphone input.
pub struct MicInput {
    #[allow(dead_code)]
    host: cpal::Host,
    // Some older versions of cpal::Device don't implement Clone on some platforms, so we need to use Arc
    device: Arc<cpal::Device>,
    config: cpal::SupportedStreamConfig,
}

impl Default for MicInput {
    fn default() -> Self {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .expect("Failed to get default input device");
        let config = device
            .default_input_config()
            .expect("Failed to get default input config");
        Self {
            host,
            device: Arc::new(device),
            config,
        }
    }
}

impl MicInput {
    /// Records audio for a given duration.
    pub async fn record_until(
        &self,
        deadline: Instant,
    ) -> Result<SamplesBuffer<f32>, anyhow::Error> {
        let mut stream = self.stream()?;
        tokio::time::sleep_until(deadline).await;
        Ok(stream.read_all())
    }

    /// Records audio for a given duration.
    pub fn record_until_blocking(
        &self,
        deadline: std::time::Instant,
    ) -> Result<SamplesBuffer<f32>, anyhow::Error> {
        let mut stream = self.stream()?;
        std::thread::sleep(deadline - std::time::Instant::now());
        Ok(stream.read_all())
    }

    /// Creates a new stream of audio data from the microphone.
    pub fn stream(&self) -> Result<MicStream, anyhow::Error> {
        let (tx, rx) = mpsc::unbounded::<Vec<f32>>();

        let config = self.config.clone();
        let device = self.device.clone();
        let (drop_tx, drop_rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            fn build_stream<S: ToSample<f32> + SizedSample>(
                device: &cpal::Device,
                config: &cpal::SupportedStreamConfig,
                mut tx: mpsc::UnboundedSender<Vec<f32>>,
            ) -> Result<cpal::Stream, cpal::BuildStreamError> {
                let channels = config.channels() as usize;
                device.build_input_stream::<S, _, _>(
                    &config.config(),
                    move |data: &[S], _: &_| {
                        let _ = tx.start_send(
                            data.iter()
                                .step_by(channels)
                                .map(|&x| x.to_sample())
                                .collect(),
                        );
                    },
                    |err| {
                        tracing::error!("an error occurred on stream: {}", err);
                    },
                    None,
                )
            }

            let start_stream = || {
                let stream = match config.sample_format() {
                    cpal::SampleFormat::I8 => build_stream::<i8>(&device, &config, tx)?,
                    cpal::SampleFormat::I16 => build_stream::<i16>(&device, &config, tx)?,
                    cpal::SampleFormat::I32 => build_stream::<i32>(&device, &config, tx)?,
                    cpal::SampleFormat::F32 => build_stream::<f32>(&device, &config, tx)?,
                    sample_format => {
                        return Err(anyhow::Error::msg(format!(
                            "Unsupported sample format '{sample_format}'"
                        )))
                    }
                };

                stream.play()?;

                Ok(stream)
            };

            let stream = match start_stream() {
                Ok(stream) => stream,
                Err(err) => {
                    tracing::error!("Error starting stream: {}", err);
                    return;
                }
            };

            // Wait for the stream to be dropped
            drop_rx.recv().unwrap();

            // Then drop the stream
            drop(stream);
        });

        let receiver = rx.map(futures_util::stream::iter).flatten();
        Ok(MicStream {
            drop_tx,
            config: self.config.clone(),
            receiver: Box::pin(receiver),
            read_data: Vec::new(),
        })
    }
}

/// A stream of audio data from the microphone.
pub struct MicStream {
    drop_tx: std::sync::mpsc::Sender<()>,
    config: cpal::SupportedStreamConfig,
    read_data: Vec<f32>,
    receiver: Pin<Box<dyn futures_core::Stream<Item = f32> + Send + Sync>>,
}

impl Drop for MicStream {
    fn drop(&mut self) {
        self.drop_tx.send(()).unwrap();
    }
}

impl Stream for MicStream {
    type Item = f32;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match self.receiver.as_mut().poll_next_unpin(cx) {
            std::task::Poll::Ready(Some(data_chunk)) => {
                self.read_data.push(data_chunk);
                std::task::Poll::Ready(Some(data_chunk))
            }
            std::task::Poll::Ready(None) => std::task::Poll::Ready(None),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

impl MicStream {
    /// Read any pending data from the stream into a vector
    fn read_sync(&mut self) -> Vec<f32> {
        let mut cx = std::task::Context::from_waker(futures_util::task::noop_waker_ref());
        while let std::task::Poll::Ready(Some(data_chunk)) = self.receiver.poll_next_unpin(&mut cx)
        {
            self.read_data.push(data_chunk);
        }
        self.read_data.clone()
    }

    /// Grab all current data in the stream
    fn read_all_samples(&mut self) -> Vec<f32> {
        self.read_sync()
    }

    fn read_all(&mut self) -> SamplesBuffer<f32> {
        let channels = self.config.channels();
        let sample_rate = self.config.sample_rate().0;
        SamplesBuffer::new(channels, sample_rate, self.read_all_samples())
    }
}

impl AsyncSource for MicStream {
    fn as_stream(&mut self) -> impl Stream<Item = f32> + '_ {
        self
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate().0
    }
}

#[test]
fn assert_mic_stream_send_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<MicStream>();
    fn assert_send<T: Send>() {}
    assert_send::<MicStream>();
}
