use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample};

use std::io::{Cursor, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};
use tokio::time::Instant;

use crate::AudioBuffer;

pub struct MicInput {
    #[allow(dead_code)]
    host: cpal::Host,
    device: cpal::Device,
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
            device,
            config,
        }
    }
}

impl MicInput {
    pub async fn record_until(&self, deadline: Instant) -> Result<AudioBuffer, anyhow::Error> {
        let stream = self.stream()?;
        tokio::time::sleep_until(deadline).await;
        let bytes = stream.finalize()?;
        Ok(bytes)
    }

    pub fn record_until_blocking(
        &self,
        deadline: std::time::Instant,
    ) -> Result<AudioBuffer, anyhow::Error> {
        let stream = self.stream()?;
        std::thread::sleep(deadline - std::time::Instant::now());
        let bytes = stream.finalize()?;
        Ok(bytes)
    }

    pub fn stream(&self) -> Result<MicStream, anyhow::Error> {
        let err_fn = move |err| {
            eprintln!("an error occurred on stream: {}", err);
        };
        let mut spec = wav_spec_from_config(&self.config);
        spec.bits_per_sample = 16;
        spec.sample_format = cpal::SampleFormat::Int;
        let bytes = SharedBuf::default();
        let writer = Arc::new(Mutex::new(Some(bytes)));
        let writer_2 = writer.clone();

        let stream = match self.config.sample_format() {
            cpal::SampleFormat::I8 => self.device.build_input_stream(
                &self.config.config(),
                move |data, _: &_| write_input_data::<i8, i16>(data, &writer_2),
                err_fn,
                None,
            )?,
            cpal::SampleFormat::I16 => self.device.build_input_stream(
                &self.config.config(),
                move |data, _: &_| write_input_data::<i16, i16>(data, &writer_2),
                err_fn,
                None,
            )?,
            cpal::SampleFormat::I32 => self.device.build_input_stream(
                &self.config.config(),
                move |data, _: &_| write_input_data::<i32, i16>(data, &writer_2),
                err_fn,
                None,
            )?,
            cpal::SampleFormat::F32 => self.device.build_input_stream(
                &self.config.config(),
                move |data, _: &_| write_input_data::<f32, i16>(data, &writer_2),
                err_fn,
                None,
            )?,
            sample_format => {
                return Err(anyhow::Error::msg(format!(
                    "Unsupported sample format '{sample_format}'"
                )))
            }
        };

        stream.play()?;

        Ok(MicStream {
            stream,
            writer,
            buf: bytes,
        })
    }
}

#[derive(Default, Clone)]
struct SharedBuf {
    buf: Arc<Mutex<Cursor<Vec<u8>>>>,
}

impl Write for SharedBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buf.lock().unwrap().write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.buf.lock().unwrap().flush()
    }
}

impl Seek for SharedBuf {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.buf.lock().unwrap().seek(pos)
    }
}

pub struct MicStream {
    stream: cpal::Stream,
    writer: WavWriterHandle,
    buf: SharedBuf,
}

impl MicStream {
    pub fn finalize(self) -> Result<AudioBuffer, anyhow::Error> {
        drop(self.stream);

        self.writer.lock().unwrap().take().unwrap().finalize()?;
        let old = std::mem::take(&mut *self.buf.buf.lock().unwrap());
        Ok(AudioBuffer::new(old.into_inner()))
    }
}

#[test]
fn record() -> Result<(), anyhow::Error> {
    let input = MicInput::default();
    let stream = input.stream()?;
    std::thread::sleep(std::time::Duration::from_secs(3));
    let bytes = stream.finalize()?;
    println!("Got {} bytes", bytes.data().len());
    Ok(())
}
