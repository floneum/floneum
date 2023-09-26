use std::io::Cursor;

use cpal::Sample;
use ringbuffer::GrowableAllocRingBuffer;

trait AudioData {
    fn read(&mut self) -> anyhow::Result<u8>;
}

pub struct AudioSpec {
    sample_rate: u32,
    bits_per_sample: u16,
    float: bool,
    channels: u16,
}

impl AudioSpec {
    pub fn sample_size_bytes(&self) -> u16 {
        self.bits_per_sample / 8
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn float(&self) -> bool {
        self.float
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }
}

pub struct AudioStream<S: Sample> {
    buffer: GrowableAllocRingBuffer<S>,
    spec: AudioSpec,
}

impl<S: Sample> AudioStream<S> {
    pub fn new(seconds: f32, spec: impl Into<AudioSpec>) -> Self {
        let spec = spec.into();
        Self {
            buffer: GrowableAllocRingBuffer::with_capacity(
                seconds as usize * spec.sample_size_bytes() as usize * spec.sample_rate() as usize,
            ),
            spec,
        }
    }

    pub fn write(&mut self, data: impl IntoIterator<Item = S>) {
        self.buffer.extend(data)
    }

    pub fn reader(&self) -> anyhow::Result<rodio::buffer::SamplesBuffer<f32>>
    where
        <S as cpal::Sample>::Float: Into<f32>,
    {
        let samples: Vec<_> = self
            .buffer
            .iter()
            .map(|s| s.to_float_sample().into())
            .collect();
        Ok(rodio::buffer::SamplesBuffer::new(
            self.spec.channels as u16,
            self.spec.sample_rate,
            samples,
        ))
    }
}

pub struct AudioBuffer {
    data: Vec<u8>,
}

impl From<Vec<u8>> for AudioBuffer {
    fn from(data: Vec<u8>) -> Self {
        Self::new(data)
    }
}

impl AudioBuffer {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, anyhow::Error> {
        Ok(Self::new(std::fs::read(path)?))
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    pub fn into_reader(self) -> Result<hound::WavReader<Cursor<Vec<u8>>>, anyhow::Error> {
        Ok(hound::WavReader::new(Cursor::new(self.data))?)
    }
}
