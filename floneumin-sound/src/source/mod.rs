use std::io::Cursor;

pub mod mic;

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
