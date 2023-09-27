use anyhow::Result;
use cpal::FromSample;
use floneumin_streams::sender::ChannelTextStream;
use futures_util::{Stream, StreamExt};
use rodio::Source;

use self::whisper::WhisperModel;

pub mod whisper;

#[derive(Debug, Clone)]
struct DecodingResult {
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    compression_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct Segment {
    start: f64,
    duration: f64,
    result: DecodingResult,
}

impl Segment {
    pub fn probability_of_no_speech(&self) -> f64 {
        self.result.no_speech_prob
    }

    pub fn text(&self) -> &str {
        &self.result.text
    }

    pub fn start(&self) -> f64 {
        self.start
    }

    pub fn duration(&self) -> f64 {
        self.duration
    }
}

impl AsRef<str> for Segment {
    fn as_ref(&self) -> &str {
        self.text()
    }
}

#[async_trait::async_trait(?Send)]
pub trait TranscribeAudioSourceExt {
    fn text(self, model: WhisperModel) -> Result<ChannelTextStream<Segment>>;
}

#[async_trait::async_trait(?Send)]
impl<S: Source> TranscribeAudioSourceExt for S
where
    <S as Iterator>::Item: rodio::Sample,
    f32: FromSample<<S as Iterator>::Item>,
{
    fn text(self, model: WhisperModel) -> Result<ChannelTextStream<Segment>> {
        model.transcribe(self)
    }
}

pub trait TranscribeAudioStreamExt {
    fn text(self, model: WhisperModel) -> ChannelTextStream<Segment>;
}

impl<S> TranscribeAudioStreamExt for S
where
    S: Stream + std::marker::Unpin + Send + 'static,
    <S as Stream>::Item: Source + Send + 'static,
    <<S as Stream>::Item as Iterator>::Item: rodio::Sample,
    f32: FromSample<<<S as Stream>::Item as Iterator>::Item>,
{
    fn text(self, model: WhisperModel) -> ChannelTextStream<Segment> {
        let mut stream = self;
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Some(source) = stream.next().await {
                match model.transcribe(source) {
                    Ok(mut stream) => {
                        while let Some(segment) = stream.next().await {
                            if let Err(err) = sender.send(segment) {
                                tracing::error!("error sending segment: {}", err);
                                return;
                            }
                        }
                    }
                    Err(err) => tracing::error!("error transcribing audio: {}", err),
                }
            }
        });
        ChannelTextStream::from(receiver)
    }
}

// let mut chunks = stream.stream().subscribe_stream(Duration::from_secs(30));

//     let mut current_time_stamp = 0.0;

//     println!("starting transcription loop");
//     loop {
//         let mut next_time_stamp = current_time_stamp;
//         let input = chunks
//             .next()
//             .await
//             .ok_or_else(|| anyhow::anyhow!("no more chunks"))?;

//         let mut transcribed = model.transcribe(input).await?;

//         while let Some(transcribed) = transcribed.next().await {
//             if transcribed.probability_of_no_speech() < 0.90 {
//                 let start = current_time_stamp + transcribed.start();
//                 let end = start + transcribed.duration();
//                 next_time_stamp = end;

//                 let text = transcribed.text();
//                 println!("({:01} - {:01}): {}", start, end, text);
//             }
//         }

//         current_time_stamp = next_time_stamp;
//     }
