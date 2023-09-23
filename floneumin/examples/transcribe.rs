
use floneumin_sound::source::mic::MicInput;
use tokio::time::{Duration, Instant};

#[tokio::main]
async fn main(){
    let mut model = WhisperBuilder::default().build()?;

    println!("recording");
    let input =
        MicInput::default().record_until(Instant::now() + Duration::from_secs(20)).await?;

    println!("transcribing");

    let start_time = Instant::now();

    let transcribed = model.transcribe(input)?;

    println!("transcribed: {:?}", transcribed);
    println!("took: {:?}", start_time.elapsed());
}
