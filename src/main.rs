use futures_util::stream::StreamExt;
use llm::{
    InferenceFeedback, InferenceRequest, InferenceResponse, InferenceStats, LoadProgress, Model,
    ModelArchitecture,
};
use rustyline::error::ReadlineError;
use spinoff::{spinners::Dots2, Spinner};
use std::{convert::Infallible, error::Error, io::Write, path::PathBuf, time::Instant};
use tokio::{fs::File, io::AsyncWriteExt};

mod plugins;

async fn download(model_type: ModelArchitecture) -> Box<dyn Model> {
    let url = match model_type {
        ModelArchitecture::Bloom => "https://huggingface.co/nouamanetazi/bloomz-560m-ggml/resolve/main/ggml-model-bloomz-560m-f16-q4_0.bin",
        ModelArchitecture::Gpt2 => todo!(),
        ModelArchitecture::GptJ => "https://huggingface.co/Kastor/GPT-J-6B-Pygway-ggml-q4_1/resolve/main/GPT-J-6B-Pygway-ggml-q4_0.bin",
        ModelArchitecture::GptNeoX => "https://huggingface.co/oeathus/stablelm-base-alpha-7b-ggml-q4/resolve/main/ggml-model-q4_0.bin",
        ModelArchitecture::Llama => todo!(),
        ModelArchitecture::Mpt => "https://huggingface.co/LLukas22/mpt-7b-ggml/resolve/main/mpt-7b-storywriter-q4_0.bin",
    };

    let path = download_model(url).await.unwrap();

    let overrides = None;

    let sp = Some(Spinner::new(Dots2, "Loading model...", None));

    let now = Instant::now();
    let prev_load_time = now;

    llm::load_dynamic(
        model_type,
        &path,
        Default::default(),
        overrides,
        load_progress_callback(sp, now, prev_load_time),
    )
    .unwrap_or_else(|err| panic!("Failed to load {model_type} model from {path:?}: {err}"))
}

async fn download_model(model_url: &str) -> Result<PathBuf, Box<dyn Error>> {
    let path: PathBuf = format!("./{}", model_url.rsplit_once('/').unwrap().1).into();
    if path.exists() {
        return Ok(path);
    }
    let response = reqwest::get(model_url).await?;
    println!("downloading model. This will take several minutes");

    let mut file = { File::create(&path).await? };

    let size = response.content_length().unwrap_or(4_294_967_296) as usize;

    let mut stream = response.bytes_stream();
    let mut current_size = 0;
    let mut old_precent = 0;
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        current_size += chunk.len();
        file.write_all(&chunk).await?;
        let new_precent = current_size * 100 / size;
        if old_precent != new_precent {
            println!("{}%", new_precent);
        }
        old_precent = new_precent;
    }

    file.flush().await?;

    println!("Finished Downloading");

    Ok(path)
}

#[tokio::main]
async fn main() {
    let model_architecture = ModelArchitecture::Mpt;
    let model = download(model_architecture).await;

    let mut session = model.start_session(Default::default());

    let character_name = "### Assistant";
    let user_name = "### Human";
    let persona = include_str!("../prompt.txt");
    let history = format!("{character_name}: Hello - How may I help you today?\n");

    session
        .feed_prompt(
            model.as_ref(),
            &Default::default(),
            format!("{persona}\n{history}").as_str(),
            &mut Default::default(),
            llm::feed_prompt_callback(prompt_callback),
        )
        .unwrap();

    let mut rl = rustyline::DefaultEditor::new().expect("Failed to create input reader");

    let mut rng = rand::thread_rng();
    let mut res = InferenceStats::default();
    let mut buf = String::new();

    loop {
        println!();
        let readline = rl.readline(format!("{user_name}: ").as_str());
        print!("{character_name}:");
        match readline {
            Ok(line) => {
                let stats = session
                    .infer(
                        model.as_ref(),
                        &mut rng,
                        &InferenceRequest {
                            prompt: format!("{user_name}: {line}\n{character_name}:")
                                .as_str()
                                .into(),
                            ..Default::default()
                        },
                        &mut Default::default(),
                        inference_callback(String::from(user_name), &mut buf),
                    )
                    .unwrap_or_else(|e| panic!("{e}"));

                res.feed_prompt_duration = res
                    .feed_prompt_duration
                    .saturating_add(stats.feed_prompt_duration);
                res.prompt_tokens += stats.prompt_tokens;
                res.predict_duration = res.predict_duration.saturating_add(stats.predict_duration);
                res.predict_tokens += stats.predict_tokens;
            }
            Err(ReadlineError::Eof) | Err(ReadlineError::Interrupted) => {
                break;
            }
            Err(err) => {
                println!("{err}");
            }
        }
    }

    println!("\n\nInference stats:\n{res}");
}

fn load_progress_callback(
    mut sp: Option<Spinner>,
    now: Instant,
    mut prev_load_time: Instant,
) -> impl FnMut(LoadProgress) {
    move |progress| match progress {
        LoadProgress::HyperparametersLoaded => {
            if let Some(sp) = sp.as_mut() {
                sp.update_text("Loaded hyperparameters")
            };
        }
        LoadProgress::ContextSize { bytes } => log::debug!(
            "ggml ctx size = {}",
            bytesize::to_string(bytes as u64, false)
        ),
        LoadProgress::TensorLoaded {
            current_tensor,
            tensor_count,
            ..
        } => {
            if prev_load_time.elapsed().as_millis() > 500 {
                // We don't want to re-render this on every message, as that causes the
                // spinner to constantly reset and not look like it's spinning (and
                // it's obviously wasteful).
                if let Some(sp) = sp.as_mut() {
                    sp.update_text(format!(
                        "Loaded tensor {}/{}",
                        current_tensor + 1,
                        tensor_count
                    ));
                };
                prev_load_time = std::time::Instant::now();
            }
        }
        LoadProgress::LoraApplied { name, source } => {
            if let Some(sp) = sp.as_mut() {
                sp.update_text(format!(
                    "Applied LoRA: {} from '{}'",
                    name,
                    source.file_name().unwrap().to_str().unwrap()
                ));
            };
        }
        LoadProgress::Loaded {
            file_size,
            tensor_count,
        } => {
            if let Some(sp) = sp.take() {
                sp.success(&format!(
                    "Loaded {tensor_count} tensors ({}) after {}ms",
                    bytesize::to_string(file_size, false),
                    now.elapsed().as_millis()
                ));
            };
        }
    }
}

fn prompt_callback(resp: InferenceResponse) -> Result<InferenceFeedback, Infallible> {
    match resp {
        InferenceResponse::PromptToken(t) | InferenceResponse::InferredToken(t) => print_token(t),
        _ => Ok(InferenceFeedback::Continue),
    }
}

#[allow(clippy::needless_lifetimes)]
fn inference_callback<'a>(
    stop_sequence: String,
    buf: &'a mut String,
) -> impl FnMut(InferenceResponse) -> Result<InferenceFeedback, Infallible> + 'a {
    move |resp| match resp {
        InferenceResponse::InferredToken(t) => {
            let mut reverse_buf = buf.clone();
            reverse_buf.push_str(t.as_str());
            if stop_sequence.as_str().eq(reverse_buf.as_str()) {
                buf.clear();
                return Ok(InferenceFeedback::Halt);
            } else if stop_sequence.as_str().starts_with(reverse_buf.as_str()) {
                buf.push_str(t.as_str());
                return Ok(InferenceFeedback::Continue);
            }

            if buf.is_empty() {
                print_token(t)
            } else {
                print_token(reverse_buf)
            }
        }
        InferenceResponse::EotToken => Ok(InferenceFeedback::Halt),
        _ => Ok(InferenceFeedback::Continue),
    }
}

fn print_token(t: String) -> Result<InferenceFeedback, Infallible> {
    print!("{t}");
    std::io::stdout().flush().unwrap();

    Ok(InferenceFeedback::Continue)
}
