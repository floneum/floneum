use crate::{GptNeoXType, LlamaType, ModelId, ModelType, MptType};
use futures_util::stream::StreamExt;
use llm::{
    InferenceFeedback, InferenceRequest, InferenceResponse, LoadProgress, Model, ModelArchitecture,
};
use spinoff::{spinners::Dots2, Spinner};
use std::{convert::Infallible, error::Error, path::PathBuf, time::Instant};
use tokio::{fs::File, io::AsyncWriteExt};

pub fn download(model_type: ModelType) -> Box<dyn Model> {
    // https://www.reddit.com/r/LocalLLaMA/wiki/models/
    let url = match model_type {
        ModelType::Llama(LlamaType::Vicuna) => {
            "https://huggingface.co/CRD716/ggml-vicuna-1.1-quantized/resolve/main/ggml-vicuna-13B-1.1-q4_0.bin"
        }
        ModelType::Llama(LlamaType::Guanaco) => {
            "https://huggingface.co/TheBloke/guanaco-7B-GGML/resolve/main/guanaco-7B.ggmlv3.q4_0.bin"
        }
        ModelType::Llama(LlamaType::Wizardlm) => {
            "https://huggingface.co/TehVenom/WizardLM-13B-Uncensored-Q5_1-GGML/blob/main/WizardML-Unc-13b-Q5_1.bin"
        }
        ModelType::GptNeoX(GptNeoXType::Stablelm) => {
            "https://huggingface.co/cakewalk/ggml-q4_0-stablelm-tuned-alpha-7b/resolve/main/ggml-model-stablelm-tuned-alpha-7b-q4_0.bin"
        }
        ModelType::GptNeoX(GptNeoXType::DollySevenB) => {
            "https://huggingface.co/mverrilli/dolly-v2-7b-ggml/resolve/main/ggml-model-f16.bin"
        }
        ModelType::GptNeoX(GptNeoXType::TinyPythia) => {
            "https://huggingface.co/rustformers/pythia-ggml/resolve/main/pythia-70m-q4_0.bin"
        }
        ModelType::GptNeoX(GptNeoXType::LargePythia) => {
            "https://huggingface.co/rustformers/pythia-ggml/resolve/main/pythia-2.8b-q4_0.bin"
        }
        ModelType::Mpt(MptType::Chat) => {
            "https://huggingface.co/rustformers/mpt-7b-ggml/resolve/main/mpt-7b-chat-q4_0.bin"
        }
        ModelType::Mpt(MptType::Story) => {
            "https://huggingface.co/rustformers/mpt-7b-ggml/resolve/main/mpt-7b-storywriter-q4_0.bin"
        }
        ModelType::Mpt(MptType::Instruct) => {
            "https://huggingface.co/rustformers/mpt-7b-ggml/resolve/main/mpt-7b-instruct-q4_0.bin"
        }
        ModelType::Mpt(MptType::Base) => {
            "https://huggingface.co/rustformers/mpt-7b-ggml/resolve/main/mpt-7b-q4_0.bin"
        }
    };
    let archetecture = match model_type {
        ModelType::Llama(_) => ModelArchitecture::Llama,
        ModelType::GptNeoX(_) => ModelArchitecture::GptNeoX,
        ModelType::Mpt(_) => ModelArchitecture::Mpt,
    };

    let rt = tokio::runtime::Runtime::new().unwrap();
    let path = rt.block_on(download_model(url)).unwrap();

    let sp = Some(Spinner::new(Dots2, "Loading model...", None));

    let now = Instant::now();
    let prev_load_time = now;

    llm::load_dynamic(
        archetecture,
        &path,
        llm::VocabularySource::Model,
        Default::default(),
        load_progress_callback(sp, now, prev_load_time),
    )
    .unwrap_or_else(|err| panic!("Failed to load model from {path:?}: {err}"))
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

#[derive(Default)]
pub struct InferenceSessions {
    sessions: slab::Slab<(Box<dyn Model>, llm::InferenceSession)>,
}

impl InferenceSessions {
    pub fn create(&mut self, ty: ModelType) -> ModelId {
        let model = download(ty);
        let session = model.start_session(Default::default());
        ModelId {
            id: self.sessions.insert((model, session)) as u32,
        }
    }

    pub fn remove(&mut self, id: ModelId) {
        self.sessions.remove(id.id as usize);
    }

    pub fn infer(
        &mut self,
        id: ModelId,
        prompt: String,
        max_tokens: Option<u32>,
        stop_on: Option<String>,
    ) -> String {
        let (model, session) = self.sessions.get_mut(id.id as usize).unwrap();

        let mut rng = rand::thread_rng();
        let mut buf = String::new();
        let request = InferenceRequest {
            prompt: (&prompt).into(),
            parameters: &Default::default(),
            play_back_previous_tokens: false,
            maximum_token_count: max_tokens.map(|x| x as usize),
        };

        session
            .infer(
                model.as_ref(),
                &mut rng,
                &request,
                &mut Default::default(),
                inference_callback(stop_on, &mut buf),
            )
            .unwrap_or_else(|e| panic!("{e}"));

        buf
    }
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

fn inference_callback(
    stop_sequence: Option<String>,
    buf: &mut String,
) -> impl FnMut(InferenceResponse) -> Result<InferenceFeedback, Infallible> + '_ {
    move |resp| match resp {
        InferenceResponse::InferredToken(t) => {
            let mut reverse_buf = buf.clone();
            reverse_buf.push_str(t.as_str());
            if let Some(stop_sequence) = &stop_sequence {
                if stop_sequence.as_str().eq(reverse_buf.as_str()) {
                    return Ok(InferenceFeedback::Halt);
                }
            }
            buf.push_str(t.as_str());

            Ok(InferenceFeedback::Continue)
        }
        InferenceResponse::EotToken => Ok(InferenceFeedback::Halt),
        _ => Ok(InferenceFeedback::Continue),
    }
}
