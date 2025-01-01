use std::io::Write;

use kalosm_llama::prelude::*;

#[tokio::main]
async fn main() {
    let model = Llama::builder()
        .with_source(LlamaSource::qwen_2_5_0_5b_instruct())
        .build()
        .await
        .unwrap();

    let mut session = model.new_session().unwrap();
    let on_token = |token: String| {
        print!("{token}");
        std::io::stdout().flush().unwrap();
        Ok(())
    };

    let sampler = GenerationParameters::default().sampler();

    let prompt = "<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
";
    model
        .stream_text_with_callback(&mut session, prompt, sampler, on_token)
        .await
        .unwrap();
}
