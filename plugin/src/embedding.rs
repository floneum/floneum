use std::panic::catch_unwind;

use crate::{ exports::plugins::main::definitions::Embedding};


pub fn get_embeddings(
    model: &dyn llm::Model,
    inference_parameters: &llm::InferenceParameters,
    embed: &str,
) -> Embedding{
    let mut session = model.start_session(Default::default());
    let mut output_request = llm::OutputRequest {
        all_logits: None,
        embeddings: Some(Vec::new()),
    };
    let _ = session.feed_prompt(
        model,
        inference_parameters,
        embed,
        &mut output_request,
        |_| Ok::<_, std::convert::Infallible>(llm::InferenceFeedback::Halt),
    );
    Embedding{
        vector:
        output_request.embeddings.unwrap()}
}
