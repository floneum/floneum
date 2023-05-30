use crate::{infer::download, GptNeoXType, ModelType};

pub fn run_embed() {
    let query = "My favourite animal is the dog";
    let comparands = vec![
        "My favourite animal is the dog".to_string(),
        "I have just adopted a cute dog".to_string(),
        "My favourite animal is the cat".to_string(),
    ];

    // Load model
    let model = download(ModelType::GptNeoX(GptNeoXType::TinyPythia));
    let inference_parameters = llm::InferenceParameters::default();

    // Generate embeddings for query and comparands
    let query_embeddings = get_embeddings(model.as_ref(), &inference_parameters, query);
    let comparand_embeddings: Vec<(String, Vec<f32>)> = comparands
        .iter()
        .map(|text| {
            (
                text.clone(),
                get_embeddings(model.as_ref(), &inference_parameters, text),
            )
        })
        .collect();

    // Print embeddings
    fn print_embeddings(text: &str, embeddings: &[f32]) {
        println!("{text}");
        println!("  Embeddings length: {}", embeddings.len());
        println!("  Embeddings first 10: {:.02?}", embeddings.get(0..10));
    }

    print_embeddings(query, &query_embeddings);
    println!("---");
    for (text, embeddings) in &comparand_embeddings {
        print_embeddings(text, embeddings);
    }

    // Calculate the cosine similarity between the query and each comparand, and sort by similarity
    let mut similarities: Vec<(&str, f32)> = comparand_embeddings
        .iter()
        .map(|(text, embeddings)| {
            (
                text.as_str(),
                cosine_similarity(&query_embeddings, &embeddings),
            )
        })
        .collect();
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Print similarities
    println!("---");
    println!("Similarities:");
    for (text, score) in similarities {
        println!("  {text}: {score}");
    }
}

fn get_embeddings(
    model: &dyn llm::Model,
    inference_parameters: &llm::InferenceParameters,
    query: &str,
) -> Vec<f32> {
    let mut session = model.start_session(Default::default());
    let mut output_request = llm::OutputRequest {
        all_logits: None,
        embeddings: Some(Vec::new()),
    };
    let _ = session.feed_prompt(
        model,
        inference_parameters,
        query,
        &mut output_request,
        |_| Ok::<_, std::convert::Infallible>(llm::InferenceFeedback::Halt),
    );
    output_request.embeddings.unwrap()
}

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot_product = dot(&v1, &v2);
    let magnitude1 = magnitude(&v1);
    let magnitude2 = magnitude(&v2);

    dot_product / (magnitude1 * magnitude2)
}

fn dot(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(&x, &y)| x * y).sum()
}

fn magnitude(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}
