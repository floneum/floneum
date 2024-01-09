use kalosm::language::*;
use kalosm_language::search::{Chunker, Hypothetical};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let documents = vec![
        Document::from_parts("Floneum Blog", "Floneum is a user-friendly editor for visual AI workflows. Unlike existing tools that may have a high barrier to entry or allow limited control, Floneum provides a solution that is both easy to use and allows for greater customization.

        For instance, while the chat GPT interface provides a straightforward entry point, it quickly becomes challenging to create structured workflows. Imagine wanting to search through files to find specific ones, such as all .txt files related to travel, and then upload them. With Floneum, you can achieve this seamlessly within a structured workflow, eliminating the need for manual interaction with external tools.
        
        On the other end of the spectrum, tools like Langchain offer extensive workflow customization but come with more system requirements and potential security concerns. Langchain requires users to install tools like Python and CUDA, making it less accessible to non-developers. In addition to this, building workflows in Python code can be impractical for individuals without programming expertise. Finally, plugins in Langchain are not sandboxed, which can expose users to malware or security risks when incorporating third-party libraries.
        
        Floneum is a single executable that runs models locally, eliminating the need for complex installations. The heart of Floneum is its graph-based editor, designed to enable users without programming knowledge to build and manage their AI workflows seamlessly.")
    ];

    let mut llm = Phi::v2().unwrap();

    let hypothetical = Hypothetical::new(&mut llm).with_chunking(
        kalosm_language::search::ChunkStrategy::Paragraph {
            paragraph_count: 1,
            overlap: 0,
        },
    );

    let mut embedder = Bert::default();
    let chunked = hypothetical
        .chunk_batch(&documents, &mut embedder)
        .await
        .unwrap();
    println!("chunked: {:?}", chunked);
}
