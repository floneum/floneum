use kalosm::{language::*, BertDistance, TestCases};
use kalosm_language::search::Hypothetical;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let pairs = [
        ("Floneum is a user-friendly editor for visual AI workflows. Unlike existing tools that may have a high barrier to entry or allow limited control, Floneum provides a solution that is both easy to use and allows for greater customization.", "What is Floneum?"),
        ("For instance, while the chat GPT interface provides a straightforward entry point, it quickly becomes challenging to create structured workflows. Imagine wanting to search through files to find specific ones, such as all .txt files related to travel, and then upload them. With Floneum, you can achieve this seamlessly within a structured workflow, eliminating the need for manual interaction with external tools.", "What are the tradeoffs of using chat GPT?"),
        ("On the other end of the spectrum, tools like Langchain offer extensive workflow customization but come with more system requirements and potential security concerns. Langchain requires users to install tools like Python and CUDA, making it less accessible to non-developers. In addition to this, building workflows in Python code can be impractical for individuals without programming expertise. Finally, plugins in Langchain are not sandboxed, which can expose users to malware or security risks when incorporating third-party libraries.", "What are the tradeoffs of using Langchain?"),
        ("Floneum is a single executable that runs models locally, eliminating the need for complex installations. The heart of Floneum is its graph-based editor, designed to enable users without programming knowledge to build and manage their AI workflows seamlessly.", "What is Floneum?"),
        ("Floneum is designed to support an expanding ecosystem of plugins. In the future, additional plugins will be added to enhance its functionality further. Furthermore, if the built-in plugins don't precisely fit your application, Floneum allows you to extend its capabilities with plugins that are fully sandboxed within their own environment. Through the utilization of a WebAssembly (WASM) compiler, plugins can only access resources within their designated sandbox. This ensures that you can trust Floneum to prevent any malicious activity from impacting your computer.", "What are Floneum plugins?"),
        ("Embeddings are a way to understand the meaning of text. They provide a representation of the meaning of the words used. It lets us focus on the meaning of the text instead of the specific wording of the text.", "What is an embedding?")
    ];

    {
        let mut llm = Phi::v2().unwrap();
    
        let hypothetical = Hypothetical::new(&mut llm).with_chunking(
            kalosm_language::search::ChunkStrategy::Paragraph {
                paragraph_count: 1,
                overlap: 0,
            },
        );
    
    
        let mut phi_test_cases = TestCases::new();
    
        for (text, expected) in pairs.iter() {
            let actual = &hypothetical.generate_question(text).await.unwrap()[0];
    
            phi_test_cases.push_case(expected.to_string(), actual.clone());
        }
    
        let mut bert_distance = BertDistance::default();
        let phi_distance = phi_test_cases.evaluate(&mut bert_distance).await.normalized();
        println!("{}", phi_distance);
    }

    {
        let mut llm = Llama::new_chat();
    
        let hypothetical = Hypothetical::new(&mut llm).with_chunking(
            kalosm_language::search::ChunkStrategy::Paragraph {
                paragraph_count: 1,
                overlap: 0,
            },
        );
    
        let mut llama_test_cases = TestCases::new();
    
        for (text, expected) in pairs.iter() {
            let actual = &hypothetical.generate_question(text).await.unwrap()[0];
    
            llama_test_cases.push_case(expected.to_string(), actual.clone());
        }
    
        let mut bert_distance = BertDistance::default();
        let phi_distance = llama_test_cases.evaluate(&mut bert_distance).await.normalized();
        println!("{}", phi_distance);
    }
}
