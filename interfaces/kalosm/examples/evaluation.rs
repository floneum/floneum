use kalosm::{language::*, BertDistance, TestCases};
use kalosm_language::search::Hypothetical;

const TEST_PAIRS :&[(&str, &str)]= &[
        ("Floneum is a user-friendly editor for visual AI workflows. Unlike existing tools that may have a high barrier to entry or allow limited control, Floneum provides a solution that is both easy to use and allows for greater customization.", "What is Floneum?"),
        ("For instance, while the chat GPT interface provides a straightforward entry point, it quickly becomes challenging to create structured workflows. Imagine wanting to search through files to find specific ones, such as all .txt files related to travel, and then upload them. With Floneum, you can achieve this seamlessly within a structured workflow, eliminating the need for manual interaction with external tools.", "What are the tradeoffs of using chat GPT?"),
        ("On the other end of the spectrum, tools like Langchain offer extensive workflow customization but come with more system requirements and potential security concerns. Langchain requires users to install tools like Python and CUDA, making it less accessible to non-developers. In addition to this, building workflows in Python code can be impractical for individuals without programming expertise. Finally, plugins in Langchain are not sandboxed, which can expose users to malware or security risks when incorporating third-party libraries.", "What are the tradeoffs of using Langchain?"),
        ("Floneum is a single executable that runs models locally, eliminating the need for complex installations. The heart of Floneum is its graph-based editor, designed to enable users without programming knowledge to build and manage their AI workflows seamlessly.", "What is Floneum?"),
        ("Floneum is designed to support an expanding ecosystem of plugins. In the future, additional plugins will be added to enhance its functionality further. Furthermore, if the built-in plugins don't precisely fit your application, Floneum allows you to extend its capabilities with plugins that are fully sandboxed within their own environment. Through the utilization of a WebAssembly (WASM) compiler, plugins can only access resources within their designated sandbox. This ensures that you can trust Floneum to prevent any malicious activity from impacting your computer.", "What are Floneum plugins?"),
        ("Embeddings are a way to understand the meaning of text. They provide a representation of the meaning of the words used. It lets us focus on the meaning of the text instead of the specific wording of the text.", "What is an embedding?"),
        ("In the world of programming languages, Rust stands out for its focus on memory safety without sacrificing performance. Its ownership system ensures that memory is managed efficiently, preventing common pitfalls like null pointer dereferencing.", "What is Rust known for?"),
        ("While traditional databases rely on a fixed schema, NoSQL databases like MongoDB offer a flexible structure, allowing you to store and retrieve data in a more dynamic way. This flexibility is particularly beneficial for applications with evolving data requirements.", "How does MongoDB differ from traditional databases?"),
        ("Cryptocurrencies like Bitcoin operate on a decentralized network using blockchain technology. This decentralized nature eliminates the need for a central authority, providing users with increased security and censorship resistance.", "How does blockchain contribute to the security of cryptocurrencies?"),
        ("Machine learning models require training on labeled datasets to make predictions. Supervised learning, a common approach, involves providing the model with input-output pairs to learn the underlying patterns and relationships.", "What is supervised learning in machine learning?"),
        ("The Internet of Things (IoT) connects everyday devices to the internet, enabling them to send and receive data. This connectivity enhances automation and allows for more efficient monitoring and control of various systems.", "What is the purpose of the Internet of Things (IoT)?"),
        ("Cybersecurity measures, such as firewalls and encryption, play a crucial role in protecting computer systems from unauthorized access and data breaches. These measures help ensure the confidentiality and integrity of sensitive information.", "How do firewalls contribute to cybersecurity?"),
        ("WebAssembly (WASM) is a binary instruction format designed for efficient execution on web browsers. It enables running code written in languages like C++ and Rust directly in the browser, expanding the possibilities of web applications.", "What is the purpose of WebAssembly (WASM)?"),
        ("Quantum computing leverages the principles of quantum mechanics to perform computations at speeds unattainable by classical computers. This emerging technology holds the potential to solve complex problems, such as factorizing large numbers, with unprecedented efficiency.", "What sets quantum computing apart from classical computing?"),
        ("Docker containers provide a lightweight and portable way to package and deploy applications, along with their dependencies. This approach streamlines the deployment process and ensures consistency across different environments.", "What is the purpose of Docker containers in application deployment?"),
        ("Neural networks, inspired by the human brain, form the backbone of deep learning. These interconnected layers of nodes learn to recognize patterns and make predictions, enabling tasks like image recognition and natural language processing.", "What is the role of neural networks in deep learning?"),
        ("The agile software development methodology emphasizes iterative and collaborative approaches to project management. This flexible framework allows teams to adapt to changing requirements and deliver incremental improvements.", "What are the key principles of agile software development?"),
        ("The concept of virtualization involves creating virtual instances of resources like servers or networks. Virtualization enables better resource utilization, scalability, and isolation in computing environments.", "What is the purpose of virtualization in computer science?"),
        ("Open-source software, like the Linux operating system, is freely accessible and allows users to view, modify, and distribute its source code. This collaborative approach fosters innovation and community-driven development.", "What distinguishes open-source software from proprietary software?"),
        ("A content delivery network (CDN) optimizes the distribution of web content by strategically placing servers worldwide. This reduces latency, accelerates content delivery, and enhances the overall user experience.", "What role does a content delivery network (CDN) play in web performance?"),
        ("Biometric authentication methods, such as fingerprint recognition and facial recognition, use unique biological characteristics for identity verification. This enhances security by providing a more robust and personalized authentication process.", "How do biometric authentication methods contribute to security?"),
        ("The concept of DevOps integrates software development and IT operations to improve collaboration and productivity. Automation tools play a crucial role in achieving continuous integration and continuous delivery (CI/CD) in DevOps practices.", "What is the relationship between DevOps and continuous integration/continuous delivery (CI/CD)?"),
        ("Natural Language Processing (NLP) involves the interaction between computers and human language. Applications of NLP include language translation, sentiment analysis, and chatbot development.", "What are some applications of Natural Language Processing (NLP)?"),
        ("The concept of edge computing involves processing data closer to the source rather than relying on a centralized cloud infrastructure. This approach reduces latency and enhances real-time processing in distributed systems.", "What is the significance of edge computing in data processing?"),
        ("APIs (Application Programming Interfaces) facilitate communication between different software systems. They define the methods and data formats applications can use to request and exchange information.", "What is the purpose of APIs in software development?"),
        ("Blockchain technology, beyond cryptocurrencies, is being explored for applications like smart contracts. Smart contracts are self-executing contracts with the terms of the agreement directly written into code.", "How is blockchain technology utilized in the concept of smart contracts?")
    ];

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    {
        let mut llm = Phi::v2().unwrap();

        let hypothetical = Hypothetical::new(&mut llm).with_chunking(
            kalosm_language::search::ChunkStrategy::Paragraph {
                paragraph_count: 1,
                overlap: 0,
            },
        );

        let mut phi_test_cases = TestCases::new();

        for (text, expected) in TEST_PAIRS.iter() {
            let actual = &hypothetical.generate_question(text).await.unwrap()[0];

            phi_test_cases.push_case(expected.to_string(), actual.clone());
        }

        let mut bert_distance = BertDistance::default();
        let phi_distance = phi_test_cases
            .evaluate(&mut bert_distance)
            .await
            .normalized();
        println!("phi");
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

        for (text, expected) in TEST_PAIRS.iter() {
            let actual = &hypothetical.generate_question(text).await.unwrap()[0];

            llama_test_cases.push_case(expected.to_string(), actual.clone());
        }

        let mut bert_distance = BertDistance::default();
        let phi_distance = llama_test_cases
            .evaluate(&mut bert_distance)
            .await
            .normalized();
        println!("open chat");
        println!("{}", phi_distance);
    }

    {
        let mut llm = Llama::builder().with_source(LlamaSource::solar_10_7b_instruct()).build().unwrap();

        let hypothetical = Hypothetical::new(&mut llm).with_chunking(
            kalosm_language::search::ChunkStrategy::Paragraph {
                paragraph_count: 1,
                overlap: 0,
            },
        );

        let mut llama_test_cases = TestCases::new();

        for (text, expected) in TEST_PAIRS.iter() {
            let actual = &hypothetical.generate_question(text).await.unwrap()[0];

            llama_test_cases.push_case(expected.to_string(), actual.clone());
        }

        let mut bert_distance = BertDistance::default();
        let phi_distance = llama_test_cases
            .evaluate(&mut bert_distance)
            .await
            .normalized();
        println!("solar");
        println!("{}", phi_distance);
    }

    {
        let mut llm = Llama::builder().with_source(LlamaSource::mistral_7b()).build().unwrap();

        let hypothetical = Hypothetical::new(&mut llm).with_chunking(
            kalosm_language::search::ChunkStrategy::Paragraph {
                paragraph_count: 1,
                overlap: 0,
            },
        );

        let mut llama_test_cases = TestCases::new();

        for (text, expected) in TEST_PAIRS.iter() {
            let actual = &hypothetical.generate_question(text).await.unwrap()[0];

            llama_test_cases.push_case(expected.to_string(), actual.clone());
        }

        let mut bert_distance = BertDistance::default();
        let phi_distance = llama_test_cases
            .evaluate(&mut bert_distance)
            .await
            .normalized();
        println!("mistral");
        println!("{}", phi_distance);
    }
}
