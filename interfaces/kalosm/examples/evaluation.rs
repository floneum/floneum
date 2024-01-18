use kalosm::{language::*, BertDistance, TestCases};
use kalosm_language::search::Hypothetical;
use rand::random;

const TEST_PAIRS: &[(&str, &str)]= &[
    ("Cryptocurrencies like Bitcoin operate on a decentralized network using blockchain technology. This decentralized nature eliminates the need for a central authority, providing users with increased security and censorship resistance.", "How does blockchain contribute to the security of cryptocurrencies?"),
    ("Machine learning models require training on labeled datasets to make predictions. Supervised learning, a common approach, involves providing the model with input-output pairs to learn the underlying patterns and relationships.", "What is supervised learning in machine learning?"),
    ("The Internet of Things or IoT connects everyday devices to the internet, enabling them to send and receive data. This connectivity enhances automation and allows for more efficient monitoring and control of various systems.", "What is the purpose of the Internet of Things?"),
    ("Cybersecurity measures, such as firewalls and encryption, play a crucial role in protecting computer systems from unauthorized access and data breaches. These measures help ensure the confidentiality and integrity of sensitive information.", "How do firewalls contribute to cybersecurity?"),
    ("WebAssembly or WASM is a binary instruction format designed for efficient execution on web browsers. It enables running code written in languages like C++ and Rust directly in the browser, expanding the possibilities of web applications.", "What is the purpose of WebAssembly?"),
    ("Quantum computing leverages the principles of quantum mechanics to perform computations at speeds unattainable by classical computers. This emerging technology holds the potential to solve complex problems, such as factorizing large numbers, with unprecedented efficiency.", "What sets quantum computing apart from classical computing?"),
    ("Docker containers provide a lightweight and portable way to package and deploy applications, along with their dependencies. This approach streamlines the deployment process and ensures consistency across different environments.", "What is the purpose of Docker containers in application deployment?"),
    ("Neural networks, inspired by the human brain, form the backbone of deep learning. These interconnected layers of nodes learn to recognize patterns and make predictions, enabling tasks like image recognition and natural language processing.", "What is the role of neural networks in deep learning?"),
    ("Floneum is a user-friendly editor for visual AI workflows. Unlike existing tools that may have a high barrier to entry or allow limited control, Floneum provides a solution that is both easy to use and allows for greater customization.", "What is Floneum?"),
    ("For instance, while the chat GPT interface provides a straightforward entry point, it quickly becomes challenging to create structured workflows. Imagine wanting to search through files to find specific ones, such as all .txt files related to travel, and then upload them. With Floneum, you can achieve this seamlessly within a structured workflow, eliminating the need for manual interaction with external tools.", "What are the tradeoffs of using chat GPT?"),
    ("On the other end of the spectrum, tools like Langchain offer extensive workflow customization but come with more system requirements and potential security concerns. Langchain requires users to install tools like Python and CUDA, making it less accessible to non-developers. In addition to this, building workflows in Python code can be impractical for individuals without programming expertise. Finally, plugins in Langchain are not sandboxed, which can expose users to malware or security risks when incorporating third-party libraries.", "What are the tradeoffs of using Langchain?"),
    ("Floneum is a single executable that runs models locally, eliminating the need for complex installations. The heart of Floneum is its graph-based editor, designed to enable users without programming knowledge to build and manage their AI workflows seamlessly.", "What is Floneum?"),
    ("Embeddings are a way to understand the meaning of text. They provide a representation of the meaning of the words used. It lets us focus on the meaning of the text instead of the specific wording of the text.", "What is an embedding?"),
];

const TRAIN_PAIRS: &[(&str, &str)]= &[
    ("In the world of programming languages, Rust stands out for its focus on memory safety without sacrificing performance. Its ownership system ensures that memory is managed efficiently, preventing common pitfalls like null pointer dereferencing.", "What is Rust known for?"),
    ("While traditional databases rely on a fixed schema, NoSQL databases like MongoDB offer a flexible structure, allowing you to store and retrieve data in a more dynamic way. This flexibility is particularly beneficial for applications with evolving data requirements.", "How does MongoDB differ from traditional databases?"),
    ("Floneum is designed to support an expanding ecosystem of plugins. In the future, additional plugins will be added to enhance its functionality further. Furthermore, if the built-in plugins don't precisely fit your application, Floneum allows you to extend its capabilities with plugins that are fully sandboxed within their own environment. Through the utilization of a WebAssembly (WASM) compiler, plugins can only access resources within their designated sandbox. This ensures that you can trust Floneum to prevent any malicious activity from impacting your computer.", "What are Floneum plugins?"),
    ("The agile software development methodology emphasizes iterative and collaborative approaches to project management. This flexible framework allows teams to adapt to changing requirements and deliver incremental improvements.", "What are the key principles of agile software development?"),
    ("The concept of virtualization involves creating virtual instances of resources like servers or networks. Virtualization enables better resource utilization, scalability, and isolation in computing environments.", "What is the purpose of virtualization in computer science?"),
    ("Open-source software, like the Linux operating system, is freely accessible and allows users to view, modify, and distribute its source code. This collaborative approach fosters innovation and community-driven development.", "What distinguishes open-source software from proprietary software?"),
    ("A content delivery network or a CDN optimizes the distribution of web content by strategically placing servers worldwide. This reduces latency, accelerates content delivery, and enhances the overall user experience.", "What role does a content delivery network play in web performance?"),
    ("Biometric authentication methods, such as fingerprint recognition and facial recognition, use unique biological characteristics for identity verification. This enhances security by providing a more robust and personalized authentication process.", "How do biometric authentication methods contribute to security?"),
    ("The concept of DevOps integrates software development and IT operations to improve collaboration and productivity. Automation tools play a crucial role in achieving continuous integration and continuous delivery in DevOps practices.", "What is the relationship between DevOps, continuous integration and continuous delivery?"),
    ("Natural Language Processing or NLP involves the interaction between computers and human language. Applications of NLP include language translation, sentiment analysis, and chatbot development.", "What are some applications of Natural Language Processing?"),
    ("The concept of edge computing involves processing data closer to the source rather than relying on a centralized cloud infrastructure. This approach reduces latency and enhances real-time processing in distributed systems.", "What is the significance of edge computing in data processing?"),
    ("APIs facilitate communication between different software systems. They define the methods and data formats applications can use to request and exchange information.", "What is the purpose of APIs in software development?"),
    ("Blockchain technology, beyond cryptocurrencies, is being explored for applications like smart contracts. Smart contracts are self-executing contracts with the terms of the agreement directly written into code.", "How is blockchain technology utilized in the concept of smart contracts?")
];

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let mut llm = Llama::new_chat();

    let mut instances = Vec::new();

    for _ in 0..10 {
        instances.push(ExamplesInstance::new(&mut llm).await);
    }

    loop {
        for instance in &mut instances {
            instance.mutate(&mut llm).await;
        }

        println!("current temperature: {}", instances[0].temperature);

        for instance in &instances {
            println!("(score = {}) {:?}", instance.current_evaluation, instance.current_examples);
        }
    }
}

struct ExamplesInstance {
    current_examples: Vec<(&'static str, &'static str)>,
    unused_examples: Vec<(&'static str, &'static str)>,
    current_evaluation: f64,
    temperature: f64,
}

impl ExamplesInstance {
    async fn new(llm: &mut Llama) -> Self {
        let current_examples = TRAIN_PAIRS[..3].to_vec();
        let unused_examples = TRAIN_PAIRS[3..].to_vec();
        let current_evaluation = evaluate(&current_examples, llm).await;

        Self {
            current_examples,
            unused_examples,
            current_evaluation,
            temperature: 0.6,
        }
    }

    async fn mutate(&mut self, llm: &mut Llama) {
        let action = if self.current_examples.is_empty() {
            2
        } else if self.unused_examples.is_empty() {
            random::<usize>() % 2
        } else {
            random::<usize>() % 3
        };

        let accept_regardless = random::<f64>() < self.temperature;

        if accept_regardless {
            println!("temperature: {}", self.temperature);
            println!("accepting regardless of score");
        }

        let mut mutated_examples = self.current_examples.clone();

        match action {
            // remove example
            0 => {
                let index = random::<usize>() % mutated_examples.len();
                let removed = mutated_examples.remove(index);

                let new_evaluation = evaluate(&mutated_examples, llm).await;

                if accept_regardless || new_evaluation > self.current_evaluation {
                    self.current_evaluation = new_evaluation;
                    self.current_examples = mutated_examples;
                    self.unused_examples.push(removed);
                }
            }
            // swap examples
            1 => {
                let index1 = random::<usize>() % mutated_examples.len();
                let index2 = random::<usize>() % mutated_examples.len();

                mutated_examples.swap(index1, index2);

                let new_evaluation = evaluate(&mutated_examples, llm).await;

                if accept_regardless || new_evaluation > self.current_evaluation {
                    self.current_evaluation = new_evaluation;
                    self.current_examples = mutated_examples;
                }
            }
            // add example
            _ => {
                let index = random::<usize>() % self.unused_examples.len();
                let added = self.unused_examples[index];
                mutated_examples.push(added);

                let new_evaluation = evaluate(&mutated_examples, llm).await;

                if accept_regardless || new_evaluation > self.current_evaluation {
                    self.current_evaluation = new_evaluation;
                    self.current_examples = mutated_examples;
                    self.unused_examples.remove(index);
                }
            }
        }

        self.temperature *= 0.9;
    }
}

async fn evaluate(examples: &[(&str, &str)], llm: &mut Llama) -> f64 {
    let examples_tokens: usize = examples
        .iter()
        .filter_map(|(text, _)| llm.tokenizer().encode(text).ok().map(|x| x.len()))
        .sum();

    let hypothetical = Hypothetical::builder(llm)
        .with_chunking(kalosm_language::search::ChunkStrategy::Paragraph {
            paragraph_count: 1,
            overlap: 0,
        })
        .with_examples(examples.iter().copied())
        .build()
        .unwrap();

    let mut llama_test_cases = TestCases::new();

    for (text, expected) in TEST_PAIRS {
        let actual = &hypothetical.generate_question(text, llm).await.unwrap()[0];

        llama_test_cases.push_case(expected.to_string(), actual.clone());
    }

    let mut bert_distance = BertDistance::default();
    let llama_distance = llama_test_cases
        .evaluate(&mut bert_distance)
        .await
        .normalized();

    println!("evaluating examples {:?}", examples);
    println!("{}", llama_distance);

    llama_distance.mean_score() - examples_tokens as f64 * 0.0001
}
