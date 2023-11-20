use candle_core::Device;
use kalosm_language_model::Embedder;
use kalosm_learning::{
    Class, Classifier, ClassifierConfig, TextClassifier, TextClassifierDatasetBuilder,
};
use rbert::{Bert, BertSpace};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Class)]
enum MyClass {
    Person,
    Thing,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut bert = Bert::builder().build()?;

    let dev = Device::cuda_if_available(0)?;
    let person_questions = vec![
        "What is the author's name?",
        "What is the author's age?",
        "Who is the queen of England?",
        "Who is the president of the United States?",
        "Who is the president of France?",
        "Tell me about the CEO of Apple.",
        "Who is the CEO of Google?",
        "Who is the CEO of Microsoft?",
        "What person invented the light bulb?",
        "What person invented the telephone?",
        "What is the name of the person who invented the light bulb?",
        "Who wrote the book 'The Lord of the Rings'?",
        "Who wrote the book 'The Hobbit'?",
        "How old is the author of the book 'The Lord of the Rings'?",
        "How old is the author of the book 'The Hobbit'?",
        "Who is the best soccer player in the world?",
        "Who is the best basketball player in the world?",
        "Who is the best tennis player in the world?",
        "Who is the best soccer player in the world right now?",
        "Who is the leader of the United States?",
        "Who is the leader of France?",
        "What is the name of the leader of the United States?",
        "What is the name of the leader of France?",
    ];
    let thing_sentences = vec![
        "What is the capital of France?",
        "What is the capital of England?",
        "What is the name of the biggest city in the world?",
        "What tool do you use to cut a tree?",
        "What tool do you use to cut a piece of paper?",
        "What is a good book to read?",
        "What is a good movie to watch?",
        "What is a good song to listen to?",
        "What is the best tool to use to create a website?",
        "What is the best tool to use to create a mobile app?",
        "How long does it take to fly from Paris to New York?",
        "How do you make a cake?",
        "How do you make a pizza?",
        "How can you make a website?",
        "What is the best way to learn a new language?",
        "What is the best way to learn a new programming language?",
        "What is a framework?",
        "What is a library?",
        "What is a good way to learn a new language?",
        "What is a good way to learn a new programming language?",
        "What is the city with the most people in the world?",
        "What is the most spoken language in the world?",
        "What is the most spoken language in the United States?",
    ];

    let mut dataset = TextClassifierDatasetBuilder::<MyClass, _, _>::new(&mut bert);

    for question in &person_questions {
        dataset.add(question, MyClass::Person).await?;
    }

    for sentence in &thing_sentences {
        dataset.add(sentence, MyClass::Thing).await?;
    }

    let dataset = dataset.build(&dev)?;

    let mut classifier;
    let layers = vec![10, 20, 10];

    loop {
        classifier = TextClassifier::<MyClass, BertSpace>::new(Classifier::new(
            &dev,
            ClassifierConfig::new(384).layers_dims(layers.clone()),
        )?);
        if let Err(error) = classifier.train(&dataset, &dev, 100) {
            println!("Error: {:?}", error);
        } else {
            break;
        }
        println!("Retrying...");
    }

    let config = classifier.config();
    classifier.save("classifier.safetensors")?;
    let mut classifier = Classifier::<MyClass>::load("classifier.safetensors", &dev, config)?;

    let tests = [
        "Who is the president of Russia?",
        "What is the capital of Russia?",
        "Who invented the TV?",
        "What is the best way to learn a how to ride a bike?",
    ];

    for test in &tests {
        let input = bert.embed(test).await?.to_vec();
        let class = classifier.run(&input)?;
        println!();
        println!("{test}");
        println!("{:?} {:?}", &input[..5], class);
    }

    Ok(())
}
