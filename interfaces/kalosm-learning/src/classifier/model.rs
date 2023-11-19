use candle_core::{safetensors::Load, DType, Device, Result, Tensor, Var, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap};
use rand::Rng;

/// A class that a [`Classifier`] can predict.
pub trait Class {
    /// The number of classes.
    const CLASSES: u32;

    /// Convert the class to a class index.
    fn to_class(&self) -> u32;
    /// Convert a class index to a class.
    fn from_class(class: u32) -> Self;
}

/// A dataset to train a [`Classifier`].
#[derive(Clone, Debug)]
pub struct ClassificationDataset {
    train_inputs: Tensor,
    train_classes: Tensor,
    test_inputs: Tensor,
    test_classes: Tensor,
}

/// A builder for [`ClassificationDataset`].
pub struct DatasetBuilder<D: candle_core::WithDType, C: Class> {
    input_size: usize,
    inputs: Vec<Vec<D>>,
    classes: Vec<C>,
}

impl<D: candle_core::WithDType, C: Class> DatasetBuilder<D, C> {
    /// Create a new dataset builder.
    pub fn new(input_size: usize) -> Self {
        Self {
            input_size,
            inputs: Vec::new(),
            classes: Vec::new(),
        }
    }

    /// Adds a pair of input and class to the dataset.
    pub fn add(&mut self, input: Vec<D>, class: C) {
        self.inputs.push(input);
        self.classes.push(class);
    }

    /// Builds the dataset.
    pub fn build(mut self, dev: &Device) -> Result<ClassificationDataset> {
        // split into train and test
        let mut rng = rand::thread_rng();
        let first_quarter = self.inputs.len() / 4;
        println!(
            "{} train/{} tests",
            self.inputs.len() - first_quarter,
            first_quarter
        );
        let mut input_test: Vec<D> = Vec::with_capacity(first_quarter);
        let mut class_test: Vec<u32> = Vec::with_capacity(first_quarter);
        let mut inputs: Vec<D> = Vec::with_capacity(self.inputs.len() - first_quarter);
        let mut classes: Vec<u32> = Vec::with_capacity(self.classes.len() - first_quarter);
        while !self.classes.is_empty() {
            let index = rng.gen_range(0..self.classes.len());
            let mut input = self.inputs.remove(index);
            let class = self.classes.remove(index);
            if input_test.len() <= first_quarter {
                input_test.append(&mut input);
                class_test.push(class.to_class());
            } else {
                inputs.append(&mut input);
                classes.push(class.to_class());
            }
        }

        // train
        let train_inputs_len = inputs.len() / self.input_size;
        let train_inputs = Tensor::from_vec(inputs, (train_inputs_len, self.input_size), dev)?;
        let train_classes_len = classes.len();
        debug_assert_eq!(train_inputs_len, train_classes_len);
        let train_classes = Tensor::from_vec(classes, train_classes_len, dev)?;

        // test
        let test_inputs_len = input_test.len() / self.input_size;
        let test_inputs = Tensor::from_vec(input_test, (test_inputs_len, self.input_size), dev)?;
        let test_classes_len = class_test.len();
        debug_assert_eq!(test_inputs_len, test_classes_len);
        let test_classes = Tensor::from_vec(class_test, test_classes_len, dev)?;

        Ok(ClassificationDataset {
            train_inputs,
            train_classes,
            test_inputs,
            test_classes,
        })
    }
}

/// A classifier.
pub struct Classifier<C: Class> {
    device: Device,
    varmap: VarMap,
    layers: Vec<Linear>,
    phantom: std::marker::PhantomData<C>,
}

impl<C: Class> Classifier<C> {
    /// Create a new classifier.
    pub fn new(dev: &Device, input_dim: usize, layers_dims: &[usize]) -> Result<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        Self::new_inner(dev.clone(), varmap, vs, input_dim, layers_dims)
    }

    fn new_inner(
        dev: Device,
        varmap: VarMap,
        vs: VarBuilder,
        input_dim: usize,
        layers_dims: &[usize],
    ) -> Result<Self> {
        let output_dim = C::CLASSES;
        if layers_dims.is_empty() {
            return Ok(Self {
                device: dev,
                varmap,
                layers: vec![candle_nn::linear(
                    input_dim,
                    output_dim as usize,
                    vs.pp("ln0"),
                )?],
                phantom: std::marker::PhantomData,
            });
        }
        let mut layers = Vec::with_capacity(layers_dims.len() + 1);
        layers.push(candle_nn::linear(
            input_dim,
            *layers_dims.first().unwrap(),
            vs.pp("ln0"),
        )?);
        for (i, (in_dim, out_dim)) in layers_dims
            .iter()
            .zip(layers_dims.iter().skip(1))
            .enumerate()
        {
            layers.push(candle_nn::linear(
                *in_dim,
                *out_dim,
                vs.pp(&format!("ln{}", i + 1)),
            )?);
        }
        layers.push(candle_nn::linear(
            *layers_dims.last().unwrap(),
            output_dim as usize,
            vs.pp(format!("ln{}", layers_dims.len() + 1)),
        )?);
        Ok(Self {
            device: dev,
            layers,
            varmap,
            phantom: std::marker::PhantomData,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
            xs = xs.relu()?;
        }
        Ok(xs)
    }

    /// Train the model on the given dataset.
    pub fn train(
        &mut self,
        m: &ClassificationDataset,
        dev: &Device,
        epochs: usize,
    ) -> anyhow::Result<()> {
        let train_results = m.train_classes.to_device(dev)?;
        let train_votes = m.train_inputs.to_device(dev)?;

        let mut sgd = candle_nn::AdamW::new_lr(self.varmap.all_vars(), 0.05)?;
        let test_votes = m.test_inputs.to_device(dev)?;
        let test_results = m.test_classes.to_device(dev)?;
        let mut final_accuracy: f32 = 0.0;
        for epoch in 1..epochs + 1 {
            let logits = self.forward(&train_votes)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_sm, &train_results)?;
            sgd.backward_step(&loss)?;

            let test_logits = self.forward(&test_votes)?;
            let sum_ok = test_logits
                .argmax(D::Minus1)?
                .eq(&test_results)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
            let test_accuracy: f32 = sum_ok / test_results.dims1()? as f32;
            final_accuracy = f32::from(100u8) * test_accuracy;
            println!(
                "Epoch: {epoch:3} Train loss: {:8.5} Test accuracy: {:5.2}%",
                loss.to_scalar::<f32>()?,
                final_accuracy
            );
        }
        if final_accuracy < 100.0 {
            Err(anyhow::Error::msg("The model is not trained well enough."))
        } else {
            Ok(())
        }
    }

    /// Save the model to a safetensors file at the given path.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.varmap.save(path)
    }

    /// Load the model from a safetensors file at the given path.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        dev: Device,
        input_dim: usize,
        layers_dims: &[usize],
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        {
            let safetensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }?;
            let tensors = safetensors.tensors();
            let mut tensor_data = varmap.data().lock().unwrap();
            for (name, value) in tensors {
                let tensor = value.load(&dev)?;
                tensor_data.insert(name.to_string(), Var::from_tensor(&tensor)?);
            }
        }
        Self::new_inner(dev, varmap, vs, input_dim, layers_dims)
    }

    /// Run the model on the given input.
    pub fn run(&mut self, input: &[f32]) -> Result<C> {
        let input = Tensor::from_vec(input.to_vec(), (1, input.len()), &self.device)?;
        let logits = self.forward(&input)?;
        let class = logits
            .flatten_to(1)?
            .argmax(D::Minus1)?
            .to_scalar::<u32>()?;
        Ok(C::from_class(class))
    }
}

#[tokio::test]
async fn simplified() -> anyhow::Result<()> {
    use kalosm_language_model::Embedder;
    use rbert::Bert;

    #[derive(Debug, Copy, Clone, PartialEq, Eq, kalosm_learning_macro::Class)]
    enum MyClass {
        Person,
        Thing,
    }

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
    let person_embeddings = bert.embed_batch(&person_questions).await?;
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
    let thing_embeddings = bert.embed_batch(&thing_sentences).await?;

    let input_size = person_embeddings[0].to_vec().len();

    let mut dataset = DatasetBuilder::<f32, MyClass>::new(384);

    for (person_embedding, thing_embedding) in person_embeddings.iter().zip(thing_embeddings.iter())
    {
        dataset.add(person_embedding.to_vec(), MyClass::Person);
        dataset.add(thing_embedding.to_vec(), MyClass::Thing);
    }

    let dataset = dataset.build(&dev)?;

    let mut classifier;
    let layers = [10, 20, 10];

    loop {
        classifier = Classifier::<MyClass>::new(&dev, input_size, &layers)?;
        let error = classifier.train(&dataset, &dev, 20).is_err();
        if !error {
            break;
        }
        println!("Retrying...");
    }

    classifier.save("classifier.safetensors")?;
    let mut classifier =
        Classifier::<MyClass>::load("classifier.safetensors", dev, input_size, &layers)?;

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
