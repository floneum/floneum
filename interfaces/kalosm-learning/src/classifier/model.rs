use rand::prelude::SliceRandom;
use std::collections::HashMap;

use candle_core::{
    safetensors::{self, Load},
    DType, Device, Result, Tensor, Var, D,
};
use candle_nn::{loss, ops, Dropout, Linear, Module, ModuleT, Optimizer, VarBuilder, VarMap};
use kalosm_common::maybe_autoreleasepool;
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

impl ClassificationDataset {
    /// Create a builder for a classification dataset.
    pub fn builder<C: Class>() -> ClassificationDatasetBuilder<C> {
        ClassificationDatasetBuilder::default()
    }

    /// Save the dataset to the given path.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let safetensors = HashMap::from([
            ("train_inputs".to_string(), self.train_inputs.clone()),
            ("train_classes".to_string(), self.train_classes.clone()),
            ("test_inputs".to_string(), self.test_inputs.clone()),
            ("test_classes".to_string(), self.test_classes.clone()),
        ]);

        safetensors::save(&safetensors, path)?;
        Ok(())
    }

    /// Load the dataset from the given path.
    pub fn load<P: AsRef<std::path::Path>>(path: P, dev: &Device) -> Result<Self> {
        let mut safetensors = safetensors::load(path, dev)?;
        Ok(Self {
            train_inputs: safetensors.remove("train_inputs").unwrap(),
            train_classes: safetensors.remove("train_classes").unwrap(),
            test_inputs: safetensors.remove("test_inputs").unwrap(),
            test_classes: safetensors.remove("test_classes").unwrap(),
        })
    }
}

/// A builder for [`ClassificationDataset`].
pub struct ClassificationDatasetBuilder<C: Class> {
    input_size: Option<usize>,
    inputs: Vec<Vec<f32>>,
    classes: Vec<C>,
}

impl<C: Class> Default for ClassificationDatasetBuilder<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Class> ClassificationDatasetBuilder<C> {
    /// Create a new dataset builder.
    pub fn new() -> Self {
        Self {
            input_size: None,
            inputs: Vec::new(),
            classes: Vec::new(),
        }
    }

    /// Adds a pair of input and class to the dataset.
    pub fn add(&mut self, input: Vec<f32>, class: C) {
        if let Some(input_size) = self.input_size {
            debug_assert_eq!(input.len(), input_size, "input size mismatch");
        } else {
            self.input_size = Some(input.len());
        }
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
        let mut input_test: Vec<f32> = Vec::with_capacity(first_quarter);
        let mut class_test: Vec<u32> = Vec::with_capacity(first_quarter);
        let mut inputs: Vec<f32> = Vec::with_capacity(self.inputs.len() - first_quarter);
        let mut classes: Vec<u32> = Vec::with_capacity(self.classes.len() - first_quarter);
        while !self.classes.is_empty() {
            let index = rng.gen_range(0..self.classes.len());
            let mut input = self.inputs.remove(index);
            let class = self.classes.remove(index);
            if class_test.len() <= first_quarter {
                input_test.append(&mut input);
                class_test.push(class.to_class());
            } else {
                inputs.append(&mut input);
                classes.push(class.to_class());
            }
        }

        let input_size = self.input_size.unwrap_or_default();

        // train
        let train_inputs_len = inputs.len() / input_size;
        let train_inputs = Tensor::from_vec(inputs, (train_inputs_len, input_size), dev)?;
        let train_classes_len = classes.len();
        debug_assert_eq!(train_inputs_len, train_classes_len);
        let train_classes = Tensor::from_vec(classes, train_classes_len, dev)?;

        // test
        let test_inputs_len = input_test.len() / input_size;
        let test_inputs = Tensor::from_vec(input_test, (test_inputs_len, input_size), dev)?;
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
    dropout: Dropout,
    dropout_rate: f32,
    phantom: std::marker::PhantomData<C>,
}

impl<C: Class> Classifier<C> {
    /// Create a new classifier.
    ///
    /// # Example
    /// ```rust
    /// use kalosm_learning::{Classifier, ClassifierConfig, Class};
    ///
    /// #[derive(Debug, Clone, Copy, Class)]
    /// enum MyClass {
    ///     Person,
    ///     Thing,
    /// }
    ///
    /// let dev = candle_core::Device::Cpu;
    /// let classifier = Classifier::<MyClass>::new(&dev, ClassifierConfig::new(4)).unwrap();
    /// ```
    pub fn new(dev: &Device, config: ClassifierConfig) -> Result<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        Self::new_inner(dev.clone(), varmap, vs, config)
    }

    /// Get the config of the classifier.
    pub fn config(&self) -> ClassifierConfig {
        ClassifierConfig {
            input_dim: self.layers.first().unwrap().weight().dims2().unwrap().1,
            layers_dims: {
                let mut layers = self
                    .layers
                    .iter()
                    .map(|l| l.weight().dims2().unwrap().0)
                    .collect::<Vec<_>>();
                layers.pop();
                layers
            },
            dropout_rate: self.dropout_rate,
        }
    }

    fn new_inner(
        dev: Device,
        varmap: VarMap,
        vs: VarBuilder,
        config: ClassifierConfig,
    ) -> Result<Self> {
        let ClassifierConfig {
            input_dim,
            layers_dims,
            dropout_rate,
        } = config;
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
                dropout: Dropout::new(dropout_rate),
                dropout_rate,
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
            dropout: Dropout::new(dropout_rate),
            dropout_rate,
            phantom: std::marker::PhantomData,
        })
    }

    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let xs = xs.clone();
        let mut xs = self.dropout.forward_t(&xs, train)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
            xs = xs.gelu_erf()?;
        }
        Ok(xs)
    }

    /// Train the model on the given dataset.
    ///
    /// # Example
    /// ```rust
    /// use kalosm_learning::{Classifier, ClassifierConfig, Class, ClassificationDatasetBuilder};
    ///
    /// #[derive(Debug, Clone, Copy, Class)]
    /// enum MyClass {
    ///     Person,
    ///     Thing,
    /// }
    ///
    /// let dev = candle_core::Device::Cpu;
    /// let mut classifier = Classifier::<MyClass>::new(&dev, ClassifierConfig::new(4)).unwrap();
    /// let mut dataset = ClassificationDatasetBuilder::new();
    /// dataset.add(vec![1.0, 2.0, 3.0, 4.0], MyClass::Person);
    /// dataset.add(vec![4.0, 3.0, 2.0, 1.0], MyClass::Thing);
    ///
    /// classifier.train(&dataset.build(&dev).unwrap(), &dev, 20, 0.05).unwrap();
    /// ```
    pub fn train(
        &mut self,
        m: &ClassificationDataset,
        dev: &Device,
        epochs: usize,
        learning_rate: f64,
        batch_size: usize,
    ) -> anyhow::Result<f32> {
        // unstack both tensors into a list of tensors
        let train_len = m.train_inputs.dims()[0];
        let train_results = m.train_classes.chunk(train_len, 0)?;
        let train_votes = m.train_inputs.chunk(train_len, 0)?;

        let mut sgd = candle_nn::AdamW::new_lr(self.varmap.all_vars(), learning_rate)?;
        let test_votes = m.test_inputs.to_device(dev)?;
        let test_results = m.test_classes.to_device(dev)?;
        let mut final_accuracy: f32 = 0.0;
        let mut rng = rand::thread_rng();
        let mut batch = 0;
        for epoch in 1..epochs + 1 {
            // create a random batch of indices
            let mut indices = (0..train_len).collect::<Vec<_>>();
            indices.shuffle(&mut rng);
            maybe_autoreleasepool(|| {
                for indices in indices.chunks(batch_size) {
                    let train_results = Tensor::cat(
                        &indices
                            .iter()
                            .copied()
                            .map(|i| train_results[i].clone())
                            .collect::<Vec<_>>(),
                        0,
                    )?
                    .to_device(dev)?;
                    let train_votes = Tensor::cat(
                        &indices
                            .iter()
                            .copied()
                            .map(|i| train_votes[i].clone())
                            .collect::<Vec<_>>(),
                        0,
                    )?
                    .to_device(dev)?;

                    let logits = self.forward_t(&train_votes, true)?;
                    let log_sm = ops::log_softmax(&logits, D::Minus1)?;
                    let loss = loss::nll(&log_sm, &train_results)?;
                    sgd.backward_step(&loss)?;
                    println!("Batch: {batch:5} Loss: {:5.5}", loss.to_scalar::<f32>()?);
                    batch += 1;
                }
                let test_logits = self.forward_t(&test_votes, false)?;
                let test_cases_passed = test_logits
                    .argmax(D::Minus1)?
                    .eq(&test_results)?
                    .to_dtype(DType::U32)?
                    .sum_all()?
                    .to_scalar::<u32>()?;
                let test_cases = test_results.dims1()?;
                let test_accuracy: f32 = test_cases_passed as f32 / test_cases as f32;
                final_accuracy = f32::from(100u8) * test_accuracy;
                println!(
                    "Epoch: {epoch:5} Test accuracy: {:5.5}% ({}/{})",
                    final_accuracy, test_cases_passed, test_cases,
                );
                Ok::<_, anyhow::Error>(())
            })?;
        }
        Ok(final_accuracy)
    }

    /// Save the model to a safetensors file at the given path.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kalosm_learning::{Classifier, ClassifierConfig, Class};
    ///
    /// #[derive(Debug, Clone, Copy, Class)]
    /// enum MyClass {
    ///     Person,
    ///     Thing,
    /// }
    ///
    /// let dev = candle_core::Device::Cpu;
    /// let classifier = Classifier::<MyClass>::new(&dev, ClassifierConfig::new(4)).unwrap();
    /// classifier.save("classifier.safetensors").unwrap();
    /// ```
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.varmap.save(path)
    }

    /// Load the model from a safetensors file at the given path.
    ///
    /// # Example
    ///
    /// ```rust, no_run
    /// use kalosm_learning::{Classifier, ClassifierConfig, Class};
    ///
    /// #[derive(Debug, Clone, Copy, Class)]
    /// enum MyClass {
    ///     Person,
    ///     Thing,
    /// }
    ///
    /// let dev = candle_core::Device::Cpu;
    /// let classifier = Classifier::<MyClass>::load("classifier.safetensors", &dev, ClassifierConfig::new(4)).unwrap();
    /// ```
    pub fn load(
        path: impl AsRef<std::path::Path>,
        dev: &Device,
        config: ClassifierConfig,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        {
            let safetensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }?;
            let tensors = safetensors.tensors();
            let mut tensor_data = varmap.data().lock().unwrap();
            for (name, value) in tensors {
                let tensor = value.load(dev)?;
                tensor_data.insert(name.to_string(), Var::from_tensor(&tensor)?);
            }
        }
        Self::new_inner(dev.clone(), varmap, vs, config)
    }

    /// Run the model on the given input.
    ///
    /// # Example
    ///
    /// use kalosm_learning::{Classifier, ClassifierConfig, Class};
    ///
    /// #[derive(Debug, Clone, Copy, Class)]
    /// enum MyClass {
    ///     Person,
    ///     Thing,
    /// }
    ///
    /// let dev = candle_core::Device::Cpu;
    /// let classifier = Classifier::<MyClass>::new(&dev, ClassifierConfig::new(4)).unwrap();
    /// let result = classifier.run(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    /// println!("Result: {:?}", result);
    /// ```
    pub fn run(&mut self, input: &[f32]) -> Result<ClassifierOutput<C>> {
        let input = Tensor::from_vec(input.to_vec(), (1, input.len()), &self.device)?;
        let logits = self.forward_t(&input, false)?;
        let classes = logits
            .flatten_all()?;
        let classes = ops::softmax(&classes, D::Minus1)?;
        let classes = classes.to_vec1()?;
        Ok(ClassifierOutput {
            classes: classes
                .into_iter().enumerate()
                .map(|(i, c)| (C::from_class(i as u32), c))
                .collect(),
        })
    }
}

/// The output of a classifier.
#[derive(Debug, Clone)]
pub struct ClassifierOutput<C: Class> {
    /// The classes along with their probabilities.
    classes: Box<[(C, f32)]>,
}

impl<C: Class> ClassifierOutput<C> {
    /// Get the probabilities of each class.
    pub fn classes(&self) -> &[(C, f32)] {
        &self.classes
    }

    /// Get the top class with the highest probability.
    pub fn top(&self) -> C where C: Clone {
        self.classes
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(c, _)| c.clone())
            .unwrap()
    }
}

#[derive(Debug, Clone)]
/// A config for a [`Classifier`].
pub struct ClassifierConfig {
    /// The input dimension.
    input_dim: usize,
    /// The dimensions of the layers.
    layers_dims: Vec<usize>,
    /// The dropout rate.
    dropout_rate: f32,
}

impl ClassifierConfig {
    /// Create a new config.
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            layers_dims: vec![4, 8, 4],
            dropout_rate: 0.1,
        }
    }

    /// Set the dimensions of the layers.
    pub fn layers_dims(mut self, layers_dims: Vec<usize>) -> Self {
        self.layers_dims = layers_dims;
        self
    }

    /// Set the dropout rate.
    pub fn dropout_rate(mut self, dropout_rate: f32) -> Self {
        self.dropout_rate = dropout_rate;
        self
    }
}
