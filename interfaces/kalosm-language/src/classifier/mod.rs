use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap};
use rand::Rng;

const EPOCHS: usize = 10;
const LEARNING_RATE: f64 = 0.5;

pub trait Class {
    const CLASSES: u32;

    fn to_class(&self) -> u32;
    fn from_class(class: u32) -> Self;
}

#[derive(Clone, Debug)]
pub struct Dataset {
    pub train_inputs: Tensor,
    pub train_classes: Tensor,
    pub test_inputs: Tensor,
    pub test_classes: Tensor,
}

pub struct DatasetBuilder<D: candle_core::WithDType, C: Class> {
    input_size: usize,
    inputs: Vec<Vec<D>>,
    classes: Vec<C>,
}

impl<D: candle_core::WithDType, C: Class> DatasetBuilder<D, C> {
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
    pub fn build(mut self, dev: &Device) -> Result<Dataset> {
        // split into train and test
        let mut rng = rand::thread_rng();
        let first_quarter = self.inputs.len() / 4;
        let mut input_test: Vec<D> = Vec::with_capacity(first_quarter);
        let mut class_test: Vec<u32> = Vec::with_capacity(first_quarter);
        let mut inputs: Vec<D> = Vec::with_capacity(self.inputs.len() - first_quarter);
        let mut classes: Vec<u32> = Vec::with_capacity(self.classes.len() - first_quarter);
        while !self.classes.is_empty() {
            let index = rng.gen_range(0..self.classes.len());
            let mut input = self.inputs.remove(index);
            let class = self.classes.remove(index);
            if input_test.len() < first_quarter {
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

        Ok(Dataset {
            train_inputs,
            train_classes,
            test_inputs,
            test_classes,
        })
    }
}

struct Classifier<C: Class> {
    varmap: VarMap,
    layers: Vec<Linear>,
    phantom: std::marker::PhantomData<C>,
}

impl<C: Class> Classifier<C> {
    fn new(dev: &Device, input_dim: usize, layers_dims: &[usize]) -> Result<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);

        let output_dim = C::CLASSES;
        if layers_dims.is_empty() {
            return Ok(Self {
                varmap,
                layers: vec![candle_nn::linear(input_dim, output_dim as usize, vs.pp("output"))?],
                phantom: std::marker::PhantomData,
            });
        }
        let mut layers = Vec::with_capacity(layers_dims.len() + 1);
        layers.push(candle_nn::linear(
            input_dim,
            *layers_dims.first().unwrap(),
            vs.pp("input"),
        )?);
        for (i, (in_dim, out_dim)) in layers_dims
            .iter()
            .zip(layers_dims.iter().skip(1))
            .enumerate()
        {
            layers.push(candle_nn::linear(
                *in_dim,
                *out_dim,
                vs.pp(&format!("ln{}", i)),
            )?);
        }
        layers.push(candle_nn::linear(
            *layers_dims.last().unwrap(),
            output_dim as usize,
            vs.pp("output"),
        )?);
        Ok(Self {
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

    fn train(&mut self, m: &Dataset, dev: &Device) -> anyhow::Result<()> {
        let train_results = m.train_classes.to_device(dev).unwrap();
        let train_votes = m.train_inputs.to_device(dev).unwrap();

        let mut sgd = candle_nn::SGD::new(self.varmap.all_vars(), LEARNING_RATE).unwrap();
        let test_votes = m.test_inputs.to_device(dev).unwrap();
        let test_results = m.test_classes.to_device(dev).unwrap();
        let mut final_accuracy: f32 = 0.0;
        for epoch in 1..EPOCHS + 1 {
            let logits = self.forward(&train_votes).unwrap();
            let log_sm = ops::log_softmax(&logits, D::Minus1).unwrap();
            let loss = loss::nll(&log_sm, &train_results).unwrap();
            sgd.backward_step(&loss).unwrap();

            let test_logits = self.forward(&test_votes).unwrap();
            let sum_ok = test_logits
                .argmax(D::Minus1)
                .unwrap()
                .eq(&test_results)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let test_accuracy: f32 = sum_ok / test_results.dims1().unwrap() as f32;
            final_accuracy = f32::from(100u8) * test_accuracy;
            println!(
                "Epoch: {epoch:3} Train loss: {:8.5} Test accuracy: {:5.2}%",
                loss.to_scalar::<f32>().unwrap(),
                final_accuracy
            );
            if final_accuracy == 100.0 {
                break;
            }
        }
        if final_accuracy < 100.0 {
            Err(anyhow::Error::msg("The model is not trained well enough."))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
enum MyClass {
    Yes,
    Maybe,
    No,
}

impl Class for MyClass {
    const CLASSES: u32 = 3;

    fn to_class(&self) -> u32 {
        *self as u32
    }

    fn from_class(class: u32) -> Self {
        match class {
            0 => Self::Yes,
            1 => Self::Maybe,
            2 => Self::No,
            _ => panic!("Invalid class."),
        }
    }
}

#[test]
fn simplified() -> anyhow::Result<()> {
    let dev = Device::cuda_if_available(0).unwrap();

    let mut dataset = DatasetBuilder::<f32, MyClass>::new(2);

    for _ in 0..100 {
        // Yes is a random point in the first quadrant
        dataset.add(
            vec![
                rand::thread_rng().gen_range(0.0..1.0),
                rand::thread_rng().gen_range(0.0..1.0),
            ],
            MyClass::Yes,
        );
        // Maybe is a random point in the second quadrant
        dataset.add(
            vec![
                rand::thread_rng().gen_range(-1.0..0.0),
                rand::thread_rng().gen_range(0.0..1.0),
            ],
            MyClass::Maybe,
        );
        // No is a random point in the third quadrant
        dataset.add(
            vec![
                rand::thread_rng().gen_range(-1.0..0.0),
                rand::thread_rng().gen_range(-1.0..0.0),
            ],
            MyClass::No,
        );
    }

    let dataset = dataset.build(&dev)?;
    println!("{:?}", dataset);

    let mut error = true;
    while error {
        let mut classifier = Classifier::<MyClass>::new(&dev, 2, &[3, 2]).unwrap();
    
        error = classifier.train(&dataset, &dev).is_err();
    }
    Ok(())
}
