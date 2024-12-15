use std::any::{Any, TypeId};

use kalosm_language::prelude::*;
use rand::{random, seq::index::sample, Rng};

use crate::{BertDistance, Metric, TestCases};

/// A builder for [`PromptAnnealer`].
pub struct PromptAnnealerBuilder<
    'a,
    M: Model,
    P = ChannelTextStream,
    Met: Metric<String> = BertDistance,
> where
    <<M as Model>::SyncModel as SyncModel>::Session: Sync + Send,
{
    llm: &'a mut M,
    metric: Option<Met>,
    train: &'a [(&'static str, &'static str)],
    test: &'a [(&'static str, &'static str)],
    task: TaskBuilder<P>,
    initial_temperature: f64,
    decay_rate: f64,
    cutoff_temperature: f64,
    initial_population: usize,
    initial_choice_range: std::ops::Range<usize>,
}

impl<'a, M: Model, P> PromptAnnealer<'a, M, P>
where
    <<M as Model>::SyncModel as SyncModel>::Session: Sync + Send,
    P: Clone + TaskBuilderReturn + Send + Sync + 'static,
{
    /// Create a new builder for [`PromptAnnealer`].
    pub fn builder(
        model: &'a mut M,
        train_set: &'a [(&'static str, &'static str)],
        task: TaskBuilder<P>,
    ) -> PromptAnnealerBuilder<'a, M, P, BertDistance> {
        PromptAnnealerBuilder {
            llm: model,
            train: train_set,
            task,
            test: &[],
            initial_temperature: 0.6,
            decay_rate: 0.9,
            cutoff_temperature: 0.10,
            initial_population: 10,
            initial_choice_range: 1..3,
            metric: None,
        }
    }
}

impl<'a, M: Model, P, Met: Metric<String> + 'static> PromptAnnealerBuilder<'a, M, P, Met>
where
    <<M as Model>::SyncModel as SyncModel>::Session: Sync + Send,
    P: Clone + TaskBuilderReturn + Send + Sync + 'static,
{
    /// Set the test set to use for evaluation. If no test set is provided, a subset of the train set will be used.
    pub fn with_test_set(mut self, test_set: &'a [(&'static str, &'static str)]) -> Self {
        self.test = test_set;
        self
    }

    /// Set the initial temperature for the annealing process a higher temperature will allow for more exploration, but it will also take longer to converge to a solution.
    pub fn with_initial_temperature(mut self, temperature: f64) -> Self {
        self.initial_temperature = temperature;
        self
    }

    /// Set the initial population size. A larger population will allow for more exploration, but it will also take longer to run.
    pub fn with_initial_population(mut self, population: usize) -> Self {
        self.initial_population = population;
        self
    }

    /// Set the initial range of examples to choose from.
    pub fn with_initial_choice_range(mut self, range: std::ops::Range<usize>) -> Self {
        self.initial_choice_range = range;
        self
    }

    /// Set the decay rate for the temperature. A higher decay rate will cause the temperature to decrease faster which will allow for faster convergence, but it will also increase the risk of getting stuck in a local optimum.
    pub fn with_decay_rate(mut self, rate: f64) -> Self {
        self.decay_rate = rate;
        self
    }

    /// Set the cutoff temperature. Once the temperature reaches this value, the annealing process will stop.
    pub fn with_cutoff_temperature(mut self, temperature: f64) -> Self {
        self.cutoff_temperature = temperature;
        self
    }

    /// Build the [`PromptAnnealer`].
    pub async fn build(self) -> Result<PromptAnnealer<'a, M, P, Met>> {
        let mut metric = match self.metric {
            Some(metric) => metric,
            None => {
                if TypeId::of::<Met>() == TypeId::of::<BertDistance>() {
                    *(Box::new(BertDistance::new(Bert::builder().build().await?)) as Box<dyn Any>)
                        .downcast::<Met>()
                        .unwrap()
                } else {
                    return Err(anyhow::anyhow!("No metric provided"));
                }
            }
        };

        let (train_set, test_set) = if self.test.is_empty() {
            tracing::warn!("No test set provided, using a subset of the train set for evaluation");

            let split = (self.train.len() / 3).max(1);

            assert!(
                split < self.train.len() || split < 1,
                "Train set is too small to split into train and test sets. Provide more examples."
            );

            (&self.train[split..], &self.train[..split])
        } else {
            (self.train, self.test)
        };

        let bert = Bert::new().await?;

        // Calculate embeddings for all examples
        let mut embedded_train_set = Vec::new();

        for train_example in train_set {
            let embedding = bert
                .embed(train_example.0)
                .await
                .expect("Failed to embed input");

            embedded_train_set.push(Example {
                input: train_example.0,
                output: train_example.1,
                embedding,
            });
        }

        let mut embedded_test_set = Vec::new();

        for test_example in test_set {
            let embedding = bert
                .embed(test_example.0)
                .await
                .expect("Failed to embed input");

            embedded_test_set.push(Example {
                input: test_example.0,
                output: test_example.1,
                embedding,
            });
        }

        let train_set = embedded_train_set;
        let test_set = embedded_test_set;

        assert!(!train_set.is_empty(), "Train set is empty");
        assert!(!test_set.is_empty(), "Test set is empty");

        assert!(
            self.initial_choice_range.end < train_set.len(),
            "Initial choice range may select more examples than the train set contains"
        );

        let mut population = Vec::new();

        let mut rng = rand::thread_rng();
        for _ in 0..self.initial_population {
            let amount = rng.gen_range(self.initial_choice_range.clone());
            let index_vec = sample(&mut rng, train_set.len(), amount);
            let index_vec = index_vec.iter().collect::<Vec<_>>();

            let mut chosen_cases = Vec::new();
            assert!(index_vec.len() <= self.initial_choice_range.end);

            for index in &index_vec {
                chosen_cases.push(train_set[*index].clone());
            }

            let remaining_cases = train_set
                .iter()
                .enumerate()
                .filter(|(i, _)| !index_vec.contains(i))
                .map(|(_, x)| x.clone())
                .collect::<Vec<_>>();

            population.push(
                ExamplesInstance::new(
                    self.llm,
                    chosen_cases,
                    remaining_cases,
                    self.initial_temperature,
                    &test_set,
                    &mut metric,
                    self.task.clone(),
                )
                .await,
            );
        }

        Ok(PromptAnnealer {
            task: self.task,
            llm: self.llm,
            test: test_set,
            population,
            metric,
            decay_rate: self.decay_rate,
            cutoff_temperature: self.cutoff_temperature,
        })
    }
}

/// A prompt annealer that takes a set of examples and tries to find the best combination and order of examples to use as a prompt for a given task.
pub struct PromptAnnealer<'a, M: Model, P = ChannelTextStream, Met: Metric<String> = BertDistance>
where
    <<M as Model>::SyncModel as SyncModel>::Session: Sync + Send,
{
    task: TaskBuilder<P>,
    llm: &'a M,
    metric: Met,
    test: Vec<Example<'static>>,
    population: Vec<ExamplesInstance>,
    decay_rate: f64,
    cutoff_temperature: f64,
}

impl<'a, M: Model, P, Met> PromptAnnealer<'a, M, P, Met>
where
    <<M as Model>::SyncModel as SyncModel>::Session: Sync + Send,
    P: Clone + TaskBuilderReturn + Send + Sync + 'static,
    Met: Metric<String>,
{
    /// Run the annealing process.
    pub async fn run(&mut self) -> Vec<AnnealingResult> {
        loop {
            for instance in &mut self.population {
                instance
                    .mutate(&self.test, self.llm, &mut self.metric, self.task.clone())
                    .await;
                instance.temperature *= self.decay_rate;
            }

            if self.population[0].temperature < self.cutoff_temperature {
                break self
                    .population
                    .iter()
                    .map(|instance| AnnealingResult {
                        examples: instance
                            .current_examples
                            .iter()
                            .map(|x| (x.input, x.output))
                            .collect(),
                        score: instance.current_evaluation,
                    })
                    .collect();
            }
            println!("current temperature: {}", self.population[0].temperature);

            for instance in &self.population {
                println!(
                    "(score = {}) {:?}",
                    instance.current_evaluation, instance.current_examples
                );
            }
        }
    }
}

/// A result example configuration produced by the annealing process.
#[derive(Debug, Clone)]
pub struct AnnealingResult {
    /// The examples used in the configuration.
    pub examples: Vec<(&'static str, &'static str)>,
    /// The score of the configuration.
    pub score: f64,
}

struct ExamplesInstance {
    current_examples: Vec<Example<'static>>,
    unused_examples: Vec<Example<'static>>,
    current_evaluation: f64,
    temperature: f64,
}

impl ExamplesInstance {
    async fn new<M, P>(llm: &mut M, current_examples:Vec<Example<'static>>,
    unused_examples:Vec<Example<'static>>,  temperature:f64,
    test:&[Example<'static>],
    metric: &mut impl Metric<String>,
    task: TaskBuilder<P>,
    ) -> Self where M: Model,
    <<M as kalosm_language::prelude::Model>::SyncModel as kalosm_language::prelude::SyncModel>::Session: Sync+ Send,
    P: TaskBuilderReturn + Send + Sync + 'static,{
        let current_evaluation = evaluate(&current_examples, test, llm, metric, task).await;

        Self {
            current_examples,
            unused_examples,
            current_evaluation,
            temperature,
        }
    }

    async fn mutate<M, P>(
        &mut self,
        test: &[Example<'static>],
        llm: &M,
        metric: &mut impl Metric<String>,
        task: TaskBuilder<P>,
    ) where
        M: Model,
        <M::SyncModel as SyncModel>::Session: Send + Sync,
        P: TaskBuilderReturn + Send + Sync + 'static,
    {
        let action = if self.current_examples.is_empty() {
            2
        } else if self.unused_examples.is_empty() {
            random::<usize>() % 2
        } else {
            random::<usize>() % 3
        };

        let mut mutated_examples = self.current_examples.clone();

        match action {
            // remove example
            0 => {
                let index = random::<usize>() % mutated_examples.len();
                let removed = mutated_examples.remove(index);

                let new_evaluation = evaluate(&mutated_examples, test, llm, metric, task).await;
                let accept_regardless = std::f64::consts::E
                    .powf((self.current_evaluation - new_evaluation) / self.temperature)
                    > random::<f64>();

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

                let new_evaluation = evaluate(&mutated_examples, test, llm, metric, task).await;
                let accept_regardless = std::f64::consts::E
                    .powf((self.current_evaluation - new_evaluation) / self.temperature)
                    > random::<f64>();

                if accept_regardless || new_evaluation > self.current_evaluation {
                    self.current_evaluation = new_evaluation;
                    self.current_examples = mutated_examples;
                }
            }
            // add example
            _ => {
                let index = random::<usize>() % self.unused_examples.len();
                let added = self.unused_examples[index].clone();
                mutated_examples.push(added);

                let new_evaluation = evaluate(&mutated_examples, test, llm, metric, task).await;
                let accept_regardless = std::f64::consts::E
                    .powf((self.current_evaluation - new_evaluation) / self.temperature)
                    > random::<f64>();

                if accept_regardless || new_evaluation > self.current_evaluation {
                    self.current_evaluation = new_evaluation;
                    self.current_examples = mutated_examples;
                    self.unused_examples.remove(index);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Example<'a> {
    input: &'a str,
    output: &'a str,
    embedding: Embedding<BertSpace>,
}

async fn evaluate<'a, M: Model, P>(
    examples: &[Example<'static>],
    test: &[Example<'static>],
    llm: &M,
    metric: &mut impl Metric<String>,
    task: TaskBuilder<P>,
) -> f64
where
    <M::SyncModel as SyncModel>::Session: Send + Sync,
    P: TaskBuilderReturn + Send + Sync + 'static,
{
    let examples_tokens: usize = examples
        .iter()
        .filter_map(|example| {
            llm.tokenizer()
                .encode(example.input, false)
                .ok()
                .map(|x| x.len())
                .and_then(|x| Some(x + llm.tokenizer().encode(example.output, false).ok()?.len()))
        })
        .sum();

    let task = task
        .with_examples(
            examples
                .iter()
                .map(|example| (example.input, example.output)),
        )
        .build();

    let mut llama_test_cases = TestCases::new();

    for example in test {
        let mut actual = task.run(example.input, llm);

        let all_text = actual.all_text().await;

        llama_test_cases.push_case(example.output.to_string(), all_text);
    }

    let llama_distance = llama_test_cases.evaluate(metric).await.normalized();

    println!("evaluating examples {:?}", examples);
    println!("{}", llama_distance);

    let similarity_scope = llama_distance.mean_score();
    let token_penalty = examples_tokens as f64 * 0.0001;
    let mut diversity_bonus = {
        // We want to incentivize diversity in the examples, so we calculate the average distance between all examples and add it as a bonus.
        examples
            .iter()
            .enumerate()
            .map(|(i, example)| {
                examples
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| i != *j)
                    .map(|(_, other_example)| {
                        example
                            .embedding
                            .cosine_similarity(&other_example.embedding)
                    })
                    .sum::<f32>()
                    / examples.len() as f32
            })
            .sum::<f32>()
            / examples.len() as f32
    };

    if diversity_bonus.is_nan() {
        diversity_bonus = 0.0;
    }

    // diversity_bonus should now be in the range 0..1
    if diversity_bonus <= 1.0 {
        println!("diversity bonus: {}", diversity_bonus);
    }

    println!(
        "similarity scope: {}, token penalty: {}, diversity bonus: {}",
        similarity_scope, token_penalty, diversity_bonus
    );

    let final_score = similarity_scope - token_penalty + diversity_bonus as f64 / 2.;

    println!("final score: {}", final_score);

    final_score
}
