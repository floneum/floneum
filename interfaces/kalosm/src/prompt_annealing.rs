use kalosm_language::{
    kalosm_language_model::{Model, SyncModel},
    search::Hypothetical,
};
use rand::{random, seq::index::sample, Rng};

use crate::{BertDistance, Metric, TestCases};

/// A builder for [`PromptAnnealer`].
pub struct PromptAnnealerBuilder<'a, M: Model, Met: Metric<String> = BertDistance>
where
    <<M as Model>::SyncModel as SyncModel>::Session: Sync + Send,
{
    llm: &'a mut M,
    metric: Met,
    train: &'a [(&'static str, &'static str)],
    test: &'a [(&'static str, &'static str)],
    initial_temperature: f64,
    decay_rate: f64,
    cutoff_temperature: f64,
    initial_population: usize,
    initial_choice_range: std::ops::Range<usize>,
}

impl<'a, M: Model> PromptAnnealer<'a, M>
where
    <<M as Model>::SyncModel as SyncModel>::Session: Sync + Send,
{
    /// Create a new builder for [`PromptAnnealer`].
    pub fn builder(
        model: &'a mut M,
        train_set: &'a [(&'static str, &'static str)],
    ) -> PromptAnnealerBuilder<'a, M, BertDistance> {
        PromptAnnealerBuilder {
            llm: model,
            train: train_set,
            test: &[],
            initial_temperature: 0.6,
            decay_rate: 0.9,
            cutoff_temperature: 0.10,
            initial_population: 10,
            initial_choice_range: 1..3,
            metric: BertDistance::default(),
        }
    }
}

impl<'a, M: Model, Met: Metric<String>> PromptAnnealerBuilder<'a, M, Met>
where
    <<M as Model>::SyncModel as SyncModel>::Session: Sync + Send,
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
    pub async fn build(mut self) -> PromptAnnealer<'a, M, Met> {
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
                chosen_cases.push(train_set[*index]);
            }

            let remaining_cases = train_set
                .iter()
                .enumerate()
                .filter(|(i, _)| !index_vec.contains(i))
                .map(|(_, x)| *x)
                .collect::<Vec<_>>();

            population.push(
                ExamplesInstance::new(
                    self.llm,
                    chosen_cases,
                    remaining_cases,
                    self.initial_temperature,
                    test_set,
                    &mut self.metric,
                )
                .await,
            );
        }

        PromptAnnealer {
            llm: self.llm,
            test: test_set,
            population,
            metric: self.metric,
            decay_rate: self.decay_rate,
            cutoff_temperature: self.cutoff_temperature,
        }
    }
}

/// A prompt annealer that takes a set of examples and tries to find the best combination and order of examples to use as a prompt for a given task.
pub struct PromptAnnealer<'a, M: Model, Met: Metric<String> = BertDistance>
where
    <<M as Model>::SyncModel as SyncModel>::Session: Sync + Send,
{
    llm: &'a mut M,
    metric: Met,
    test: &'a [(&'static str, &'static str)],
    population: Vec<ExamplesInstance>,
    decay_rate: f64,
    cutoff_temperature: f64,
}

impl<'a, M: Model> PromptAnnealer<'a, M>
where
    <<M as Model>::SyncModel as SyncModel>::Session: Sync + Send,
{
    /// Run the annealing process.
    pub async fn run(&mut self) -> Vec<AnnealingResult> {
        loop {
            for instance in &mut self.population {
                instance.mutate(self.test, self.llm, &mut self.metric).await;
                instance.temperature *= self.decay_rate;
            }

            if self.population[0].temperature < self.cutoff_temperature {
                break self
                    .population
                    .iter()
                    .map(|instance| AnnealingResult {
                        examples: instance.current_examples.clone(),
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
    current_examples: Vec<(&'static str, &'static str)>,
    unused_examples: Vec<(&'static str, &'static str)>,
    current_evaluation: f64,
    temperature: f64,
}

impl ExamplesInstance {
    async fn new<M: Model>(llm: &mut M, current_examples:Vec<(&'static str, &'static str)>,
    unused_examples:Vec<(&'static str, &'static str),>,  temperature:f64
    ,
    test:&[(&'static str, &'static str)],
    metric: &mut impl Metric<String>
    ) -> Self where <<M as kalosm_language::prelude::Model>::SyncModel as kalosm_language::prelude::SyncModel>::Session: Sync+ Send{
        let current_evaluation = evaluate(&current_examples, test, llm, metric).await;

        Self {
            current_examples,
            unused_examples,
            current_evaluation,
            temperature,
        }
    }

    async fn mutate<M: Model>(&mut self,
        test:&[(&'static str, &'static str)],
    llm: &mut M,
    metric: &mut impl Metric<String>
    ) where <<M as kalosm_language::prelude::Model>::SyncModel as kalosm_language::prelude::SyncModel>::Session: Sync+ Send{
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

                let new_evaluation = evaluate(&mutated_examples, test, llm, metric).await;

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

                let new_evaluation = evaluate(&mutated_examples, test, llm, metric).await;

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

                let new_evaluation = evaluate(&mutated_examples, test, llm, metric).await;

                if accept_regardless || new_evaluation > self.current_evaluation {
                    self.current_evaluation = new_evaluation;
                    self.current_examples = mutated_examples;
                    self.unused_examples.remove(index);
                }
            }
        }
    }
}

async fn evaluate<M: Model>(examples: &[(&str, &str)], test:&[(&str, &str)], llm: &mut M, metric: &mut impl Metric<String>) -> f64 where <<M as kalosm_language::prelude::Model>::SyncModel as kalosm_language::prelude::SyncModel>::Session: Sync+ Send{
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

    for (text, expected) in test {
        let actual = &hypothetical.generate_question(text, llm).await.unwrap()[0];

        llama_test_cases.push_case(expected.to_string(), actual.clone());
    }

    let llama_distance = llama_test_cases.evaluate(metric).await.normalized();

    println!("evaluating examples {:?}", examples);
    println!("{}", llama_distance);

    llama_distance.mean_score() - examples_tokens as f64 * 0.0001
}
