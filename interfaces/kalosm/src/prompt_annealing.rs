use kalosm_language::{
    kalosm_language_model::{Model, SyncModel},
    search::Hypothetical,
};
use rand::{
    random,
    seq::{index::sample, SliceRandom},
    Rng,
};

use crate::{BertDistance, Metric, TestCases};

pub struct PromptAnnealerBuilder<'a, M: Model, Met: Metric<String> = BertDistance>
where
    <<M as Model>::SyncModel as SyncModel>::Session: Sync + Send,
{
    llm: &'a mut M,
    metric: Met,
    train: &'a [(&'static str, &'static str)],
    test: &'a [(&'static str, &'static str)],
    initial_temperature: f64,
    initial_population: usize,
    initial_choice_range: std::ops::Range<usize>,
}

impl<'a, M: Model> PromptAnnealer<'a, M>
where
    <<M as Model>::SyncModel as SyncModel>::Session: Sync + Send,
{
    pub fn builder(
        model: &'a mut M,
        train_set: &'a [(&'static str, &'static str)],
    ) -> PromptAnnealerBuilder<'a, M, BertDistance> {
        PromptAnnealerBuilder {
            llm: model,
            train: train_set,
            test: &[],
            initial_temperature: 0.6,
            initial_population: 20,
            initial_choice_range: 0..3,
            metric: BertDistance::default(),
        }
    }
}

impl<'a, M: Model, Met: Metric<String>> PromptAnnealerBuilder<'a, M, Met>
where
    <<M as Model>::SyncModel as SyncModel>::Session: Sync + Send,
{
    pub fn with_test_set(mut self, test_set: &'a [(&'static str, &'static str)]) -> Self {
        self.test = test_set;
        self
    }

    pub fn with_initial_temperature(mut self, temperature: f64) -> Self {
        self.initial_temperature = temperature;
        self
    }

    pub fn with_initial_population(mut self, population: usize) -> Self {
        self.initial_population = population;
        self
    }

    pub fn with_initial_choice_range(mut self, range: std::ops::Range<usize>) -> Self {
        self.initial_choice_range = range;
        self
    }

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

            let mut chosen_cases = train_set.to_vec();
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
            train: train_set,
            test: test_set,
            population,
            metric: self.metric,
        }
    }
}

pub struct PromptAnnealer<'a, M: Model, Met: Metric<String> = BertDistance>
where
    <<M as Model>::SyncModel as SyncModel>::Session: Sync + Send,
{
    llm: &'a mut M,
    metric: Met,
    train: &'a [(&'static str, &'static str)],
    test: &'a [(&'static str, &'static str)],
    population: Vec<ExamplesInstance>,
}

impl<'a, M: Model> PromptAnnealer<'a, M>
where
    <<M as Model>::SyncModel as SyncModel>::Session: Sync + Send,
{
    pub async fn run(&mut self) {
        loop {
            for instance in &mut self.population {
                instance.mutate(self.test, self.llm, &mut self.metric).await;
            }

            if self.population[0].temperature < 0.10 {
                break;
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

                if accept_regardless {
                    self.current_examples = mutated_examples;
                    self.unused_examples.push(removed);
                } else {
                    let new_evaluation = evaluate(&mutated_examples, test, llm, metric).await;

                    if new_evaluation > self.current_evaluation {
                        self.current_evaluation = new_evaluation;
                        self.current_examples = mutated_examples;
                        self.unused_examples.push(removed);
                    }
                }
            }
            // swap examples
            1 => {
                let index1 = random::<usize>() % mutated_examples.len();
                let index2 = random::<usize>() % mutated_examples.len();

                mutated_examples.swap(index1, index2);

                if accept_regardless {
                    self.current_examples = mutated_examples;
                } else {
                    let new_evaluation = evaluate(&mutated_examples, test, llm, metric).await;

                    if new_evaluation > self.current_evaluation {
                        self.current_evaluation = new_evaluation;
                        self.current_examples = mutated_examples;
                    }
                }
            }
            // add example
            _ => {
                let index = random::<usize>() % self.unused_examples.len();
                let added = self.unused_examples[index];
                mutated_examples.push(added);

                if accept_regardless {
                    self.current_examples = mutated_examples;
                    self.unused_examples.remove(index);
                } else {
                    let new_evaluation = evaluate(&mutated_examples, test, llm, metric).await;

                    if new_evaluation > self.current_evaluation {
                        self.current_evaluation = new_evaluation;
                        self.current_examples = mutated_examples;
                        self.unused_examples.remove(index);
                    }
                }
            }
        }

        self.temperature *= 0.9;
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

    println!("evaluating {} examples {:?}", examples.len(), examples);
    println!("{}", llama_distance);

    llama_distance.mean_score() - examples_tokens as f64 * 0.0001
}
