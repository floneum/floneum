use std::collections::{HashMap, HashSet};

use sygus_types::{Run, Summary};
use walkdir::WalkDir;

#[derive(Copy, Clone, Debug, PartialEq)]
enum Model {
    SmolLm,        // smol-lm
    Qwen1_5b,      // qwen1.5b
    Qwen3b,        // qwen3b
    Qwen7b,        // qwen7b
    Qwen0_5b,      // qwen0.5b
    Llama1b,       // llama1b
    Llama3b,       // llama3b
    Llama8b,       // llama8b
    Qwen1_5bThink, // qwen1.5b-think
    Qwen7bThink,   // qwen7b-think
}

impl Model {
    const ALL: &[Model] = &[
        Model::SmolLm,
        Model::Qwen1_5b,
        Model::Qwen3b,
        Model::Qwen7b,
        Model::Qwen0_5b,
        // Model::Llama1b,
        // Model::Llama3b,
        // Model::Llama8b,
        // Model::Qwen1_5bThink,
        // Model::Qwen7bThink,
    ];

    fn as_str(&self) -> &'static str {
        match self {
            Model::Qwen1_5bThink => "qwen1.5b-think",
            Model::Qwen7bThink => "qwen7b-think",
            Model::SmolLm => "smol-lm",
            Model::Qwen1_5b => "qwen1.5b",
            Model::Qwen3b => "qwen3b",
            Model::Qwen7b => "qwen7b",
            Model::Qwen0_5b => "qwen0.5b",
            Model::Llama1b => "llama1b",
            Model::Llama3b => "llama3b",
            Model::Llama8b => "llama8b",
        }
    }
}

struct Runs {
    fast: bool,
    problem: String,
    model: Model,
    runs: Vec<Run>,
}

fn main() {
    let walker = WalkDir::new(".");
    let mut runs: Vec<Runs> = Vec::new();
    for entry in walker.into_iter().flatten() {
        let path = entry.path();
        if path.extension() == Some(std::ffi::OsStr::new("jsonl")) {
            let contents = std::fs::read_to_string(path).unwrap();
            let Some(model) = Model::ALL
                .iter()
                .find(|model| path.to_string_lossy().contains(model.as_str()))
                .copied()
            else {
                continue;
            };
            let problem = path.file_stem().unwrap().to_string_lossy();
            let problem = problem.split_at(problem.find(model.as_str()).unwrap()).0;
            let mut new_runs = Runs {
                fast: path.to_string_lossy().contains("true"),
                model,
                runs: Vec::new(),
                problem: problem.to_string(),
            };
            for line in contents.lines() {
                if let Ok(run) = serde_json::from_str(line) {
                    new_runs.runs.push(run);
                }
            }
            runs.push(new_runs);
        }
    }

    print_headers();

    print_summary("all", runs.iter());
    print_summary("fast", runs.iter().filter(|run| run.fast));
    print_summary("slow", runs.iter().filter(|run| !run.fast));

    // Print fast vs slow per model
    // Group by model
    for model in Model::ALL {
        // Find only problems that both models successfully ran
        let mut problems_ran: HashMap<String, Vec<bool>> = HashMap::new();
        for run in runs
            .iter()
            .filter(|run| run.model == *model && !run.runs.is_empty())
        {
            let entry = problems_ran.entry(run.problem.clone()).or_default();
            entry.push(run.fast);
        }
        let problems_ran_by_both = problems_ran
            .iter()
            .filter(|(_, v)| v.len() == 2)
            .map(|(k, _)| k.clone())
            .collect::<HashSet<_>>();
        print_summary(
            &format!("fast-{}", model.as_str()),
            runs.iter().filter(|run| {
                run.fast && run.model == *model && problems_ran_by_both.contains(&run.problem)
            }),
        );
        print_summary(
            &format!("slow-{}", model.as_str()),
            runs.iter().filter(|run| {
                !run.fast && run.model == *model && problems_ran_by_both.contains(&run.problem)
            }),
        );
    }
}

const LABELS: &[&str] = &[
    "label",
    "correct",
    "total",
    "sampler_time",
    "constraint_time",
    "parser_time",
    "transformer_time",
    "trie_time",
    "total_time",
    "average_entropy",
    "average_entropy_diff",
    "average_tokenization_error",
    "average_tokens_after_first_token_error",
    "average_token_count",
    "total_duration",
];

fn print_headers() {
    for (i, label) in LABELS.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("{label}");
    }
    println!();
}

fn print_summary<'a>(label: &str, runs: impl IntoIterator<Item = &'a Runs>) {
    let runs: Vec<_> = runs.into_iter().collect();
    let correct = runs
        .iter()
        .filter(|runs| runs.runs.iter().any(|run| run.pass))
        .count();
    let run_count = runs.len();
    let summary = runs
        .iter()
        .flat_map(|runs| runs.runs.iter())
        .cloned()
        .sum::<Summary>();
    print!("{},", label);
    print!("{},", correct);
    print!("{},", run_count);
    print!("{},", summary.average_metadata.sampler_time.as_millis());
    print!("{},", summary.average_metadata.constraint_time.as_millis());
    print!("{},", summary.average_metadata.parser_time.as_millis());
    print!("{},", summary.average_metadata.transformer_time.as_millis());
    print!("{},", summary.average_metadata.trie_time.as_millis());
    print!("{},", summary.average_metadata.total_time.as_millis());
    print!("{},", summary.average_entropy);
    print!("{},", summary.average_entropy_diff);
    print!("{},", summary.average_tokenization_error);
    print!("{},", summary.average_tokens_after_first_token_error);
    print!("{},", summary.average_token_count);
    print!("{}", summary.total_duration.as_millis());
    println!();
}
