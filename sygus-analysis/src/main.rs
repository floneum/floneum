use sygus_types::{Run, Summary};
use walkdir::WalkDir;

struct Runs {
    fast: bool,
    runs: Vec<Run>,
}

fn main() {
    let walker = WalkDir::new(".");
    let mut runs: Vec<Runs> = Vec::new();
    for entry in walker.into_iter().flatten() {
        let path = entry.path();
        if path.extension() == Some(std::ffi::OsStr::new("jsonl")) {
            let contents = std::fs::read_to_string(path).unwrap();
            let mut new_runs = Runs {
                fast: path.to_string_lossy().contains("true"),
                runs: Vec::new(),
            };
            for line in contents.lines() {
                if let Ok(run) = serde_json::from_str(line) {
                    new_runs.runs.push(run);
                }
            }
            runs.push(new_runs);
        }
    }

    println!("Overall Summary:");
    print_summary(
        &runs
            .iter()
            .flat_map(|runs| runs.runs.iter())
            .cloned()
            .sum::<Summary>(),
    );

    let fast: Vec<_> = runs.iter().filter(|run| run.fast).collect();
    let slow: Vec<_> = runs.iter().filter(|run| !run.fast).collect();
    println!("Fast Summary:");
    println!(
        "\truns: {}",
        fast.iter().map(|runs| runs.runs.len()).sum::<usize>()
    );
    println!(
        "\tpass rate: {}/{}",
        fast.iter()
            .filter(|runs| runs.runs.iter().any(|run| run.pass))
            .count(),
        fast.len()
    );
    print_summary(
        &fast
            .iter()
            .flat_map(|runs| runs.runs.iter())
            .cloned()
            .sum::<Summary>(),
    );

    println!("Slow Summary:");
    println!(
        "\truns: {}",
        slow.iter().map(|runs| runs.runs.len()).sum::<usize>()
    );
    println!(
        "\tpass rate: {}/{}",
        slow.iter()
            .filter(|runs| runs.runs.iter().any(|run| run.pass))
            .count(),
        slow.len()
    );
    print_summary(
        &slow
            .iter()
            .flat_map(|runs| runs.runs.iter())
            .cloned()
            .sum::<Summary>(),
    );
}

fn print_summary(summary: &Summary) {
    println!("\taverage_metadata: {}", summary.average_metadata);
    println!("\taverage_entropy: {}", summary.average_entropy);
    println!("\taverage_entropy_diff: {}", summary.average_entropy_diff);
    println!(
        "\taverage_tokenization_error: {}",
        summary.average_tokenization_error
    );
    println!(
        "\taverage_tokens_after_first_token_error: {}",
        summary.average_tokens_after_first_token_error
    );
    println!("\taverage_token_count: {}", summary.average_token_count);
    println!("\ttotal_time: {:?}", summary.total_duration);
}
