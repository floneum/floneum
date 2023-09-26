use std::io::Write;

use rphi::*;

fn main() -> anyhow::Result<()> {
    let mut phi = Phi::default();

    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut question = String::new();
        std::io::stdin().read_line(&mut question)?;
        let prompt = format!(
            "Think step by step to complete the following prompt.\n1) What is the relevant context I need to complete this prompt?\n2) How should I approach completing this prompt?\n3) What should I complete this prompt with show the entire prompt alongside your completion?\n4) Repeat just the completion from the previous question\nUse those steps to complete the following prompt:\n{question}\n1) What is the relevant context I need to complete this prompt?\nThe relevant context is"
        );
        phi.run(&InferenceSettings::new(&prompt).with_sample_len(1000))?;
    }
}
