
use kalosm::language::*;
use std::fmt::Display;

#[tokio::main]
async fn main() {
    println!("Downloading and starting model...");
    let model = Llama::builder()
        .with_source(LlamaSource::functionary_7b())
        .build()
        .await
        .unwrap();
    println!("Model ready");
    
    let functions = FunctionCallingPrompt{
        prompt: String::from(r#"You are an assistant that navigates a webpage like a user."#),
        functions: vec![
            FunctionDefinition {
                name: "search_page".to_string(),
                description: "Search the current page for text".to_string(),
                parameters: vec![Parameter {
                    name: "query".to_string(),
                    ty: "string".to_string(),
                    description: "The search query.".to_string(),
                }],
                return_type: "string".to_string(),
            },
            FunctionDefinition {
                name: "google_search".to_string(),
                description: "Search for information on a particular topic".to_string(),
                parameters: vec![Parameter {
                    name: "query".to_string(),
                    ty: "string".to_string(),
                    description: "The search query.".to_string(),
                }],
                return_type: "string".to_string(),
            },
            FunctionDefinition {
                name: "click_element".to_string(),
                description: "Click an element on the page".to_string(),
                parameters: vec![Parameter {
                    name: "button".to_string(),
                    ty: "string".to_string(),
                    description: "The button to click.".to_string(),
                }],
                return_type: "any".to_string(),
            }
        ],
    };
    let prompt = prompt_input("Question: ").unwrap();
    let prompt = format!("{functions}<|from|>user
<|recipient|>all
<|content|>{prompt}
<|from|>assistant
<|recipient|>");
    let model_stream = model
        .stream_text(&prompt)
        .with_max_length(1000)
        // .with_stop_on("<|stop|>")
        .await
        .unwrap();

    model_stream.to_std_out().await.unwrap();
}

struct FunctionCallingPrompt {
    prompt: String,
    functions: Vec<FunctionDefinition>,
}

impl Display for FunctionCallingPrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { prompt, functions } = self;
        let mut functions_string = String::new();
        for function in functions.iter() {
            functions_string += &function.to_string();
        }
        write!(
            f,
            "<|from|>system
<|recipient|>all
<|content|>// Supported function definitions that should be called when necessary.
namespace functions {{

{functions_string}

}} // namespace functions
<|from|>system
<|recipient|>all
<|content|>{prompt}
"
        )
    }
}

struct FunctionDefinition {
    name: String,
    description: String,
    parameters: Vec<Parameter>,
    return_type: String,
}

impl Display for FunctionDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = &self.name;
        let description = &self.description;
        let return_type = &self.return_type;
        let mut parameters = String::new();
        for parameter in self.parameters.iter() {
            parameters += &parameter.to_string();
        }
        write!(
            f,
            "// {description}
type {name} = (_: {{
{parameters}
}}) => {return_type};"
        )
    }
}

struct Parameter {
    name: String,
    ty: String,
    description: String,
}

impl Display for Parameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            name,
            ty,
            description,
        } = self;
        write!(
            f,
            "// {description}
{name}: {ty}"
        )
    }
}
