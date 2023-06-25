use floneum_rust::*;

#[export_plugin]
/// formats a string
fn format(
    /// the format string
    template: String,
    /// the input to the format string
    input: Vec<String>,
) -> String {
    let mut input = input;
    let mut new_text = String::new();
    for section in template.split("{}") {
        new_text.push_str(section);
        if let Some(text) = input.pop() {
            new_text.push_str(&text);
        }
    }
    new_text
}
