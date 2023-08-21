use floneum_rust::*;

#[export_plugin]
/// Formats some text by replacing any instances of {} in order with the texts passed in.
///
/// Example:
///
/// template: "Who is {}?"
/// inputs: "queen of england"
///
/// result: Who is the queen of england?
fn format(
    /// The template to format text with
    template: String,
    /// The inputs to the template
    input: Vec<String>,
) -> String {
    let mut new_text = String::new();
    let mut input_iter = input.into_iter();
    for section in template.split("{}") {
        new_text.push_str(section);
        if let Some(text) = input_iter.next() {
            new_text.push_str(&text);
        }
    }
    new_text
}
