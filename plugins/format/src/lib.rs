use floneum_rust::*;

#[export_plugin]
/// formats a string
fn format(
    /// the format string
    template: String,
    /// the input to the format string
    input: String,
) -> String {
    template.replacen("{}", &input, 1)
}
