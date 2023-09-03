use std::io::Write;

use floneum_rust::*;

#[export_plugin]
/// Writes some text to a file at the given path (in the /sandbox directory)
/// 
/// ### Examples
/// vec![
///     Example {
///         inputs: vec![String::from("Hello World!").into_input_value(), File::from(std::path::PathBuf::from("hello_world.txt")).into_input_value()],
///         outputs: vec![],
///     },
/// ]
pub fn write_to_file(
    /// The text to write to the file
    text: String,
    /// The path to the file to write to
    file_path: File,
) {
    if let Some(parent) = file_path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut file = std::fs::File::create(&*file_path).unwrap();
    file.write_all(text.as_bytes()).unwrap();
}
