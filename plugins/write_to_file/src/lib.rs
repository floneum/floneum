use std::io::Write;

use floneum_rust::*;

#[export_plugin]
/// Writes some text to a file at the given path (in the /sandbox directory)
pub fn write_to_file(
    /// The text to write to the file
    text: String,
    /// The path to the file to write to
    file_path: File,
) {
    let mut file = std::fs::File::create(&*file_path).unwrap();
    file.write_all(text.as_bytes()).unwrap();
}
