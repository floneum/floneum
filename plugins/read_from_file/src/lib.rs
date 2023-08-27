use floneum_rust::*;

#[export_plugin]
/// Reads some text from a file at the given path (in the /sandbox directory)
pub fn read_from_file(
    /// The path to the file to read from
    file_path: File,
) -> String {
    let file_path = std::path::PathBuf::from(&*file_path);
    std::fs::read_to_string(file_path).unwrap()
}
