use floneum_rust::*;

#[export_plugin]
/// Take a slice of an list between two positions in the list. The start index is inclusive and the end index is exclusive.
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![vec![String::from("Text to slice"), String::from("Another text to slice")].into_input_value(), 1.into_input_value(), 1.into_input_value()],
///         outputs: vec![vec![String::from("Text to split")].into_return_value()]
///     },
/// ]
fn slice(
    /// The list to slice
    list: Vec<String>,
    /// The position to start the slice at
    start: i64,
    /// The position to end the slice at
    end: i64,
) -> Vec<String> {
    let start_index = if start >= 0 {
        start as usize - 1
    } else {
        (list.len() as i64 + start) as usize
    };
    let end_index = if end >= 0 {
        end as usize - 1
    } else {
        (list.len() as i64 + end) as usize
    };

    let min_index = start_index.min(end_index);
    let max_index = start_index.max(end_index);

    list[min_index..max_index].to_vec()
}
