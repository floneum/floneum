use floneum_rust::*;

#[export_plugin]
/// Takes one number to the power of another number
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![2.into_input_value(), 2.into_input_value()],
///         outputs: vec![4.into_return_value()]
///     },
/// ]
fn power(
    /// the number
    number: i64,
    /// the power
    power: i64,
) -> i64 {
    number.pow(power as u32)
}
