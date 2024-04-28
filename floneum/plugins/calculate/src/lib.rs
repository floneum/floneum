use floneum_rust::*;

#[export_plugin]
/// Evaluate the value of an equation
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![String::from("1+2*3").into_input_value()],
///         outputs: vec![(7 as f64).into_return_value()]
///     },
/// ]
fn calculate(
    /// The equation
    equation: String,
) -> f64 {
    meval::eval_str(equation).expect("Invalid Equation")
}
