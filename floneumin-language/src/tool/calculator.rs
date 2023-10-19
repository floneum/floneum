use crate::tool::Tool;

/// A tool that can search the web
pub struct CalculatorTool;

#[async_trait::async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> String {
        "Calculator".to_string()
    }

    fn description(&self) -> String {
        "Evaluate a mathematical expression (made only of numbers and one of the prebuilt math functions). Available functions: sqrt, abs, exp, ln, sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, asinh, acosh, atanh, floor, ceil, round, signum, pi, e\nUse tool with:\nAction: Calculator\nAction Input: the expression\nExample:\nQuestion: What is 2 + 2?\nThought: I should calculate 2 + 2.\nAction: Calculator\nAction Input: 2 + 2\nObservation: 4\nThought: I now know that 2 + 2 is 4.\nFinal Answer: 4".to_string()
    }

    async fn run(&mut self, expr: &str) -> String {
        match meval::eval_str(expr){
            Ok(result) => result.to_string(),
            Err(e) => format!("Input was invalid, try again making sure to only use numbers and one of the prebuilt math functions. {e}"),
        }
    }
}
