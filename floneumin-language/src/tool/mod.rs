pub trait Tool{
    fn description(&self) -> String;
    fn constraints(&self) -> String;
    fn run(&self, args: Vec<String>) -> String;
}
