use floneum_plugin::exports::plugins::main::definitions::Input;

#[derive(Clone)]
pub struct Edge {
    pub start: usize,
    pub end: usize,
    pub value: Option<Input>,
}
