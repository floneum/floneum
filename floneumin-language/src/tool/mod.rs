use floneumin_sample::structured_parser::StructureParser;
mod search;
pub use search::*;

#[async_trait::async_trait]
pub trait Tool {
    fn name(&self) -> String;
    fn description(&self) -> String;
    fn constraints(&self) -> StructureParser;
    async fn run(&self, args: Vec<String>) -> String;
}

#[derive(Default)]
pub struct ToolManager {
    tools: Vec<Box<dyn Tool>>,
}

impl std::fmt::Debug for ToolManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolManager")
            .field(
                "tools",
                &self.tools.iter().map(|t| t.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl ToolManager {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn with_tool(self, tool: impl Tool + 'static) -> Self {
        let mut tools = self.tools;
        tools.push(Box::new(tool));
        Self { tools }
    }

    pub fn add_tool(&mut self, tool: impl Tool + 'static) {
        self.tools.push(Box::new(tool));
    }

    pub fn get_tools(&self) -> &[Box<dyn Tool>] {
        &self.tools
    }

    pub fn prompt(&self, question: impl std::fmt::Display) -> String {
        let mut tools = String::new();
        let mut tool_names = String::new();
        for (i, tool) in self.tools.iter().enumerate() {
            tools.push_str(&format!("{}: {}\n{}", i, tool.name(), tool.description()));
            tool_names.push_str(&format!("'{}'", tool.name()));
        }
        format!(
            r#"Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
Thought:"#
        )
    }
}

macro_rules! impl_from_tool_tuple {
    ($($name:ident),*) => {
        #[allow(non_snake_case)]
        impl<$($name: Tool + 'static),*> From<($($name,)*)> for ToolManager {
            fn from(tools: ($($name,)*)) -> Self {
                let ($($name,)*) = tools;
                Self::new()$(.with_tool($name))*
            }
        }
    };
}

impl_from_tool_tuple!();
impl_from_tool_tuple!(A);
impl_from_tool_tuple!(A, B);
impl_from_tool_tuple!(A, B, C);
impl_from_tool_tuple!(A, B, C, D);
impl_from_tool_tuple!(A, B, C, D, E);
impl_from_tool_tuple!(A, B, C, D, E, F);
impl_from_tool_tuple!(A, B, C, D, E, F, G);
impl_from_tool_tuple!(A, B, C, D, E, F, G, H);
impl_from_tool_tuple!(A, B, C, D, E, F, G, H, I);
impl_from_tool_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_from_tool_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_from_tool_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
