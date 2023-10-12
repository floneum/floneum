use floneumin_sample::StructureParser;
mod search;
pub use search::*;

/// A tool that can be used by a [`floneumin_language_model::Model`]
// TODO: Add example
#[async_trait::async_trait]
pub trait Tool {
    /// The name of the tool
    fn name(&self) -> String;
    /// A description of the tool
    fn description(&self) -> String;
    /// The constraints to use when filling in the parameters for the tool
    fn constraints(&self) -> StructureParser;
    /// Run the tool with the given arguments
    async fn run(&self, args: Vec<String>) -> String;
}

/// A set of tools that can be used by a [`floneumin_language_model::Model`]
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
    /// Create a new tool empty manager
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    /// Add a tool to the manager
    pub fn with_tool(self, tool: impl Tool + 'static) -> Self {
        let mut tools = self.tools;
        tools.push(Box::new(tool));
        Self { tools }
    }

    /// Add a tool to the manager
    pub fn add_tool(&mut self, tool: impl Tool + 'static) {
        self.tools.push(Box::new(tool));
    }

    /// Get the tools in the manager
    pub fn get_tools(&self) -> &[Box<dyn Tool>] {
        &self.tools
    }

    /// Get a prompt for the tools in the manager
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
