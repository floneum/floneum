mod search;
pub use search::*;
mod calculator;
pub use calculator::*;
mod document;
pub use document::*;

/// A tool that can be used by a [`floneumin_language_model::Model`]
// TODO: Add example
#[async_trait::async_trait]
pub trait Tool {
    /// The name of the tool
    fn name(&self) -> String;
    /// A description of the tool
    fn description(&self) -> String;
    /// Run the tool with the given arguments
    async fn run(&mut self, args: &str) -> String;
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

    /// Get a tool by name
    pub fn get_tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.iter().find(|t| t.name() == name).map(|t| &**t)
    }

    /// Get a tool mutably by name
    pub fn get_tool_mut<'a>(&'a mut self, name: &str) -> Option<&'a mut dyn Tool> {
        for tool in &mut self.tools {
            if tool.name() == name {
                return Some(&mut **tool);
            }
        }
        None
    }

    /// Get a prompt for the tools in the manager
    pub fn prompt(&self, question: impl std::fmt::Display) -> String {
        let mut tools = String::new();
        let mut tool_names = String::new();
        for tool in self.tools.iter() {
            tools.push_str(&format!("# {}\n{}", tool.name(), tool.description()));
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

While you use tools keep in mind that duplicating a question will end the game. Begin!

Question: {question}
"#
        )
    }

    /// Get the constraints for the tools in the manager
    pub fn tool_choices(&self) -> Option<StructureParser> {
        let mut choices: Option<StructureParser> = None;
        for tool in self.tools.iter() {
            if let Some(current_choices) = choices.take() {
                choices = Some(StructureParser::Either {
                    first: Box::new(current_choices),
                    second: Box::new(StructureParser::Literal(tool.name())),
                });
            } else {
                choices = Some(StructureParser::Literal(tool.name()));
            }
        }
        choices
    }

    /// Get the constraints for the thought action
    pub fn thought_constraints(&self) -> impl Validate + Send + Sync {
        let constraints = "Thought: ";
        let constraints = constraints.then(OneLine);
        constraints
    }

    /// Get the constraints for the action action
    pub fn action_constraints(&self) -> impl Validate + Send + Sync {
        let constraints = "Action: ";
        let constraints = constraints.then(self.tool_choices().unwrap());
        let constraints = constraints.then(StructureParser::Literal("\nAction Input: ".into()));
        let constraints = constraints.then(OneLine);
        constraints
    }

    /// Get the constraints for the answer action
    pub fn answer_constraints(&self) -> impl Validate + Send + Sync {
        let constraints = "Final Answer: ";
        let constraints = constraints.then(OneLine);
        constraints
    }

    /// Get the constraints for any action
    pub fn any_action_constraint(&self) -> impl Validate + Send + Sync {
        self.thought_constraints()
            .or(self.action_constraints())
            .or(self.answer_constraints())
    }
}

struct OneLine;

impl Validate for OneLine {
    fn validate<'a>(&self, mut stream: ParseStream<'a>) -> ParseStatus<'a> {
        if stream.is_empty() {
            return ParseStatus::Incomplete {
                required_next: None,
            };
        }
        let mut iter = stream.iter();
        while let Some(c) = iter.next() {
            if c == '\n' {
                return ParseStatus::Complete(
                    iter.peek().is_some().then(|| ParseStream::from(iter)),
                );
            }
        }
        ParseStatus::Incomplete {
            required_next: None,
        }
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
