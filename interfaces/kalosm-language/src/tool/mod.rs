//! Tools that can be used by [`kalosm_language_model::Model`]'s to perform actions.

mod search;
use std::{
    any::Any,
    borrow::Cow,
    error::Error,
    sync::{Arc, Mutex},
};

use kalosm_language_model::{GenerationParameters, SyncModel, SyncModelExt};
use kalosm_sample::{
    ArcParser, ChoiceParser, CreateParserState, Either, LiteralMismatchError, LiteralParser,
    LiteralParserOffset, ParseResult, Parser, ParserExt, SequenceParser, SequenceParserState,
};
pub use search::*;
mod calculator;
pub use calculator::*;
mod document;
pub use document::*;

/// A tool that can be used by a [`kalosm_language_model::Model`]
// TODO: Add example
#[async_trait::async_trait]
pub trait Tool {
    /// The constraints for the input to the tool
    type Constraint: Parser + CreateParserState;
    /// The constraints for the input to the tool
    fn constraints(&self) -> Self::Constraint;

    /// The name of the tool
    fn name(&self) -> String;
    /// The prompt for the input to the tool
    fn input_prompt(&self) -> String;
    /// A description of the tool
    fn description(&self) -> String;

    /// Run the tool with the given arguments
    async fn run(&mut self, args: <<Self as Tool>::Constraint as Parser>::Output) -> String;
}

/// An extension trait for [`Tool`] that allows for dynamic dispatch
pub trait DynToolExt {
    /// Convert a tool into a dynamic tool
    fn boxed(self) -> BoxedTool;
}

impl DynToolExt for BoxedTool {
    fn boxed(self) -> BoxedTool {
        self
    }
}

struct DynToolWrapper<T: Tool>
where
    <T::Constraint as Parser>::Output: Clone + Send + Sync + 'static,
    <T::Constraint as Parser>::PartialState: Send + Sync + 'static,
    <T::Constraint as Parser>::Error: Error + Send + Sync + 'static,
    <T as Tool>::Constraint: std::marker::Send + std::marker::Sync + 'static,
{
    tool: T,
}

#[async_trait::async_trait]
impl<T: Tool + Send> Tool for DynToolWrapper<T>
where
    <T::Constraint as Parser>::Output: Clone + Send + Sync + 'static,
    <T::Constraint as Parser>::PartialState: Send + Sync + 'static,
    <T::Constraint as Parser>::Error: Error + Send + Sync + 'static,
    <T as Tool>::Constraint: std::marker::Send + std::marker::Sync + 'static,
{
    type Constraint = ArcParser;
    fn constraints(&self) -> Self::Constraint {
        self.tool.constraints().boxed()
    }

    fn name(&self) -> String {
        self.tool.name()
    }
    fn input_prompt(&self) -> String {
        self.tool.input_prompt()
    }
    fn description(&self) -> String {
        self.tool.description()
    }
    async fn run(&mut self, args: <<Self as Tool>::Constraint as Parser>::Output) -> String {
        let args = args
            .downcast_ref::<<T::Constraint as Parser>::Output>()
            .unwrap()
            .clone();
        self.tool.run(args).await
    }
}

/// A dynamic tool that can be used by a [`kalosm_language_model::Model`]
pub type BoxedTool = Box<dyn Tool<Constraint = ArcParser> + Sync + Send>;

/// A set of tools that can be used by a [`kalosm_language_model::Model`]
#[derive(Default)]
pub struct ToolManager {
    tools: Vec<BoxedTool>,
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
    pub fn with_tool<T>(mut self, tool: T) -> Self
    where
        T: Tool + Send + Sync + 'static,
        <T::Constraint as Parser>::Output: Clone + Send + Sync + 'static,
        <T::Constraint as Parser>::PartialState: Send + Sync + 'static,
        <T::Constraint as Parser>::Error: Error + Send + Sync + 'static,
        <T as Tool>::Constraint: std::marker::Send + std::marker::Sync + 'static,
    {
        self.add_tool(tool);
        self
    }

    /// Add a tool to the manager
    pub fn add_tool<T>(&mut self, tool: T)
    where
        T: Tool + Send + Sync + 'static,
        <T::Constraint as Parser>::Output: Clone + Send + Sync + 'static,
        <T::Constraint as Parser>::PartialState: Send + Sync + 'static,
        <T::Constraint as Parser>::Error: Error + Send + Sync + 'static,
        <T as Tool>::Constraint: std::marker::Send + std::marker::Sync + 'static,
    {
        self.tools.push(Box::new(DynToolWrapper { tool }));
    }

    /// Get the tools in the manager
    pub fn get_tools(&self) -> &[BoxedTool] {
        &self.tools
    }

    /// Get a tool by name
    pub fn get_tool(&self, name: &str) -> Option<&BoxedTool> {
        self.tools.iter().find(|t| t.name() == name)
    }

    /// Get a tool mutably by name
    pub fn get_tool_mut<'a>(&'a mut self, name: &str) -> Option<&'a mut BoxedTool> {
        for tool in &mut self.tools {
            if tool.name() == name {
                return Some(&mut *tool);
            }
        }
        None
    }

    /// Get a tool by index
    pub fn get_tool_by_index(&self, index: usize) -> Option<&BoxedTool> {
        self.tools.get(index)
    }

    /// Get a tool mutably by index
    pub fn get_tool_mut_by_index(&mut self, index: usize) -> Option<&mut BoxedTool> {
        self.tools.get_mut(index)
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
            r#"Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

You have access to the following tools:

{tools}

Begin!

Question: {question}
"#
        )
    }

    /// Get the constraints for the tools in the manager
    pub fn tool_choices(
        &self,
    ) -> Option<
        IndexParser<
            SequenceParser<LiteralParser<String>, ArcParser>,
            Either<LiteralMismatchError, Arc<anyhow::Error>>,
            (
                (),
                Arc<(dyn std::any::Any + std::marker::Send + std::marker::Sync + 'static)>,
            ),
            SequenceParserState<LiteralParserOffset, Arc<dyn Any + Send + Sync>, ()>,
        >,
    > {
        let mut parsers = Vec::with_capacity(self.tools.len());
        for tool in self.tools.iter() {
            let name = tool.name();
            let prompt = tool.input_prompt();
            let input_constraint = tool.constraints();
            let tool_input_parser =
                LiteralParser::from(format!("{name}\n{prompt}")).then(input_constraint);
            parsers.push(tool_input_parser);
        }
        (!parsers.is_empty()).then_some(IndexParser { parsers })
    }

    /// Get the constraints for the thought action
    pub fn thought_constraints(
        &self,
    ) -> impl Parser<
        Error = Either<LiteralMismatchError, OneLineError>,
        Output = ((), String),
        PartialState = SequenceParserState<LiteralParserOffset, OneLineState, ()>,
    > + CreateParserState
           + Send
           + Sync
           + 'static {
        let constraints = "Thought: ";
        LiteralParser::from(constraints).then(OneLine)
    }

    /// Get the constraints for the action action
    pub fn action_constraints(
        &self,
    ) -> SequenceParser<
        LiteralParser<&'static str>,
        IndexParser<
            SequenceParser<LiteralParser<String>, ArcParser>,
            Either<LiteralMismatchError, Arc<anyhow::Error>>,
            ((), Arc<dyn Any + Send + Sync>),
            SequenceParserState<LiteralParserOffset, Arc<dyn Any + Send + Sync>, ()>,
        >,
    > {
        let constraints = LiteralParser::from("Action: ");
        constraints.then(self.tool_choices().unwrap())
    }

    /// Get the constraints for the answer action
    pub fn answer_constraints(
        &self,
    ) -> impl Parser<
        Error = Either<LiteralMismatchError, OneLineError>,
        Output = ((), String),
        PartialState = SequenceParserState<LiteralParserOffset, OneLineState, ()>,
    > + CreateParserState
           + Send
           + Sync
           + 'static {
        let constraints = LiteralParser::from("Final Answer: ");
        constraints.then(OneLine)
    }

    /// Get the constraints for any action
    pub fn any_action_constraint(
        &self,
    ) -> ChoiceParser<
        ChoiceParser<
            impl Parser<
                    Error = Either<LiteralMismatchError, OneLineError>,
                    Output = ((), String),
                    PartialState = SequenceParserState<LiteralParserOffset, OneLineState, ()>,
                > + CreateParserState
                + Send
                + Sync
                + 'static,
            SequenceParser<
                LiteralParser<&'static str>,
                IndexParser<
                    SequenceParser<LiteralParser<String>, ArcParser>,
                    Either<LiteralMismatchError, Arc<anyhow::Error>>,
                    ((), Arc<dyn Any + Send + Sync>),
                    SequenceParserState<LiteralParserOffset, Arc<dyn Any + Send + Sync>, ()>,
                >,
            >,
        >,
        impl Parser<
                Error = Either<LiteralMismatchError, OneLineError>,
                Output = ((), String),
                PartialState = SequenceParserState<LiteralParserOffset, OneLineState, ()>,
            > + CreateParserState
            + Send
            + Sync
            + 'static,
    > {
        self.thought_constraints()
            .or(self.action_constraints())
            .or(self.answer_constraints())
    }

    /// Run one step of the tool manager
    pub async fn run_step<M: SyncModel>(
        &mut self,
        prompt: &str,
        llm: &mut M,
        llm_session: &mut M::Session,
        mut add_token: impl FnMut(String) -> anyhow::Result<()>,
    ) -> anyhow::Result<ToolManagerStepResult> {
        let mut new_text = String::new();

        let constraints = self.any_action_constraint();
        let validator_state = constraints.create_parser_state();
        let result = llm.generate_structured(
            llm_session,
            prompt,
            constraints,
            validator_state,
            Arc::new(Mutex::new(GenerationParameters::default().sampler())),
            &mut add_token,
        )?;

        Ok(match result {
            Either::Left(Either::Left(thought)) => {
                new_text += &thought.1;
                new_text += "\n";
                add_token(new_text)?;
                ToolManagerStepResult::Thought(thought.1)
            }
            Either::Left(Either::Right(((), (tool_index, ((), left))))) => {
                let result = self
                    .get_tool_mut_by_index(tool_index)
                    .unwrap()
                    .run(left)
                    .await;
                new_text += &result;
                new_text += "\n";
                add_token(new_text)?;
                ToolManagerStepResult::Action(result)
            }
            Either::Right(right) => ToolManagerStepResult::Finished(right.1),
        })
    }
}

/// The result of a step in the tool manager
pub enum ToolManagerStepResult {
    /// The task was completed
    Finished(String),
    /// The model produced a new thought
    Thought(String),
    /// The model produced a new action result
    Action(String),
}

/// The state of the [`IndexParser`] parser
#[derive(Debug, Clone)]
pub struct IndexParserState<PA, E> {
    states: Vec<Result<PA, E>>,
}

/// A parser that parses a sequence of parsers and returns the index of the first parser that succeeds
#[derive(Debug, Clone)]
pub struct IndexParser<S: Parser<Error = E, Output = O, PartialState = PA>, E, O, PA> {
    parsers: Vec<S>,
}

impl<S: Parser<Error = E, Output = O, PartialState = PA>, E, O, PA> IndexParser<S, E, O, PA> {
    /// Create a new index parser
    pub fn new(parsers: Vec<S>) -> Self {
        Self { parsers }
    }
}

impl<S, E, O, PA> CreateParserState for IndexParser<S, E, O, PA>
where
    S: Parser<Error = E, Output = O, PartialState = PA> + CreateParserState,
    E: Clone,
    PA: Clone,
{
    fn create_parser_state(&self) -> Self::PartialState {
        IndexParserState {
            states: self
                .parsers
                .iter()
                .map(|s| Ok(s.create_parser_state()))
                .collect(),
        }
    }
}

impl<S, E, O, PA> Parser for IndexParser<S, E, O, PA>
where
    S: Parser<Error = E, Output = O, PartialState = PA>,
    E: Clone,
    PA: Clone,
{
    type Error = E;
    type Output = (usize, S::Output);
    type PartialState = IndexParserState<PA, E>;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<kalosm_sample::ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        let mut states = state.states.clone();
        let mut has_incomplete_option = false;
        let mut required_next: Option<Cow<'static, str>> = None;
        let last_index = self.parsers.len() - 1;
        for (i, parser) in self.parsers.iter().enumerate() {
            match &states[i] {
                Ok(state) => {
                    let result = parser.parse(state, input);
                    match result {
                        Ok(ParseResult::Finished {
                            result,
                            remaining: r,
                        }) => {
                            return Ok(ParseResult::Finished {
                                result: (i, result),
                                remaining: r,
                            })
                        }
                        Ok(ParseResult::Incomplete {
                            new_state: s,
                            required_next: new_required_next,
                        }) => {
                            states[i] = Ok(s);
                            has_incomplete_option = true;
                            match required_next {
                                Some(r) => {
                                    let mut common_bytes = 0;
                                    for (byte1, byte2) in r.bytes().zip(new_required_next.bytes()) {
                                        if byte1 != byte2 {
                                            break;
                                        }
                                        common_bytes += 1;
                                    }
                                    required_next = Some(match (r, new_required_next) {
                                        (Cow::Borrowed(required_next), _) => {
                                            Cow::Borrowed(&required_next[common_bytes..])
                                        }
                                        (_, Cow::Borrowed(required_next)) => {
                                            Cow::Borrowed(&required_next[common_bytes..])
                                        }
                                        (Cow::Owned(mut required_next), _) => {
                                            required_next.truncate(common_bytes);
                                            Cow::Owned(required_next)
                                        }
                                    });
                                }
                                None => {
                                    required_next = Some(new_required_next);
                                }
                            }
                        }
                        Err(e) => {
                            if !has_incomplete_option && i == last_index {
                                return Err(e);
                            }
                            states[i] = Err(e);
                        }
                    }
                }
                Err(err) => {
                    if !has_incomplete_option && i == last_index {
                        return Err(err.clone());
                    }
                }
            }
        }
        Ok(ParseResult::Incomplete {
            new_state: IndexParserState { states },
            required_next: required_next.unwrap_or_default(),
        })
    }
}

/// One line of text with some non-whitespace characters
#[derive(Debug, Clone, Copy)]
pub struct OneLine;

/// The state of the [`OneLine`] parser
#[derive(Debug, Clone)]
pub struct OneLineState {
    all_whitespace: bool,
    bytes: Vec<u8>,
}

impl CreateParserState for OneLine {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        OneLineState {
            all_whitespace: true,
            bytes: Vec::new(),
        }
    }
}

/// An error that can occur when parsing a [`OneLine`]
#[derive(Debug, Clone)]
pub struct OneLineError;

impl std::fmt::Display for OneLineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OneLineError")
    }
}

impl Error for OneLineError {}

impl Parser for OneLine {
    type Error = OneLineError;
    type Output = String;
    type PartialState = OneLineState;

    fn parse<'a>(
        &self,
        state: &Self::PartialState,
        input: &'a [u8],
    ) -> Result<kalosm_sample::ParseResult<'a, Self::PartialState, Self::Output>, Self::Error> {
        if input.is_empty() {
            return Ok(ParseResult::Incomplete {
                new_state: state.clone(),
                required_next: Default::default(),
            });
        }
        let mut state = state.clone();
        let mut iter = input.iter();
        while let Some(&c) = iter.next() {
            if state.all_whitespace {
                let c = char::from(c);
                if !c.is_whitespace() {
                    state.all_whitespace = false;
                }
            }
            if c == b'\n' || c == b'\r' {
                if state.all_whitespace {
                    return Err(OneLineError);
                } else {
                    return Ok(ParseResult::Finished {
                        result: String::from_utf8_lossy(&state.bytes).to_string(),
                        remaining: iter.as_slice(),
                    });
                }
            }
            state.bytes.push(c);
        }
        Ok(ParseResult::Incomplete {
            new_state: state,
            required_next: Default::default(),
        })
    }
}

macro_rules! impl_from_tool_tuple {
    ($($name:ident),*) => {
        #[allow(non_snake_case)]
        impl<$($name: Tool + Send + Sync + 'static),*> From<($($name,)*)> for ToolManager where
            $(
                <$name::Constraint as Parser>::Output: Clone+Send + Sync + 'static,
                <$name::Constraint as Parser>::PartialState: Send + Sync + 'static,
                <$name::Constraint as Parser>::Error: Error + Send + Sync + 'static,
                <$name as Tool>::Constraint: std::marker::Send + std::marker::Sync + 'static,
            )*
        {
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
