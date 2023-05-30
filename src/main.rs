#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::{
    egui::{self, DragValue, TextEdit, TextStyle},
    epaint::{ahash::HashSet, Rect},
};
use egui_node_graph::*;
use plugin::exports::plugins::main::definitions::ValueType;
use plugin::{Plugin, PluginEngine, PluginInstance};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, path::PathBuf};

#[tokio::main]
async fn main() {
    use eframe::egui::Visuals;

    eframe::run_native(
        "Egui node graph example",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(Visuals::dark());
            Box::<NodeGraphExample>::default()
        }),
    )
    .expect("Failed to run native example");
}

// ========= First, define your user data types =============

/// The NodeData holds a custom data struct inside each node. It's useful to
/// store additional information that doesn't live in parameters. For this
/// example, the node data stores the template (i.e. the "type") of the node.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct MyNodeData {
    #[serde(skip)]
    instance: PluginInstance,
}

/// `DataType`s are what defines the possible range of connections when
/// attaching two ports together. The graph UI will make sure to not allow
/// attaching incompatible datatypes.
#[derive(PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MyDataType {
    Text,
    Embedding,
}

/// In the graph, input parameters can optionally have a constant value. This
/// value can be directly edited in a widget inside the node itself.
///
/// There will usually be a correspondence between DataTypes and ValueTypes. But
/// this library makes no attempt to check this consistency. For instance, it is
/// up to the user code in this example to make sure no parameter is created
/// with a DataType of Text and a ValueType of Embedding.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MyValueType {
    Text(String),
    Embedding,
}

impl Default for MyValueType {
    fn default() -> Self {
        // NOTE: This is just a dummy `Default` implementation. The library
        // requires it to circumvent some internal borrow checker issues.
        Self::Text(String::new())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
pub struct PluginId(usize);

/// The response type is used to encode side-effects produced when drawing a
/// node in the graph. Most side-effects (creating new nodes, deleting existing
/// nodes, handling connections...) are already handled by the library, but this
/// mechanism allows creating additional side effects from user code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MyResponse {
    SetActiveNode(NodeId),
    ClearActiveNode,
}

/// The graph 'global' state. This state struct is passed around to the node and
/// parameter drawing callbacks. The contents of this struct are entirely up to
/// the user. For this example, we use it to keep track of the 'active' node.
#[derive(Default, serde::Serialize, serde::Deserialize)]
pub struct MyGraphState {
    #[serde(skip)]
    pub plugin_engine: PluginEngine,
    pub active_node: Option<NodeId>,
    #[serde(skip)]
    pub plugins: slab::Slab<Plugin>,
    pub all_plugins: HashSet<PluginId>,
}

impl MyGraphState {
    fn get_plugin(&self, id: PluginId) -> &Plugin {
        &self.plugins[id.0]
    }
}

// =========== Then, you need to implement some traits ============

// A trait for the data types, to tell the library how to display them
impl DataTypeTrait<MyGraphState> for MyDataType {
    fn data_type_color(&self, _user_state: &mut MyGraphState) -> egui::Color32 {
        match self {
            MyDataType::Text => egui::Color32::from_rgb(38, 109, 211),
            MyDataType::Embedding => egui::Color32::from_rgb(238, 207, 109),
        }
    }

    fn name(&self) -> Cow<'_, str> {
        match self {
            MyDataType::Text => Cow::Borrowed("Text"),
            MyDataType::Embedding => Cow::Borrowed("2d vector"),
        }
    }
}

// A trait for the node kinds, which tells the library how to build new nodes
// from the templates in the node finder
impl NodeTemplateTrait for PluginId {
    type NodeData = MyNodeData;
    type DataType = MyDataType;
    type ValueType = MyValueType;
    type UserState = MyGraphState;
    type CategoryType = &'static str;

    fn node_finder_label(&self, user_state: &mut Self::UserState) -> Cow<'_, str> {
        Cow::Owned(user_state.get_plugin(*self).name())
    }

    // this is what allows the library to show collapsible lists in the node finder.
    fn node_finder_categories(&self, _user_state: &mut Self::UserState) -> Vec<&'static str> {
        vec!["Plugins"]
    }

    fn node_graph_label(&self, user_state: &mut Self::UserState) -> String {
        // It's okay to delegate this to node_finder_label if you don't want to
        // show different names in the node finder and the node itself.
        self.node_finder_label(user_state).into()
    }

    fn user_data(&self, user_state: &mut Self::UserState) -> Self::NodeData {
        MyNodeData {
            instance: user_state
                .get_plugin(*self)
                .instance(&user_state.plugin_engine),
        }
    }

    fn build_node(
        &self,
        graph: &mut Graph<Self::NodeData, Self::DataType, Self::ValueType>,
        _user_state: &mut Self::UserState,
        node_id: NodeId,
    ) {
        // The nodes are created empty by default. This function needs to take
        // care of creating the desired inputs and outputs based on the template

        // We define some closures here to avoid boilerplate. Note that this is
        // entirely optional.
        let input_text = |graph: &mut MyGraph, name: &str| {
            graph.add_input_param(
                node_id,
                name.to_string(),
                MyDataType::Text,
                MyValueType::Text(String::new()),
                InputParamKind::ConnectionOrConstant,
                true,
            );
        };
        let input_vector = |graph: &mut MyGraph, name: &str| {
            graph.add_input_param(
                node_id,
                name.to_string(),
                MyDataType::Embedding,
                MyValueType::Embedding,
                InputParamKind::ConnectionOrConstant,
                true,
            );
        };

        let output_text = |graph: &mut MyGraph, name: &str| {
            graph.add_output_param(node_id, name.to_string(), MyDataType::Text);
        };
        let output_vector = |graph: &mut MyGraph, name: &str| {
            graph.add_output_param(node_id, name.to_string(), MyDataType::Embedding);
        };

        let node = &graph[node_id];

        let meta = node.user_data.instance.metadata().clone();

        for input in &meta.inputs {
            let name = &input.name;
            match &input.ty {
                ValueType::Text => input_text(graph, name),
                ValueType::Embedding => input_vector(graph, name),
            }
        }

        for output in &meta.outputs {
            let name = &output.name;
            match &output.ty {
                ValueType::Text => output_text(graph, name),
                ValueType::Embedding => output_vector(graph, name),
            }
        }
    }
}

pub struct AllMyNodeTemplates(Vec<PluginId>);

impl NodeTemplateIter for AllMyNodeTemplates {
    type Item = PluginId;

    fn all_kinds(&self) -> Vec<Self::Item> {
        // This function must return a list of node kinds, which the node finder
        // will use to display it to the user. Crates like strum can reduce the
        // boilerplate in enumerating all variants of an enum.
        self.0.clone()
    }
}

impl WidgetValueTrait for MyValueType {
    type Response = MyResponse;
    type UserState = MyGraphState;
    type NodeData = MyNodeData;
    fn value_widget(
        &mut self,
        param_name: &str,
        _node_id: NodeId,
        ui: &mut egui::Ui,
        _user_state: &mut MyGraphState,
        _node_data: &MyNodeData,
    ) -> Vec<MyResponse> {
        // This trait is used to tell the library which UI to display for the
        // inline parameter widgets.
        match self {
            MyValueType::Embedding => {
                ui.label(param_name);
                ui.horizontal(|ui| {
                    ui.label("embedding");
                });
            }
            MyValueType::Text(value) => {
                ui.horizontal(|ui| {
                    ui.label(param_name);
                    ui.add(TextEdit::singleline(value));
                });
            }
        }
        // This allows you to return your responses from the inline widgets.
        Vec::new()
    }
}

impl UserResponseTrait for MyResponse {}
impl NodeDataTrait for MyNodeData {
    type Response = MyResponse;
    type UserState = MyGraphState;
    type DataType = MyDataType;
    type ValueType = MyValueType;

    // This method will be called when drawing each node. This allows adding
    // extra ui elements inside the nodes. In this case, we create an "active"
    // button which introduces the concept of having an active node in the
    // graph. This is done entirely from user code with no modifications to the
    // node graph library.
    fn bottom_ui(
        &self,
        _ui: &mut egui::Ui,
        _node_id: NodeId,
        _graph: &Graph<MyNodeData, MyDataType, MyValueType>,
        _user_state: &mut Self::UserState,
    ) -> Vec<NodeResponse<MyResponse, MyNodeData>>
    where
        MyResponse: UserResponseTrait,
    {
        // This logic is entirely up to the user. In this case, we check if the
        // current node we're drawing is the active one, by comparing against
        // the value stored in the global user state, and draw different button
        // UIs based on that.

        vec![]
    }
}

type MyGraph = Graph<MyNodeData, MyDataType, MyValueType>;
type MyEditorState = GraphEditorState<MyNodeData, MyDataType, MyValueType, PluginId, MyGraphState>;

#[derive(Default)]
pub struct NodeGraphExample {
    // The `GraphEditorState` is the top-level object. You "register" all your
    // custom types by specifying it as its generic parameters.
    state: MyEditorState,

    user_state: MyGraphState,

    search_text: String,
}

const PERSISTENCE_KEY: &str = "egui_node_graph";

impl NodeGraphExample {
    /// If the persistence feature is enabled, Called once before the first frame.
    /// Load previous app state (if any).
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let state = cc
            .storage
            .and_then(|storage| eframe::get_value(storage, PERSISTENCE_KEY))
            .unwrap_or_default();
        Self {
            state,
            user_state: MyGraphState::default(),
            search_text: String::new(),
        }
    }
}

impl eframe::App for NodeGraphExample {
    /// If the persistence function is enabled,
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, PERSISTENCE_KEY, &self.state);
    }
    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                egui::widgets::global_dark_light_mode_switch(ui);
                let response = ui.add(egui::TextEdit::singleline(&mut self.search_text));
                if response.changed() {
                    println!("search text changed: {}", self.search_text)
                }
                if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    let path = PathBuf::from(&self.search_text);
                    if path.exists() {
                        println!("path exists");
                        let plugin = self.user_state.plugin_engine.load_plugin(&path);
                        let id = self.user_state.plugins.insert(plugin);
                        self.user_state.all_plugins.insert(PluginId(id));
                    }
                }
            });
        });

        let _graph_response = egui::CentralPanel::default()
            .show(ctx, |ui| {
                self.state.draw_graph_editor(
                    ui,
                    AllMyNodeTemplates(self.user_state.all_plugins.iter().copied().collect()),
                    &mut self.user_state,
                    Vec::default(),
                )
            })
            .inner;

        // for node_response in graph_response.node_responses {
        //     // Here, we ignore all other graph events. But you may find
        //     // some use for them. For example, by playing a sound when a new
        //     // connection is created
        //     if let NodeResponse::User(user_event) = node_response {
        //         match user_event {
        //             MyResponse::SetActiveNode(node) => self.user_state.active_node = Some(node),
        //             MyResponse::ClearActiveNode => self.user_state.active_node = None,
        //         }
        //     }
        // }
    }
}

// type OutputsCache = HashMap<OutputId, MyValueType>;

// /// Recursively evaluates all dependencies of this node, then evaluates the node itself.
// pub fn evaluate_node(
//     graph: &MyGraph,
//     node_id: NodeId,
//     outputs_cache: &mut OutputsCache,
// ) -> anyhow::Result<MyValueType> {
//     // To solve a similar problem as creating node types above, we define an
//     // Evaluator as a convenience. It may be overkill for this small example,
//     // but something like this makes the code much more readable when the
//     // number of nodes starts growing.

//     struct Evaluator<'a> {
//         graph: &'a MyGraph,
//         outputs_cache: &'a mut OutputsCache,
//         node_id: NodeId,
//     }
//     impl<'a> Evaluator<'a> {
//         fn new(graph: &'a MyGraph, outputs_cache: &'a mut OutputsCache, node_id: NodeId) -> Self {
//             Self {
//                 graph,
//                 outputs_cache,
//                 node_id,
//             }
//         }
//         fn evaluate_input(&mut self, name: &str) -> anyhow::Result<MyValueType> {
//             // Calling `evaluate_input` recursively evaluates other nodes in the
//             // graph until the input value for a paramater has been computed.
//             evaluate_input(self.graph, self.node_id, name, self.outputs_cache)
//         }
//         fn populate_output(
//             &mut self,
//             name: &str,
//             value: MyValueType,
//         ) -> anyhow::Result<MyValueType> {
//             // After computing an output, we don't just return it, but we also
//             // populate the outputs cache with it. This ensures the evaluation
//             // only ever computes an output once.
//             //
//             // The return value of the function is the "final" output of the
//             // node, the thing we want to get from the evaluation. The example
//             // would be slightly more contrived when we had multiple output
//             // values, as we would need to choose which of the outputs is the
//             // one we want to return. Other outputs could be used as
//             // intermediate values.
//             //
//             // Note that this is just one possible semantic interpretation of
//             // the graphs, you can come up with your own evaluation semantics!
//             populate_output(self.graph, self.outputs_cache, self.node_id, name, value)
//         }
//         fn input_vector(&mut self, name: &str) -> anyhow::Result<egui::Embedding> {
//             self.evaluate_input(name)?.try_to_Embedding()
//         }
//         fn input_Text(&mut self, name: &str) -> anyhow::Result<f32> {
//             self.evaluate_input(name)?.try_to_Text()
//         }
//         fn output_vector(&mut self, name: &str, value: egui::Embedding) -> anyhow::Result<MyValueType> {
//             self.populate_output(name, MyValueType::Embedding { value })
//         }
//         fn output_Text(&mut self, name: &str, value: f32) -> anyhow::Result<MyValueType> {
//             self.populate_output(name, MyValueType::Text { value })
//         }
//     }

//     let node = &graph[node_id];
//     let mut evaluator = Evaluator::new(graph, outputs_cache, node_id);
//     match node.user_data.template {
//         MyNodeTemplate::AddText => {
//             let a = evaluator.input_Text("A")?;
//             let b = evaluator.input_Text("B")?;
//             evaluator.output_Text("out", a + b)
//         }
//         MyNodeTemplate::SubtractText => {
//             let a = evaluator.input_Text("A")?;
//             let b = evaluator.input_Text("B")?;
//             evaluator.output_Text("out", a - b)
//         }
//         MyNodeTemplate::VectorTimesText => {
//             let Text = evaluator.input_Text("Text")?;
//             let vector = evaluator.input_vector("vector")?;
//             evaluator.output_vector("out", vector * Text)
//         }
//         MyNodeTemplate::AddVector => {
//             let v1 = evaluator.input_vector("v1")?;
//             let v2 = evaluator.input_vector("v2")?;
//             evaluator.output_vector("out", v1 + v2)
//         }
//         MyNodeTemplate::SubtractVector => {
//             let v1 = evaluator.input_vector("v1")?;
//             let v2 = evaluator.input_vector("v2")?;
//             evaluator.output_vector("out", v1 - v2)
//         }
//         MyNodeTemplate::MakeVector => {
//             let x = evaluator.input_Text("x")?;
//             let y = evaluator.input_Text("y")?;
//             evaluator.output_vector("out", egui::Embedding(x, y))
//         }
//         MyNodeTemplate::MakeText => {
//             let value = evaluator.input_Text("value")?;
//             evaluator.output_Text("out", value)
//         }
//     }
// }

// fn populate_output(
//     graph: &MyGraph,
//     outputs_cache: &mut OutputsCache,
//     node_id: NodeId,
//     param_name: &str,
//     value: MyValueType,
// ) -> anyhow::Result<MyValueType> {
//     let output_id = graph[node_id].get_output(param_name)?;
//     outputs_cache.insert(output_id, value);
//     Ok(value)
// }

// // Evaluates the input value of
// fn evaluate_input(
//     graph: &MyGraph,
//     node_id: NodeId,
//     param_name: &str,
//     outputs_cache: &mut OutputsCache,
// ) -> anyhow::Result<MyValueType> {
//     let input_id = graph[node_id].get_input(param_name)?;

//     // The output of another node is connected.
//     if let Some(other_output_id) = graph.connection(input_id) {
//         // The value was already computed due to the evaluation of some other
//         // node. We simply return value from the cache.
//         if let Some(other_value) = outputs_cache.get(&other_output_id) {
//             Ok(*other_value)
//         }
//         // This is the first time encountering this node, so we need to
//         // recursively evaluate it.
//         else {
//             // Calling this will populate the cache
//             evaluate_node(graph, graph[other_output_id].node, outputs_cache)?;

//             // Now that we know the value is cached, return it
//             Ok(*outputs_cache
//                 .get(&other_output_id)
//                 .expect("Cache should be populated"))
//         }
//     }
//     // No existing connection, take the inline value instead.
//     else {
//         Ok(graph[input_id].value)
//     }
// }
