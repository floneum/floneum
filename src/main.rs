#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::{
    egui::{self, TextEdit},
    epaint::ahash::{HashMap, HashSet},
};
use egui_node_graph::*;
use plugin::exports::plugins::main::definitions::{Embedding, Value, ValueType, PrimitiveValueType, PrimitiveValue};
use plugin::{Plugin, PluginEngine, PluginInstance};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, path::PathBuf};
use tokio::sync::mpsc::{Receiver, Sender};

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

struct SetOutputMessage {
    node_id: NodeId,
    values: Vec<Value>,
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
    Single(MyPrimitiveDataType),
    List(MyPrimitiveDataType),
}

#[derive(PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MyPrimitiveDataType {
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
    Single(MyPrimitiveValueType),
    List(Vec<MyPrimitiveValueType>),
    Unset,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MyPrimitiveValueType {
    Text(String),
    Embedding(Vec<f32>),
}

impl Into<Value> for MyValueType {
    fn into(self) -> Value {
        match self {
            Self::Single(value) => Value::Single(match value {
                MyPrimitiveValueType::Text(text) => PrimitiveValue::Text(text),
                MyPrimitiveValueType::Embedding(embedding) => PrimitiveValue::Embedding(Embedding { vector: embedding }),
            }),
            Self::List(values) => Value::Many(values.into_iter().map(|value| match value {
                MyPrimitiveValueType::Text(text) => PrimitiveValue::Text(text),
                MyPrimitiveValueType::Embedding(embedding) => PrimitiveValue::Embedding(Embedding { vector: embedding }),
            }).collect()),
            _ => todo!(),
        }
    }
}

impl From<Value> for MyValueType {
    fn from(value: Value) -> Self {
        match value {
            Value::Single(value) => match value {
                PrimitiveValue::Text(text) => Self::Single(MyPrimitiveValueType::Text(text)),
                PrimitiveValue::Embedding(embedding) => Self::Single(MyPrimitiveValueType::Embedding(embedding.vector)),
            },
            Value::Many(values) => Self::List(values.into_iter().map(|value| match value {
                PrimitiveValue::Text(text) => MyPrimitiveValueType::Text(text),
                PrimitiveValue::Embedding(embedding) => MyPrimitiveValueType::Embedding(embedding.vector),
            }).collect()),
        }
    }
}

impl Default for MyValueType {
    fn default() -> Self {
        // NOTE: This is just a dummy `Default` implementation. The library
        // requires it to circumvent some internal borrow checker issues.
        Self::Unset
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
    RunNode(NodeId),
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
    #[serde(skip)]
    pub node_outputs: HashMap<OutputId, MyValueType>,
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
            MyDataType::Single(MyPrimitiveDataType::Text) => egui::Color32::from_rgb(38, 109, 211),
            MyDataType::Single(MyPrimitiveDataType::Embedding) => egui::Color32::from_rgb(238, 207, 109),
            MyDataType::List(MyPrimitiveDataType::Text) => egui::Color32::from_rgb(38, 109, 211),
            MyDataType::List(MyPrimitiveDataType::Embedding) => egui::Color32::from_rgb(238, 207, 109),
        }
    }

    fn name(&self) -> Cow<'_, str> {
        match self {
            MyDataType::Single(MyPrimitiveDataType::Text) => Cow::Borrowed("text"),
            MyDataType::Single(MyPrimitiveDataType::Embedding) => Cow::Borrowed("embedding"),
            MyDataType::List(MyPrimitiveDataType::Text) => Cow::Borrowed("list of texts"),
            MyDataType::List(MyPrimitiveDataType::Embedding) => Cow::Borrowed("list of embeddings"),
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

        let node = &graph[node_id];

        let meta = node.user_data.instance.metadata().clone();

        for input in &meta.inputs {
            let name = &input.name;
            match &input.ty {
                ValueType::Single(ty) => match ty {
                    PrimitiveValueType::Text => graph.add_input_param(
                        node_id,
                        name.to_string(),
                        MyDataType::Single(MyPrimitiveDataType::Text),
                        MyValueType::Single(MyPrimitiveValueType::Text(String::new())),
                        InputParamKind::ConnectionOrConstant,
                        true,
                    ),
                    PrimitiveValueType::Embedding => graph.add_input_param(
                        node_id,
                        name.to_string(),
                        MyDataType::Single(MyPrimitiveDataType::Embedding),
                        MyValueType::Single(MyPrimitiveValueType::Embedding(Vec::new())),
                        InputParamKind::ConnectionOrConstant,
                        true,
                    ),
                },
                ValueType::Many(ty) => match ty {
                    PrimitiveValueType::Text => graph.add_input_param(
                        node_id,
                        name.to_string(),
                        MyDataType::List(MyPrimitiveDataType::Text),
                        MyValueType::List(vec![MyPrimitiveValueType::Text(String::new())]),
                        InputParamKind::ConnectionOrConstant,
                        true,
                    ),
                    PrimitiveValueType::Embedding => graph.add_input_param(
                        node_id,
                        name.to_string(),
                        MyDataType::List(MyPrimitiveDataType::Embedding),
                        MyValueType::List(vec![MyPrimitiveValueType::Embedding(Vec::new())]),
                        InputParamKind::ConnectionOrConstant,
                        true,
                    ),
                },
            };
        }

        for output in &meta.outputs {
            let name = &output.name;
            let ty =match &output.ty {
                ValueType::Many(ty) => match ty{
                    PrimitiveValueType::Text => MyDataType::List(MyPrimitiveDataType::Text),
                    PrimitiveValueType::Embedding => MyDataType::List(MyPrimitiveDataType::Embedding),
                }
                ValueType::Single(ty) => match ty{
                    PrimitiveValueType::Text => MyDataType::Single(MyPrimitiveDataType::Text),
                    PrimitiveValueType::Embedding => MyDataType::Single(MyPrimitiveDataType::Embedding),
                }
            };
            graph.add_output_param(node_id, name.to_string(),ty);
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
            MyValueType::Single(value) => {
                ui.label(param_name);
                match value{

                    MyPrimitiveValueType::Text(value) => {
                        ui.horizontal(|ui| {
                            ui.add(TextEdit::multiline(value));
                        });
                    }
                    MyPrimitiveValueType::Embedding(_) => {
                        ui.horizontal(|ui| {
                            ui.label("Embedding")
                        });
                    }
                }
            }
            MyValueType::List(values) => {
                ui.horizontal(|ui| {
                    ui.label(param_name);
                    for value in values {
                        match value{

                            MyPrimitiveValueType::Text(value) => {
                                ui.add(TextEdit::multiline(value));
                            }
                            MyPrimitiveValueType::Embedding(_) => {
                                ui.label("Embedding");
                            }
                        }
                    }
                });
            }
            MyValueType::Unset => {}
        }

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
        ui: &mut egui::Ui,
        node_id: NodeId,
        graph: &Graph<MyNodeData, MyDataType, MyValueType>,
        user_state: &mut Self::UserState,
    ) -> Vec<NodeResponse<MyResponse, MyNodeData>>
    where
        MyResponse: UserResponseTrait,
    {
        // This logic is entirely up to the user. In this case, we check if the
        // current node we're drawing is the active one, by comparing against
        // the value stored in the global user state, and draw different button
        // UIs based on that.

        // This allows you to return your responses from the inline widgets.
        let run_button = ui.button("Run");
        if run_button.clicked() {
            return vec![NodeResponse::User(MyResponse::RunNode(node_id))];
        }

        // Render the current output of the node
        let outputs = &graph[node_id].outputs;

        for (_, id) in outputs {
            let value = user_state.node_outputs.get(id).cloned().unwrap_or_default();
            ui.horizontal(|ui| {
                match &value {
                MyValueType::Single(single) => {
                    match single{
                        MyPrimitiveValueType::Text(value) => {
                            ui.label(value);
                        }
                        MyPrimitiveValueType::Embedding(value) => {
                            ui.label(format!("{:?}", &value[..5]));
                        }
                    }
                }
                MyValueType::List(many) => {
                    for value in many {
                        match value {
                            MyPrimitiveValueType::Text(value) => {
                                ui.label(value);
                            }
                            MyPrimitiveValueType::Embedding(value) => {
                                ui.label(format!("{:?}", &value[..5]));
                            }
                        }
                    }
                }
                MyValueType::Unset => {}
            }
        });
        }

        vec![]
    }
}

type MyEditorState = GraphEditorState<MyNodeData, MyDataType, MyValueType, PluginId, MyGraphState>;

pub struct NodeGraphExample {
    state: MyEditorState,

    user_state: MyGraphState,

    search_text: String,

    rx: Receiver<SetOutputMessage>,

    tx: Sender<SetOutputMessage>,
}

impl Default for NodeGraphExample {
    fn default() -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        Self {
            state: MyEditorState::default(),
            user_state: MyGraphState::default(),
            search_text: String::new(),
            rx,
            tx,
        }
    }
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
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        Self {
            state,
            user_state: MyGraphState::default(),
            search_text: String::new(),
            rx,
            tx,
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
        // Recieve any async messages about setting node outputs.
        while let Ok(msg) = self.rx.try_recv() {
            let node = &self.state.graph[msg.node_id].outputs;
            for ((_, id), value) in node.iter().zip(msg.values.into_iter()) {
                self.user_state.node_outputs.insert(*id, value.into());
            }
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                egui::widgets::global_dark_light_mode_switch(ui);
                let response = ui.add(egui::TextEdit::singleline(&mut self.search_text));
                if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    let path = PathBuf::from(&self.search_text);
                    if path.exists() {
                        let plugin = self.user_state.plugin_engine.load_plugin(&path);
                        let id = self.user_state.plugins.insert(plugin);
                        self.user_state.all_plugins.insert(PluginId(id));
                    }
                }
            });
        });

        let graph_response = egui::CentralPanel::default()
            .show(ctx, |ui| {
                self.state.draw_graph_editor(
                    ui,
                    AllMyNodeTemplates(self.user_state.all_plugins.iter().copied().collect()),
                    &mut self.user_state,
                    Vec::default(),
                )
            })
            .inner;

        'o: for responce in graph_response.node_responses {
            if let NodeResponse::User(MyResponse::RunNode(id)) = responce {
                let node = &self.state.graph[id];

                let mut values: Vec<Value> = Vec::new();
                for (_, id) in &node.inputs {
                    let input = self.state.graph.get_input(*id);
                    let connection = self.state.graph.connections.get(input.id);
                    let value = match connection {
                        Some(&connection) => {
                            let connection = self.state.graph.get_output(connection);
                            let output_id = connection.id;
                            if let Some(value) = self.user_state.node_outputs.get(&output_id) {
                                value
                            } else {
                                continue 'o;
                            }
                        }
                        None => &input.value,
                    };
                    match &value {
                        MyValueType::Unset => continue 'o,
                        _ => values.push(value.clone().into()),
                    }
                }

                let fut = node.user_data.instance.run(values);
                let sender = self.tx.clone();

                tokio::spawn(async move {
                    let outputs = fut.await;

                    let _ = sender
                        .send(SetOutputMessage {
                            node_id: id,
                            values: outputs,
                        })
                        .await;
                });
            }
        }
    }
}
