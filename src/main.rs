#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::egui::{DragValue, Visuals};
use eframe::{
    egui::{self, TextEdit},
    epaint::ahash::{HashMap, HashSet},
};
use egui_node_graph::*;
use floneum_plugin::exports::plugins::main::definitions::{
    Embedding, PrimitiveValue, PrimitiveValueType, Value, ValueType,
};
use floneum_plugin::plugins::main::types::{EmbeddingDbId, ModelId};
use floneum_plugin::{Plugin, PluginEngine, PluginInstance};
use log::LevelFilter;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    borrow::Cow,
    fmt::{Debug, Formatter},
    fs::File,
    io::Read,
    io::Write,
    path::PathBuf,
};
use tokio::sync::mpsc::{Receiver, Sender};

fn save_to_file<D: Serialize>(data: D) {
    let mut current_dir = std::env::current_dir().unwrap();
    current_dir.push("save.bin");
    match File::create(current_dir) {
        Ok(mut file) => {
            log::info!("serializing");
            match bincode::serialize(&data) {
                Ok(bytes) => {
                    log::info!("done serializing");
                    let result = file.write_all(&bytes);
                    log::info!("done writing {result:?}");
                }
                Err(err) => {
                    log::error!("{}", err)
                }
            }
        }
        Err(err) => {
            log::error!("{}", err)
        }
    }
}

fn get_from_file<D: DeserializeOwned + Default>() -> D {
    let mut current_dir = std::env::current_dir().unwrap();
    current_dir.push("save.bin");
    if let Ok(mut file) = File::open(current_dir) {
        let mut buffer = Vec::new();

        if file.read_to_end(&mut buffer).is_err() {
            return Default::default();
        }

        if let Ok(from_storage) = bincode::deserialize(&buffer[..]) {
            from_storage
        } else {
            Default::default()
        }
    } else {
        Default::default()
    }
}

#[tokio::main]
async fn main() {
    simple_logger::SimpleLogger::new()
        .with_level(LevelFilter::Off)
        .with_module_level("floneum", LevelFilter::Info)
        .init()
        .unwrap();

    eframe::run_native(
        "Floneum",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(Visuals::dark());
            let app: NodeGraphExample = get_from_file();
            Box::new(app)
        }),
    )
    .expect("Failed to run native example");
}

struct SetOutputMessage {
    node_id: NodeId,
    values: Vec<Value>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct MyNodeData {
    instance: PluginInstance,
    running: bool,
}

#[derive(PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MyDataType {
    Single(MyPrimitiveDataType),
    List(MyPrimitiveDataType),
}

impl From<ValueType> for MyDataType {
    fn from(value: ValueType) -> Self {
        match value {
            ValueType::Single(value) => match value {
                PrimitiveValueType::Number => Self::Single(MyPrimitiveDataType::Number),
                PrimitiveValueType::Text => Self::Single(MyPrimitiveDataType::Text),
                PrimitiveValueType::Embedding => Self::Single(MyPrimitiveDataType::Embedding),
                PrimitiveValueType::Database => Self::Single(MyPrimitiveDataType::Database),
                PrimitiveValueType::Model => Self::Single(MyPrimitiveDataType::Model),
            },
            ValueType::Many(value) => match value {
                PrimitiveValueType::Number => Self::List(MyPrimitiveDataType::Number),
                PrimitiveValueType::Text => Self::List(MyPrimitiveDataType::Text),
                PrimitiveValueType::Embedding => Self::List(MyPrimitiveDataType::Embedding),
                PrimitiveValueType::Database => Self::List(MyPrimitiveDataType::Database),
                PrimitiveValueType::Model => Self::List(MyPrimitiveDataType::Model),
            },
        }
    }
}

#[derive(PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MyPrimitiveDataType {
    Number,
    Text,
    Embedding,
    Model,
    Database,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MyValueType {
    Single(MyPrimitiveValueType),
    List(Vec<MyPrimitiveValueType>),
    Unset,
}

impl MyValueType {
    fn default_of_type(ty: &MyDataType) -> Self {
        match ty {
            MyDataType::Single(value) => match value {
                MyPrimitiveDataType::Text => {
                    Self::Single(MyPrimitiveValueType::Text(String::new()))
                }
                MyPrimitiveDataType::Embedding => {
                    Self::Single(MyPrimitiveValueType::Embedding(Vec::new()))
                }
                MyPrimitiveDataType::Number => Self::Single(MyPrimitiveValueType::Number(0)),
                MyPrimitiveDataType::Model => Self::Single(MyPrimitiveValueType::Model(0)),
                MyPrimitiveDataType::Database => Self::Single(MyPrimitiveValueType::Database(0)),
            },
            MyDataType::List(_) => Self::List(Vec::new()),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MyPrimitiveValueType {
    Number(i64),
    Text(String),
    Embedding(Vec<f32>),
    Model(u32),
    Database(u32),
}

impl From<MyValueType> for Value {
    fn from(value: MyValueType) -> Self {
        match value {
            MyValueType::Single(value) => Self::Single(match value {
                MyPrimitiveValueType::Number(text) => PrimitiveValue::Number(text),
                MyPrimitiveValueType::Text(text) => PrimitiveValue::Text(text),
                MyPrimitiveValueType::Embedding(embedding) => {
                    PrimitiveValue::Embedding(Embedding { vector: embedding })
                }
                MyPrimitiveValueType::Model(id) => PrimitiveValue::Model(ModelId { id }),
                MyPrimitiveValueType::Database(id) => {
                    PrimitiveValue::Database(EmbeddingDbId { id })
                }
            }),
            MyValueType::List(values) => Self::Many(
                values
                    .into_iter()
                    .map(|value| match value {
                        MyPrimitiveValueType::Number(text) => PrimitiveValue::Number(text),
                        MyPrimitiveValueType::Text(text) => PrimitiveValue::Text(text),
                        MyPrimitiveValueType::Embedding(embedding) => {
                            PrimitiveValue::Embedding(Embedding { vector: embedding })
                        }
                        MyPrimitiveValueType::Model(id) => PrimitiveValue::Model(ModelId { id }),
                        MyPrimitiveValueType::Database(id) => {
                            PrimitiveValue::Database(EmbeddingDbId { id })
                        }
                    })
                    .collect(),
            ),
            MyValueType::Unset => todo!(),
        }
    }
}

impl From<Value> for MyValueType {
    fn from(value: Value) -> Self {
        match value {
            Value::Single(value) => Self::Single(match value {
                PrimitiveValue::Number(text) => MyPrimitiveValueType::Number(text),
                PrimitiveValue::Text(text) => MyPrimitiveValueType::Text(text),
                PrimitiveValue::Embedding(embedding) => {
                    MyPrimitiveValueType::Embedding(embedding.vector)
                }
                PrimitiveValue::Model(id) => MyPrimitiveValueType::Model(id.id),
                PrimitiveValue::Database(id) => MyPrimitiveValueType::Database(id.id),
            }),
            Value::Many(values) => Self::List(
                values
                    .into_iter()
                    .map(|value| match value {
                        PrimitiveValue::Number(text) => MyPrimitiveValueType::Number(text),
                        PrimitiveValue::Text(text) => MyPrimitiveValueType::Text(text),
                        PrimitiveValue::Embedding(embedding) => {
                            MyPrimitiveValueType::Embedding(embedding.vector)
                        }
                        PrimitiveValue::Model(id) => MyPrimitiveValueType::Model(id.id),
                        PrimitiveValue::Database(id) => MyPrimitiveValueType::Database(id.id),
                    })
                    .collect(),
            ),
        }
    }
}

impl Default for MyValueType {
    fn default() -> Self {
        Self::Unset
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
pub struct PluginId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MyResponse {
    RunNode(NodeId),
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
pub struct MyGraphState {
    #[serde(skip)]
    pub plugin_engine: PluginEngine,
    pub plugins: slab::Slab<Plugin>,
    pub all_plugins: HashSet<PluginId>,
    pub node_outputs: HashMap<OutputId, MyValueType>,
}

impl MyGraphState {
    fn get_plugin(&self, id: PluginId) -> &Plugin {
        &self.plugins[id.0]
    }
}

impl DataTypeTrait<MyGraphState> for MyDataType {
    fn data_type_color(&self, _user_state: &mut MyGraphState) -> egui::Color32 {
        match self {
            MyDataType::Single(MyPrimitiveDataType::Text) => egui::Color32::from_rgb(38, 109, 211),
            MyDataType::Single(MyPrimitiveDataType::Embedding) => {
                egui::Color32::from_rgb(238, 207, 109)
            }
            MyDataType::Single(MyPrimitiveDataType::Number) => egui::Color32::from_rgb(211, 38, 38),
            MyDataType::Single(MyPrimitiveDataType::Model) => egui::Color32::from_rgb(38, 211, 109),
            MyDataType::Single(MyPrimitiveDataType::Database) => {
                egui::Color32::from_rgb(38, 211, 109)
            }
            MyDataType::List(MyPrimitiveDataType::Text) => egui::Color32::from_rgb(38, 109, 211),
            MyDataType::List(MyPrimitiveDataType::Embedding) => {
                egui::Color32::from_rgb(238, 207, 109)
            }
            MyDataType::List(MyPrimitiveDataType::Number) => egui::Color32::from_rgb(211, 38, 38),
            MyDataType::List(MyPrimitiveDataType::Model) => egui::Color32::from_rgb(38, 211, 109),
            MyDataType::List(MyPrimitiveDataType::Database) => {
                egui::Color32::from_rgb(38, 211, 109)
            }
        }
    }

    fn name(&self) -> Cow<'_, str> {
        match self {
            MyDataType::Single(MyPrimitiveDataType::Text) => Cow::Borrowed("text"),
            MyDataType::Single(MyPrimitiveDataType::Embedding) => Cow::Borrowed("embedding"),
            MyDataType::Single(MyPrimitiveDataType::Number) => Cow::Borrowed("number"),
            MyDataType::Single(MyPrimitiveDataType::Model) => Cow::Borrowed("model"),
            MyDataType::Single(MyPrimitiveDataType::Database) => Cow::Borrowed("database"),
            MyDataType::List(MyPrimitiveDataType::Text) => Cow::Borrowed("list of texts"),
            MyDataType::List(MyPrimitiveDataType::Embedding) => Cow::Borrowed("list of embeddings"),
            MyDataType::List(MyPrimitiveDataType::Number) => Cow::Borrowed("list of numbers"),
            MyDataType::List(MyPrimitiveDataType::Model) => Cow::Borrowed("list of models"),
            MyDataType::List(MyPrimitiveDataType::Database) => Cow::Borrowed("list of databases"),
        }
    }
}

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
            running: false,
            instance: user_state.get_plugin(*self).instance(),
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
            let ty = input.ty.into();
            let value = MyValueType::default_of_type(&ty);
            graph.add_input_param(
                node_id,
                name.to_string(),
                ty,
                value,
                InputParamKind::ConnectionOrConstant,
                true,
            );
        }

        for output in &meta.outputs {
            let name = &output.name;
            let ty: MyDataType = output.ty.into();
            graph.add_output_param(node_id, name.to_string(), ty);
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
        node_id: NodeId,
        ui: &mut egui::Ui,
        _user_state: &mut MyGraphState,
        _node_data: &MyNodeData,
    ) -> Vec<MyResponse> {
        // This trait is used to tell the library which UI to display for the
        // inline parameter widgets.
        egui::ScrollArea::vertical()
            .id_source((node_id, param_name))
            .show(ui, |ui| match self {
                MyValueType::Single(value) => {
                    ui.label(param_name);
                    match value {
                        MyPrimitiveValueType::Text(value) => {
                            ui.add(TextEdit::multiline(value));
                        }
                        MyPrimitiveValueType::Embedding(_) => {
                            ui.label("Embedding");
                        }
                        MyPrimitiveValueType::Model(_) => {
                            ui.label("Model");
                        }
                        MyPrimitiveValueType::Database(_) => {
                            ui.label("Database");
                        }
                        MyPrimitiveValueType::Number(value) => {
                            ui.add(DragValue::new(value));
                        }
                    }
                }
                MyValueType::List(values) => {
                    ui.label(param_name);
                    for value in values {
                        match value {
                            MyPrimitiveValueType::Text(value) => {
                                ui.add(TextEdit::multiline(value));
                            }
                            MyPrimitiveValueType::Embedding(_) => {
                                ui.label("Embedding");
                            }
                            MyPrimitiveValueType::Model(_) => {
                                ui.label("Model");
                            }
                            MyPrimitiveValueType::Database(_) => {
                                ui.label("Database");
                            }
                            MyPrimitiveValueType::Number(value) => {
                                ui.add(DragValue::new(value));
                            }
                        }
                    }
                }
                MyValueType::Unset => {}
            });

        Vec::new()
    }
}

impl UserResponseTrait for MyResponse {}
impl NodeDataTrait for MyNodeData {
    type Response = MyResponse;
    type UserState = MyGraphState;
    type DataType = MyDataType;
    type ValueType = MyValueType;

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
        let node = &graph[node_id];

        if node.user_data.running {
            ui.with_layout(
                egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
                |ui| {
                    ui.add(egui::widgets::Spinner::new());
                },
            );
            return vec![];
        }

        // This allows you to return your responses from the inline widgets.
        let run_button = ui.button("Run");
        if run_button.clicked() {
            return vec![NodeResponse::User(MyResponse::RunNode(node_id))];
        }

        // Render the current output of the node
        let outputs = &node.outputs;

        for (name, id) in outputs {
            let value = user_state.node_outputs.get(id).cloned().unwrap_or_default();
            egui::ScrollArea::vertical()
                .id_source((node_id, name))
                .show(ui, |ui| match &value {
                    MyValueType::Single(single) => match single {
                        MyPrimitiveValueType::Text(value) => {
                            ui.label(value);
                        }
                        MyPrimitiveValueType::Embedding(value) => {
                            ui.label(format!("{:?}", &value[..5]));
                        }
                        MyPrimitiveValueType::Model(id) => {
                            ui.label(format!("Model: {id:?}"));
                        }
                        MyPrimitiveValueType::Database(id) => {
                            ui.label(format!("Database: {id:?}"));
                        }
                        MyPrimitiveValueType::Number(value) => {
                            ui.label(format!("{:02}", value));
                        }
                    },
                    MyValueType::List(many) => {
                        for value in many {
                            match value {
                                MyPrimitiveValueType::Text(value) => {
                                    ui.label(value);
                                }
                                MyPrimitiveValueType::Embedding(value) => {
                                    ui.label(format!("{:?}", &value[..5]));
                                }
                                MyPrimitiveValueType::Model(id) => {
                                    ui.label(format!("Model: {id:?}"));
                                }
                                MyPrimitiveValueType::Database(id) => {
                                    ui.label(format!("Database: {id:?}"));
                                }
                                MyPrimitiveValueType::Number(value) => {
                                    ui.label(format!("{:02}", value));
                                }
                            }
                        }
                    }
                    MyValueType::Unset => {}
                });
        }

        vec![]
    }
}

type MyEditorState = GraphEditorState<MyNodeData, MyDataType, MyValueType, PluginId, MyGraphState>;

#[derive(Serialize, Deserialize)]
pub struct NodeGraphExample {
    state: MyEditorState,

    user_state: MyGraphState,

    search_text: String,

    #[serde(skip)]
    txrx: TxRx,
}

impl Debug for NodeGraphExample {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeGraphExample")
            .field("search_text", &self.search_text)
            .finish()
    }
}

impl Default for NodeGraphExample {
    fn default() -> Self {
        let mut user_state = MyGraphState::default();

        // try to load any default plugins (built in the target/wasm32-unknown-unknown/release directory)
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("target");
        path.push("wasm32-unknown-unknown");
        path.push("release");

        if let Ok(dir) = std::fs::read_dir(&path) {
            for entry in dir.flatten() {
                let path = entry.path();
                if path.extension().unwrap_or_default() == "wasm" {
                    let plugin = user_state.plugin_engine.load_plugin(&path);
                    let id = user_state.plugins.insert(plugin);
                    user_state.all_plugins.insert(PluginId(id));
                }
            }
        }

        Self {
            state: MyEditorState::default(),
            user_state,
            search_text: String::new(),
            txrx: Default::default(),
        }
    }
}

struct TxRx {
    tx: Sender<SetOutputMessage>,
    rx: Receiver<SetOutputMessage>,
}

impl Default for TxRx {
    fn default() -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        Self { tx, rx }
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

        Self {
            state,
            user_state: MyGraphState::default(),
            search_text: String::new(),
            txrx: TxRx::default(),
        }
    }
}

impl eframe::App for NodeGraphExample {
    /// If the persistence function is enabled,
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, _: &mut dyn eframe::Storage) {
        println!("Saving state");
        save_to_file(self);
    }
    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Recieve any async messages about setting node outputs.
        while let Ok(msg) = self.txrx.rx.try_recv() {
            let node = &self.state.graph[msg.node_id].outputs;
            for ((_, id), value) in node.iter().zip(msg.values.into_iter()) {
                self.user_state.node_outputs.insert(*id, value.into());
            }
            self.state.graph[msg.node_id].user_data.running = false;
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                egui::widgets::global_dark_light_mode_switch(ui);
                let response = ui.add(egui::TextEdit::singleline(&mut self.search_text));
                let button = ui.button("Load Plugin at path");
                if button.clicked()
                    || (response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                {
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
                let sender = self.txrx.tx.clone();
                self.state.graph[id].user_data.running = true;

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
