use std::{collections::HashSet, fmt::Debug};

use dioxus::{html::geometry::euclid::Point2D, prelude::*};
use floneum_plugin::PluginInstance;
use petgraph::{
    stable_graph::StableGraph,
    visit::{EdgeRef, IntoEdgeReferences, IntoNodeIdentifiers},
};
use serde::{Deserialize, Serialize};

use crate::{
    node_value::{NodeInput, NodeOutput},
    Colored, Connection, Edge, Node, Signal,
};

#[derive(Serialize, Deserialize, Default)]
pub struct VisualGraphInner {
    pub graph: StableGraph<Signal<Node>, Signal<Edge>>,
    pub currently_dragging: Option<CurrentlyDragging>,
}

#[derive(PartialEq, Clone, Serialize, Deserialize)]
pub enum CurrentlyDragging {
    Node(NodeDragInfo),
    Connection(CurrentlyDraggingProps),
}

impl Debug for CurrentlyDragging {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CurrentlyDragging::Node(_) => write!(f, "Node"),
            CurrentlyDragging::Connection(_) => write!(f, "Connection"),
        }
    }
}

#[derive(PartialEq, Clone, Serialize, Deserialize)]
pub struct NodeDragInfo {
    pub element_offset: Point2D<f32, f32>,
    pub node: Signal<Node>,
}

#[derive(PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum DraggingIndex {
    Input(crate::edge::Connection),
    Output(usize),
}

#[derive(Props, PartialEq, Clone, Serialize, Deserialize)]
pub struct CurrentlyDraggingProps {
    pub from: Signal<Node>,
    pub index: DraggingIndex,
    pub to: Signal<Point2D<f32, f32>>,
}

#[derive(Props, Clone, Serialize, Deserialize, Default)]
pub struct VisualGraph {
    pub inner: Signal<VisualGraphInner>,
}

impl VisualGraph {
    pub fn create_node(&self, instance: PluginInstance) {
        let mut inner = self.inner.write();

        let mut inputs = Vec::new();

        for input in &instance.metadata().inputs {
            inputs.push(Signal::new(NodeInput::new(
                input.clone(),
                input.ty.create(),
            )));
        }

        let mut outputs = Vec::new();

        for output in &instance.metadata().outputs {
            outputs.push(Signal::new(NodeOutput {
                definition: output.clone(),
                value: output.ty.create_output(),
            }));
        }

        let node = Signal::new(Node {
            instance,
            position: Point2D::new(0.0, 0.0),
            running: false,
            queued: false,
            error: None,
            id: Default::default(),
            inputs,
            outputs,
            width: 120.0,
            height: 120.0,
        });
        let idx = inner.graph.add_node(node);
        inner.graph[idx].write().id = idx;
    }

    pub fn clear_dragging(&self) {
        self.inner.write().currently_dragging = None;
    }

    pub fn update_mouse(&self, evt: &MouseData) {
        let mut inner = self.inner.write();
        match &mut inner.currently_dragging {
            Some(CurrentlyDragging::Connection(current_graph_dragging)) => {
                let mut to = current_graph_dragging.to.write();
                to.x = evt.page_coordinates().x as f32;
                to.y = evt.page_coordinates().y as f32;
            }
            Some(CurrentlyDragging::Node(current_graph_dragging)) => {
                let mut node = current_graph_dragging.node.write();
                node.position.x =
                    evt.page_coordinates().x as f32 - current_graph_dragging.element_offset.x;
                node.position.y =
                    evt.page_coordinates().y as f32 - current_graph_dragging.element_offset.y;
            }
            _ => {}
        }
    }

    pub fn start_dragging_node(&self, _evt: &MouseData, node: Signal<Node>) {
        let mut inner = self.inner.write();
        inner.currently_dragging = Some(CurrentlyDragging::Node(NodeDragInfo {
            element_offset: {
                let current_node = node.read();
                Point2D::new(current_node.height / 2.0, current_node.width / 4.0)
            },
            node,
        }));
    }

    fn should_run_node(&self, id: petgraph::graph::NodeIndex) -> bool {
        log::info!("Checking if node {id:?} should run");
        let graph = self.inner.read();
        // traverse back through inputs to see if any of those nodes are running
        let mut visited: HashSet<petgraph::stable_graph::NodeIndex> = HashSet::default();
        visited.insert(id);
        let mut should_visit = Vec::new();
        {
            // first add all of the inputs to the current node
            let node = &graph.graph[id].read();
            if node.running {
                log::info!("Node {id:?} is running, so we shouldn't run it again");
                return false;
            }

            for input in graph
                .graph
                .edges_directed(id, petgraph::Direction::Incoming)
            {
                let source = input.source();
                should_visit.push(source);
                visited.insert(source);
            }
        }

        while let Some(new_id) = should_visit.pop() {
            if new_id == id {
                continue;
            }
            let node = graph.graph[new_id].read();
            if node.running || node.queued {
                log::info!("Node {new_id:?} is running... we should wait until it's done");
                return false;
            }
            for input in graph
                .graph
                .edges_directed(id, petgraph::Direction::Incoming)
            {
                let source = input.source();
                if !visited.contains(&source) {
                    should_visit.push(source);
                    visited.insert(source);
                }
            }
        }

        true
    }

    pub fn set_input_nodes(&self, id: petgraph::graph::NodeIndex) -> bool {
        if !self.should_run_node(id) {
            log::info!(
                "node {:?} has unresolved dependencies, skipping running",
                id
            );
            return false;
        }
        let graph = self.inner.read();

        let inputs = &graph.graph[id].read().inputs;
        for input in graph
            .graph
            .edges_directed(id, petgraph::Direction::Incoming)
        {
            let source = input.source();
            let edge = input.weight().read();
            let start_index = edge.start;
            let end_index = edge.end;
            let input = graph.graph[source].read();
            let value = input.outputs[start_index].read().as_input();
            if let Some(value) = value {
                let input = inputs[end_index.index];
                let mut input = input.write();
                input.set_connection(end_index.ty, value);
            }
        }

        true
    }

    pub(crate) fn finish_connection(
        &self,
        node_id: petgraph::graph::NodeIndex,
        index: DraggingIndex,
    ) {
        let mut current_graph = self.inner.write();
        if let Some(CurrentlyDragging::Connection(currently_dragging)) =
            &current_graph.currently_dragging
        {
            let currently_dragging_id = currently_dragging.from.read().id;
            let ((input_id, input_index), (output_id, output_index)) =
                match (index, currently_dragging.index) {
                    (DraggingIndex::Input(input), DraggingIndex::Output(output)) => {
                        ((node_id, input), (currently_dragging_id, output))
                    }
                    (DraggingIndex::Output(output), DraggingIndex::Input(input)) => {
                        ((currently_dragging_id, input), (node_id, output))
                    }
                    _ => return,
                };
            let start_node = current_graph.graph[input_id].read();
            let ty = start_node.output_type(output_index).unwrap();
            drop(start_node);
            let edge = Signal::new(Edge::new(output_index, input_index, ty));
            current_graph.graph.add_edge(output_id, input_id, edge);
        }
        current_graph.currently_dragging = None;
    }
}

impl PartialEq for VisualGraph {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

#[derive(Props, PartialEq)]
pub struct FlowViewProps {
    graph: VisualGraph,
}

pub fn FlowView(cx: Scope<FlowViewProps>) -> Element {
    use_context_provider(cx, || cx.props.graph.clone());
    let graph = cx.props.graph.inner;
    let current_graph = graph.read();
    let current_graph_dragging = current_graph.currently_dragging.clone();

    render! {
        div { position: "relative",
            width: "100%",
            height: "100%",
            svg {
                width: "100%",
                height: "100%",
                onmouseenter: move |data| {
                    if data.held_buttons().is_empty() {
                        cx.props.graph.clear_dragging();
                    }
                },
                onmouseup: move |_| {
                    cx.props.graph.clear_dragging();
                },
                onmousemove: move |evt| {
                    cx.props.graph.update_mouse(&evt);
                },

                current_graph.graph.edge_references().map(|edge_ref|{
                    let edge = current_graph.graph[edge_ref.id()];
                    let start_id = edge_ref.target();
                    let start = current_graph.graph[start_id];
                    let end_id = edge_ref.source();
                    let end = current_graph.graph[end_id];
                    rsx! {
                        NodeConnection {
                            key: "{edge_ref.id():?}",
                            start: start,
                            connection: edge,
                            end: end,
                        }
                    }
                }),
                current_graph.graph.node_identifiers().map(|id|{
                    let node = current_graph.graph[id];
                    rsx! {
                        Node {
                            key: "{id:?}",
                            node: node,
                        }
                    }
                }),

                if let Some(CurrentlyDragging::Connection(current_graph_dragging)) = &current_graph_dragging {
                    let current_graph_dragging = current_graph_dragging.clone();
                    rsx! {
                        CurrentlyDragging {
                            from: current_graph_dragging.from,
                            index: current_graph_dragging.index,
                            to: current_graph_dragging.to,
                        }
                    }
                }
            }
        }
    }
}

#[derive(Props, PartialEq)]
struct ConnectionProps {
    start: Signal<Node>,
    connection: Signal<Edge>,
    end: Signal<Node>,
}

fn CurrentlyDragging(cx: Scope<CurrentlyDraggingProps>) -> Element {
    let start = cx.props.from;
    let current_start = start.read();
    let start_pos;
    let color;
    match cx.props.index {
        DraggingIndex::Input(index) => {
            color = current_start.input_color(index);
            start_pos = current_start.input_pos(index);
        }
        DraggingIndex::Output(index) => {
            color = current_start.output_color(index);
            start_pos = current_start.output_pos(index);
        }
    };
    let end = cx.props.to;
    let end_pos = end.read();

    render! { Connection { start_pos: start_pos, end_pos: *end_pos, color: color } }
}

fn NodeConnection(cx: Scope<ConnectionProps>) -> Element {
    let start = cx.props.start;
    let connection = cx.props.connection;
    let end = cx.props.end;

    let current_connection = connection.read();
    let start_index = current_connection.end;
    let start_node = start.read();
    let start = start_node.input_pos(start_index);
    let end_index = current_connection.start;
    let end = end.read().output_pos(end_index);

    let ty = start_node.input_type(start_index).unwrap();
    let color = ty.color();

    render! { Connection { start_pos: start, end_pos: end, color: color } }
}
