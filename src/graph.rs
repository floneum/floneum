use std::{collections::HashSet, fmt::Debug};

use dioxus::{html::geometry::euclid::Point2D, prelude::*};
use floneum_plugin::{exports::plugins::main::definitions::Input, PluginInstance};
use petgraph::{
    stable_graph::StableGraph,
    visit::{EdgeRef, IntoEdgeReferences, IntoNodeIdentifiers},
};
use serde::{Deserialize, Serialize};

use crate::{
    node_value::{NodeInput, NodeOutput},
    Connection, Edge, LocalSubscription, Node,
};

#[derive(Serialize, Deserialize, Default)]
pub struct VisualGraphInner {
    pub graph: StableGraph<LocalSubscription<Node>, LocalSubscription<Edge>>,
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
    pub node: LocalSubscription<Node>,
}

#[derive(PartialEq, Clone, Serialize, Deserialize)]
pub enum DraggingIndex {
    Input(usize),
    Output(usize),
}

#[derive(Props, PartialEq, Clone, Serialize, Deserialize)]
pub struct CurrentlyDraggingProps {
    pub from: LocalSubscription<Node>,
    pub index: DraggingIndex,
    pub to: LocalSubscription<Point2D<f32, f32>>,
}

#[derive(Props, Clone, Serialize, Deserialize, Default)]
pub struct VisualGraph {
    pub inner: LocalSubscription<VisualGraphInner>,
}

impl VisualGraph {
    pub fn create_node(&self, instance: PluginInstance) {
        let mut inner = self.inner.write();

        let mut inputs = Vec::new();

        for input in &instance.metadata().inputs {
            inputs.push(LocalSubscription::new(NodeInput {
                definition: input.clone(),
                value: input.ty.create(),
            }));
        }

        let mut outputs = Vec::new();

        for output in &instance.metadata().outputs {
            outputs.push(LocalSubscription::new(NodeOutput {
                definition: output.clone(),
                value: output.ty.create_output(),
            }));
        }

        let node = LocalSubscription::new(Node {
            instance,
            position: Point2D::new(0.0, 0.0),
            running: false,
            queued: false,
            error: None,
            id: Default::default(),
            inputs,
            outputs,
            width: 100.0,
            height: 100.0,
        });
        let idx = inner.graph.add_node(node);
        inner.graph[idx].write().id = dbg!(idx);
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

    pub fn start_dragging_node(&self, evt: &MouseData, node: LocalSubscription<Node>) {
        let mut inner = self.inner.write();
        inner.currently_dragging = Some(CurrentlyDragging::Node(NodeDragInfo {
            node,
            element_offset: Point2D::new(
                evt.element_coordinates().x as f32,
                evt.element_coordinates().y as f32,
            ),
        }));
    }

    fn should_run_node(&self, id: petgraph::graph::NodeIndex) -> bool {
        log::info!("Checking if node {id:?} should run");
        let graph = self.inner.read_silent();
        // traverse back through inputs to see if any of those nodes are running
        let mut visited: HashSet<petgraph::stable_graph::NodeIndex> = HashSet::default();
        visited.insert(id);
        let mut should_visit = Vec::new();
        {
            // first add all of the inputs to the current node
            let node = &graph.graph[id].read_silent();
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
            let node = graph.graph[new_id].read_silent();
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

    fn get_node_inputs(&mut self, id: petgraph::graph::NodeIndex) -> Option<Vec<Input>> {
        if !self.should_run_node(id) {
            log::info!(
                "node {:?} has unresolved dependencies, skipping running",
                id
            );
            return None;
        }
        let graph = self.inner.read_silent();

        let mut values: Vec<Input> = Vec::new();
        for input in graph
            .graph
            .edges_directed(id, petgraph::Direction::Incoming)
        {
            match &input.weight().read_silent().value {
                Some(value) => values.push(value.read_silent().clone()),
                None => {
                    log::error!("missing value for output: {:?}", input.id());

                    return None;
                }
            }
        }

        Some(values)
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
    let graph = cx.props.graph.inner.use_(cx);
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
                    cx.props.graph.update_mouse(&**evt);
                },

                current_graph.graph.edge_references().map(|edge_ref|{
                    let edge = current_graph.graph[edge_ref.id()].clone();
                    let start_id = edge_ref.target();
                    let start = current_graph.graph[start_id].clone();
                    let end_id = edge_ref.source();
                    let end = current_graph.graph[end_id].clone();
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
                    let node = current_graph.graph[id].clone();
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
    start: LocalSubscription<Node>,
    connection: LocalSubscription<Edge>,
    end: LocalSubscription<Node>,
}

fn CurrentlyDragging(cx: Scope<CurrentlyDraggingProps>) -> Element {
    let start = cx.props.from.use_(cx);
    let start_pos = match cx.props.index {
        DraggingIndex::Input(index) => start.read().input_pos(index),
        DraggingIndex::Output(index) => start.read().output_pos(index),
    };
    let end = cx.props.to.use_(cx);
    let end_pos = end.read();

    render! { Connection { start_pos: start_pos, end_pos: *end_pos } }
}

fn NodeConnection(cx: Scope<ConnectionProps>) -> Element {
    let start = cx.props.start.use_(cx);
    let connection = cx.props.connection.use_(cx);
    let end = cx.props.end.use_(cx);

    let current_connection = connection.read();
    let start_index = current_connection.start;
    let start = start.read().input_pos(start_index);
    let end_index = current_connection.end;
    let end = end.read().output_pos(end_index);

    render! { Connection { start_pos: start, end_pos: end } }
}
