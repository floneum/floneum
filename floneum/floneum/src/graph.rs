use std::{collections::HashSet, fmt::Debug};

use dioxus::{
    html::geometry::{euclid::Point2D, PagePoint},
    prelude::{SvgAttributes, *},
};
use floneum_plugin::PluginInstance;
use petgraph::{
    stable_graph::StableGraph,
    visit::{EdgeRef, IntoEdgeReferences, IntoNodeIdentifiers},
};
use serde::{Deserialize, Serialize};
use slab::Slab;

use crate::{
    node_value::{NodeInput, NodeOutput},
    Connection, Edge, Node, Signal,
};

pub struct VisualGraphInner {
    pub graph: StableGraph<Signal<Node>, Signal<Edge>>,
    pub connections: Slab<ConnectionProps>,
    pub currently_dragging: Option<CurrentlyDragging>,
    pub pan_pos: Point2D<f32, f32>,
    pub zoom: f32,
}

impl Default for VisualGraphInner {
    fn default() -> Self {
        Self {
            graph: StableGraph::default(),
            connections: Slab::default(),
            currently_dragging: None,
            pan_pos: Point2D::new(0.0, 0.0),
            zoom: 1.0,
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
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

#[derive(PartialEq, Clone, Copy)]
pub struct NodeDragInfo {
    pub element_offset: Point2D<f32, f32>,
    pub node: Signal<Node>,
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum DraggingIndex {
    Input(crate::edge::Connection),
    Output(usize),
}

#[derive(Props, PartialEq, Clone, Copy)]
pub struct CurrentlyDraggingProps {
    pub from: Signal<Node>,
    pub index: DraggingIndex,
    pub from_pos: Point2D<f32, f32>,
    pub to: Signal<Point2D<f32, f32>>,
}

#[derive(Props, Clone, Copy, Default, PartialEq)]
pub struct VisualGraph {
    pub inner: Signal<VisualGraphInner>,
}

impl VisualGraph {
    pub fn create_node(&self, instance: PluginInstance) {
        let position = self.scale_screen_pos(PagePoint::new(0., 0.));
        let mut inner_mut = self.inner;
        let mut inner = inner_mut.write();

        let mut inputs = Vec::new();

        for input in &instance.metadata().inputs {
            inputs.push(Signal::new_in_scope(
                NodeInput::new(
                    input.clone(),
                    vec![input
                        .ty
                        .create()?
                        ],
                ),
                self.inner.origin_scope(),
            ));
        }

        let mut outputs = Vec::new();

        for output in &instance.metadata().outputs {
            outputs.push(Signal::new_in_scope(
                NodeOutput {
                    definition: output.clone(),
                    value: output.ty.create(),
                },
                self.inner.origin_scope(),
            ));
        }

        let node = Signal::new_in_scope(
            Node {
                instance,
                position,
                running: false,
                queued: false,
                error: None,
                id: Default::default(),
                inputs,
                outputs,
            },
            ScopeId::ROOT,
        );
        let idx = inner.graph.add_node(node);
        inner.graph[idx].write().id = idx;
    }

    pub fn scale_screen_pos(&self, pos: PagePoint) -> Point2D<f32, f32> {
        let graph = self.inner.read();
        let mut pos = Point2D::new(pos.x as f32, pos.y as f32);
        pos.x -= graph.pan_pos.x;
        pos.y -= graph.pan_pos.y;
        pos.x /= graph.zoom;
        pos.y /= graph.zoom;
        pos
    }

    pub fn clear_dragging(&mut self) {
        self.inner.write().currently_dragging = None;
    }

    pub fn update_mouse(&mut self, evt: &MouseData) {
        let new_pos = self.scale_screen_pos(evt.page_coordinates());
        let mut inner = self.inner.write();
        match &mut inner.currently_dragging {
            Some(CurrentlyDragging::Connection(current_graph_dragging)) => {
                let mut to = current_graph_dragging.to.write();
                *to = new_pos;
            }
            Some(CurrentlyDragging::Node(current_graph_dragging)) => {
                let mut node = current_graph_dragging.node.write();
                node.position.x = new_pos.x - current_graph_dragging.element_offset.x;
                node.position.y = new_pos.y - current_graph_dragging.element_offset.y;
            }
            _ => {}
        }
    }

    pub fn start_dragging_node(&mut self, evt: &MouseData, node: Signal<Node>) {
        let mut inner = self.inner.write();
        inner.currently_dragging = Some(CurrentlyDragging::Node(NodeDragInfo {
            element_offset: evt.element_coordinates().cast().cast_unit(),
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
            let mut input = inputs[end_index.index];
            let mut input = input.write();
            input.set_connection(end_index.ty, value);
        }

        true
    }

    pub fn run_node(&self, mut node: Signal<Node>) {
        let current_node_id = {
            let current = node.read();
            current.id
        };
        if self.set_input_nodes(current_node_id) {
            let inputs = {
                let mut current_node = node.write();
                current_node.running = true;
                current_node.queued = true;
                current_node
                    .inputs
                    .iter()
                    .map(|input| input.read().value())
                    .collect()
            };
            log::info!(
                "Running node {:?} with inputs {:?}",
                current_node_id,
                inputs
            );

            let graph = self.inner;
            spawn(async move {
                let mut current_node_write = node.write();
                let fut = current_node_write.instance.run(inputs);
                let result = { fut.await };
                match result.as_deref() {
                    Some(Ok(result)) => {
                        for (out, current) in result.iter().zip(current_node_write.outputs.iter()) {
                            current.write_unchecked().value = out.clone();
                        }

                        let current_graph = graph.read();
                        for edge in current_graph
                            .graph
                            .edges_directed(current_node_id, petgraph::Direction::Outgoing)
                        {
                            let new_node_id = edge.target();
                            let mut node = current_graph.graph[new_node_id];
                            node.write().queued = true;
                        }
                    }
                    Some(Err(err)) => {
                        log::error!("Error running node {:?}: {:?}", current_node_id, err);
                        current_node_write.error = Some(err.to_string());
                    }
                    None => {}
                }
                current_node_write.running = false;
                current_node_write.queued = false;
            });
        }
    }

    pub fn check_connection_validity(
        &self,
        input_id: petgraph::graph::NodeIndex,
        output_id: petgraph::graph::NodeIndex,
        edge: Signal<Edge>,
    ) -> bool {
        let edge = edge.read();
        let graph = self.inner.read();
        let input = graph.graph[input_id]
            .read()
            .output_type(edge.start)
            .unwrap();
        let output = graph.graph[output_id].read().input_type(edge.end).unwrap();
        input.compatible(&output)
    }

    pub fn connect(
        &mut self,
        input_id: petgraph::graph::NodeIndex,
        output_id: petgraph::graph::NodeIndex,
        edge: Signal<Edge>,
    ) {
        if !self.check_connection_validity(input_id, output_id, edge) {
            return;
        }
        let mut current_graph = self.inner.write();
        // remove any existing connections to this input
        let mut edges_to_remove = Vec::new();
        {
            let input_index = edge.read().end;
            for edge in current_graph
                .graph
                .edges_directed(output_id, petgraph::Direction::Incoming)
            {
                if edge.weight().read().end == input_index {
                    edges_to_remove.push(edge.id());
                }
            }
            for edge in edges_to_remove {
                current_graph.graph.remove_edge(edge);
            }
        }
        current_graph.graph.add_edge(input_id, output_id, edge);
    }

    pub(crate) fn finish_connection(
        &mut self,
        node_id: petgraph::graph::NodeIndex,
        index: DraggingIndex,
    ) {
        let current_graph = self.inner.read();
        if let Some(CurrentlyDragging::Connection(currently_dragging)) =
            &current_graph.currently_dragging
        {
            let currently_dragging_id = {
                let from = currently_dragging.from.read();
                from.id
            };
            let ((output_id, output_index), (input_id, input_index)) =
                match (index, currently_dragging.index) {
                    (DraggingIndex::Input(input), DraggingIndex::Output(output)) => {
                        ((node_id, input), (currently_dragging_id, output))
                    }
                    (DraggingIndex::Output(output), DraggingIndex::Input(input)) => {
                        ((currently_dragging_id, input), (node_id, output))
                    }
                    _ => return,
                };
            drop(current_graph);
            let edge = Signal::new(Edge::new(input_index, output_index));
            self.connect(input_id, output_id, edge);
        } else {
            drop(current_graph);
        }
        self.inner.write().currently_dragging = None;
    }
}

#[derive(Props, PartialEq, Clone)]
pub struct FlowViewProps {
    graph: VisualGraph,
}

pub fn FlowView(mut props: FlowViewProps) -> Element {
    use_context_provider(|| props.graph);
    let mut graph = props.graph.inner;
    let current_graph = graph.read();
    let current_graph_dragging = current_graph.currently_dragging;
    let mut drag_start_pos = use_signal(|| Option::<Point2D<f32, f32>>::None);
    let mut drag_pan_pos = use_signal(|| Option::<Point2D<f32, f32>>::None);
    let pan_pos = current_graph.pan_pos;
    let zoom = current_graph.zoom;
    let mut transform_matrix = [1., 0., 0., 1., 0., 0.];
    for i in &mut transform_matrix {
        *i *= zoom;
    }
    transform_matrix[4] = pan_pos.x;
    transform_matrix[5] = pan_pos.y;

    let transform = format!(
        "matrix({} {} {} {} {} {})",
        transform_matrix[0],
        transform_matrix[1],
        transform_matrix[2],
        transform_matrix[3],
        transform_matrix[4],
        transform_matrix[5]
    );

    rsx! {
        div { position: "relative",
            style: "-webkit-user-select: none; -ms-user-select: none; user-select: none;",
            width: "100%",
            height: "100%",
            onmousemove: move |evt| props.graph.update_mouse(&evt),
            div {
                position: "absolute",
                top: "0",
                left: "0",
                class: "border-b-2 border-r-2 rounded-br-md p-2",
                button {
                    class: "m-1",
                    onclick: move |_| {
                        let new_zoom = zoom * 1.1;
                        graph.with_mut(|graph| {
                            graph.zoom = new_zoom;
                        });
                    },
                    "+"
                }
                button {
                    class: "m-1",
                    onclick: move |_| {
                        let new_zoom = zoom * 0.9;
                        graph.with_mut(|graph| {
                            graph.zoom = new_zoom;
                        });
                    },
                    "-"
                }
            }

            for id in current_graph.graph.node_identifiers() {
                Node {
                    key: "{id:?}",
                    node: current_graph.graph[id],
                }
            }

            svg {
                width: "100%",
                height: "100%",
                onmouseenter: move |data| {
                    if data.held_buttons().is_empty() {
                        props.graph.clear_dragging();
                    }
                },
                onmousedown: move |evt| {
                    let pos = evt.element_coordinates();
                    drag_start_pos.set(Some(Point2D::new(pos.x as f32, pos.y as f32)));
                    drag_pan_pos.set(Some(pan_pos));
                },
                onmouseup: move |_| {
                    drag_start_pos.set(None);
                    props.graph.clear_dragging();
                },
                onmousemove: move |evt| {
                    if let (Some(drag_start_pos), Some(drag_pan_pos)) = (drag_start_pos(), drag_pan_pos()) {
                        let pos = evt.element_coordinates();
                        let end_pos = Point2D::new(pos.x as f32, pos.y as f32);
                        let diff = end_pos - drag_start_pos;
                        graph.with_mut(|graph| {
                            graph.pan_pos.x = drag_pan_pos.x + diff.x;
                            graph.pan_pos.y = drag_pan_pos.y + diff.y;
                        });
                    }
                    props.graph.update_mouse(&evt);
                },

                g {
                    transform: "{transform}",
                    for edge_ref in current_graph.graph.edge_references() {
                        NodeConnection {
                            key: "{edge_ref.id():?}",
                            start: current_graph.graph[edge_ref.target()],
                            connection: current_graph.graph[edge_ref.id()],
                            end: current_graph.graph[edge_ref.source()],
                        }
                    }

                    if let Some(CurrentlyDragging::Connection(current_graph_dragging)) = &current_graph_dragging {
                        CurrentlyDragging {
                            from_pos: current_graph_dragging.from_pos,
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

#[derive(Clone, Props, PartialEq)]
pub struct ConnectionProps {
    start: Signal<Node>,
    connection: Signal<Edge>,
    end: Signal<Node>,
}

fn CurrentlyDragging(props: CurrentlyDraggingProps) -> Element {
    let start = props.from;
    let current_start = start.read();
    let start_pos = props.from_pos;
    let color;
    match props.index {
        DraggingIndex::Input(index) => {
            color = current_start.input_color(index);
        }
        DraggingIndex::Output(index) => {
            color = current_start.output_color(index);
        }
    };
    let end = props.to;
    let end_pos = end.read();

    rsx! { Connection { start_pos: start_pos, end_pos: *end_pos, color: color } }
}

fn NodeConnection(props: ConnectionProps) -> Element {
    let start = props.start;
    let connection = props.connection;
    let end = props.end;

    let current_connection = connection.read();
    let start_index = current_connection.end;
    let start_node = start.read();
    // let start = start_node.input_pos(start_index);
    // let end_index = current_connection.start;
    // let end = end.read().output_pos(end_index);

    // let ty = start_node.input_type(start_index).unwrap();
    // let color = ty.color();

    // rsx! { Connection { start_pos: start, end_pos: end, color: color } }
    None
}
