use dioxus::{html::geometry::euclid::Point2D, prelude::*};
use dioxus_free_icons::Icon;
use floneum_plugin::exports::plugins::main::definitions::ValueType;
use floneum_plugin::PluginInstance;
use petgraph::{graph::NodeIndex, stable_graph::DefaultIx};
use serde::{Deserialize, Serialize};

use crate::graph::CurrentlyDragging;
use crate::node_value::{NodeInput, NodeOutput};
use crate::{use_application_state, Colored, CurrentlyDraggingProps, DraggingIndex, Edge};
use crate::{Point, VisualGraph};
use dioxus_signals::*;

const SNAP_DISTANCE: f32 = 15.;
const NODE_KNOB_SIZE: f64 = 5.;
const NODE_MARGIN: f64 = 2.;

#[derive(Serialize, Deserialize)]
pub struct Node {
    pub instance: PluginInstance,
    #[serde(skip)]
    pub running: bool,
    #[serde(skip)]
    pub queued: bool,
    #[serde(skip)]
    pub error: Option<String>,
    pub id: NodeIndex<DefaultIx>,
    pub position: Point,
    pub inputs: Vec<Signal<NodeInput>>,
    pub outputs: Vec<Signal<NodeOutput>>,
    pub width: f32,
    pub height: f32,
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Node {
    pub fn center(&self) -> Point2D<f32, f32> {
        (Point2D::new(self.position.x, self.position.y)
            - Point2D::new(self.width, self.height) / 2.)
            .to_point()
    }

    pub fn input_pos(&self, index: usize) -> Point2D<f32, f32> {
        Point2D::new(
            self.position.x - 1.,
            self.position.y + ((index as f32 + 1.) * self.height / (self.inputs.len() as f32 + 1.)),
        )
    }

    pub fn output_pos(&self, index: usize) -> Point2D<f32, f32> {
        Point2D::new(
            self.position.x + self.width - 1.,
            self.position.y
                + ((index as f32 + 1.) * self.height / (self.outputs.len() as f32 + 1.)),
        )
    }

    pub fn output_type(&self, index: usize) -> Option<ValueType> {
        self.outputs
            .get(index)
            .map(|output| output.read().definition.ty)
    }

    pub fn output_color(&self, index: usize) -> String {
        match self.output_type(index) {
            Some(ty) => ty.color(),
            None => "black".to_string(),
        }
    }

    pub fn input_type(&self, index: usize) -> Option<ValueType> {
        self.inputs
            .get(index)
            .map(|input| input.read().definition.ty)
    }

    pub fn input_color(&self, index: usize) -> String {
        match self.input_type(index) {
            Some(ty) => ty.color(),
            None => "black".to_string(),
        }
    }

    pub fn help_text(&self) -> String {
        self.instance.metadata().description.to_string()
    }
}

#[derive(Props, PartialEq)]
pub struct NodeProps {
    node: Signal<Node>,
}

pub fn Node(cx: Scope<NodeProps>) -> Element {
    let application = use_application_state(cx);
    let node = cx.props.node;
    let current_node = node.read();
    let current_node_id = current_node.id;
    let width = current_node.width;
    let height = current_node.height;
    let pos = current_node.position - Point::new(1., 0.);

    render! {
        // inputs
        (0..current_node.inputs.len()).map(|i| {
            let pos = current_node.input_pos(i);
            let color = current_node.input_color(i);
            rsx! {
                circle {
                    cx: pos.x as f64 + NODE_KNOB_SIZE + NODE_MARGIN,
                    cy: pos.y as f64,
                    r: NODE_KNOB_SIZE,
                    fill: "{color}",
                    onmousedown: move |evt| {
                        let graph: VisualGraph = cx.consume_context().unwrap();
                        graph.inner.write().currently_dragging = Some(CurrentlyDragging::Connection(CurrentlyDraggingProps {
                            from: cx.props.node.clone(),
                            index: DraggingIndex::Input(i),
                            to: Signal::new(Point2D::new(evt.page_coordinates().x as f32, evt.page_coordinates().y as f32)),
                        }));
                    },
                    onmouseup: move |_| {
                        // Set this as the end of the connection if we're currently dragging and this is the right type of connection
                        let graph: VisualGraph = cx.consume_context().unwrap();
                        let mut current_graph = graph.inner.write();
                        if let Some(CurrentlyDragging::Connection(currently_dragging)) = &current_graph.currently_dragging {
                            let start_index = match currently_dragging.index {
                                DraggingIndex::Output(index) => index,
                                _ => return,
                            };
                            let start_node = currently_dragging.from.read();
                            let start_id = start_node.id;
                            let ty = start_node.output_type(start_index).unwrap();
                            drop(start_node);
                            let edge = Signal::new(Edge::new(
                                start_index,
                                i,
                                ty,
                            ));
                            current_graph.graph.add_edge(start_id, current_node_id, edge);
                        }
                        graph.clear_dragging();
                    },
                    onmousemove: move |evt| {
                        let graph: VisualGraph = cx.consume_context().unwrap();
                        graph.update_mouse(&**evt);
                    },
                }
            }
        }),

        // center UI/Configuration
        foreignObject {
            x: "{pos.x}",
            y: "{pos.y}",
            width: width as f64,
            height: height as f64,
            onmousedown: move |evt| {
                let graph: VisualGraph = cx.consume_context().unwrap();
                {
                    let node = node.read();
                    if let Some((index, dist))
                        = (0..node.inputs.len())
                            .map(|i| {
                                let input_pos = node.input_pos(i);
                                (
                                    DraggingIndex::Input(i),
                                    (input_pos.x - evt.page_coordinates().x as f32).powi(2)
                                        + (input_pos.y - evt.page_coordinates().y as f32).powi(2),
                                )
                            })
                            .chain(
                                (0..node.outputs.len())
                                    .map(|i| {
                                        let output_pos = node.output_pos(i);
                                        (
                                            DraggingIndex::Output(i),
                                            (output_pos.x - evt.page_coordinates().x as f32).powi(2)
                                                + (output_pos.y - evt.page_coordinates().y as f32).powi(2),
                                        )
                                    }),
                            )
                            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    {
                        if dist < SNAP_DISTANCE.powi(2) {
                            let mut current_graph = graph.inner.write();
                            current_graph
                                .currently_dragging = Some(
                                CurrentlyDragging::Connection(CurrentlyDraggingProps {
                                    from: cx.props.node.clone(),
                                    index,
                                    to: Signal::new(
                                        Point2D::new(
                                            evt.page_coordinates().x as f32,
                                            evt.page_coordinates().y as f32,
                                        ),
                                    ),
                                }),
                            );
                        } else {
                            graph.start_dragging_node(&*evt, cx.props.node.clone());
                        }
                    } else {
                        graph.start_dragging_node(&*evt, cx.props.node.clone());
                    }
                }
            },
            onmousemove: |evt| {
                let graph: VisualGraph = cx.consume_context().unwrap();
                graph.update_mouse(&**evt);
            },
            onmouseup: move |evt| {
                let graph: VisualGraph = cx.consume_context().unwrap();
                {
                    let mut current_graph = graph.inner.write();
                    if let Some(CurrentlyDragging::Connection(currently_dragging))
                        = &current_graph.currently_dragging
                    {
                        let dist;
                        let edge;
                        let start_id;
                        let end_id;
                        match currently_dragging.index {
                            DraggingIndex::Output(start_index) => {
                                let node = node.read();
                                let combined = (0..node.inputs.len())
                                    .map(|i| {
                                        let input_pos = node.input_pos(i);
                                        (
                                            i,
                                            (input_pos.x - evt.page_coordinates().x as f32).powi(2)
                                                + (input_pos.y - evt.page_coordinates().y as f32).powi(2),
                                        )
                                    })
                                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                    .unwrap();
                                let input_idx = combined.0;
                                dist = combined.1;
                                let start_node = currently_dragging.from.read();
                                start_id = start_node.id;
                                end_id = current_node_id;
                                let ty = start_node.output_type(start_index).unwrap();
                                edge = Signal::new(Edge::new(start_index, input_idx, ty));
                            }
                            DraggingIndex::Input(start_index) => {
                                let node = node.read();
                                let combined = (0..node.outputs.len())
                                    .map(|i| {
                                        let output_pos = node.output_pos(i);
                                        (
                                            i,
                                            (output_pos.x - evt.page_coordinates().x as f32).powi(2)
                                                + (output_pos.y - evt.page_coordinates().y as f32).powi(2),
                                        )
                                    })
                                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                    .unwrap();
                                let output_idx = combined.0;
                                dist = combined.1;
                                let start_node = currently_dragging.from.read();
                                end_id = start_node.id;
                                start_id = current_node_id;
                                let ty = start_node.output_type(output_idx).unwrap();
                                edge = Signal::new(Edge::new(output_idx, start_index, ty));
                            }
                        }
                        if dist < SNAP_DISTANCE.powi(2) {
                            current_graph.graph.add_edge(start_id, end_id, edge);
                        }
                    }
                }
                graph.clear_dragging();

                // Focus or unfocus this node
                let mut application = application.write();
                match &application.currently_focused {
                    Some(currently_focused_node) if currently_focused_node == &cx.props.node => {
                        application.currently_focused = None;
                    }
                    _ => {
                        application.currently_focused = Some(cx.props.node.clone());
                    }
                }
            },

            CenterNodeUI {
                node: cx.props.node.clone(),
            }
        }

        // outputs
        (0..current_node.outputs.len()).map(|i| {
            let pos = current_node.output_pos(i);
            let color = current_node.output_color(i);
            let ty = current_node.output_type(i).unwrap();
            rsx! {
                circle {
                    cx: pos.x as f64 - NODE_KNOB_SIZE - NODE_MARGIN,
                    cy: pos.y as f64,
                    r: NODE_KNOB_SIZE,
                    fill: "{color}",
                    onmousedown: move |evt| {
                        let graph: VisualGraph = cx.consume_context().unwrap();
                        graph.inner.write().currently_dragging = Some(CurrentlyDragging::Connection(CurrentlyDraggingProps {
                            from: cx.props.node.clone(),
                            index: DraggingIndex::Output(i),
                            to: Signal::new(Point2D::new(evt.page_coordinates().x as f32, evt.page_coordinates().y as f32)),
                        }));
                    },
                    onmouseup: move |_| {
                        // Set this as the end of the connection if we're currently dragging and this is the right type of connection
                        let graph: VisualGraph = cx.consume_context().unwrap();
                        {
                            let mut current_graph = graph.inner.write();
                            if let Some(CurrentlyDragging::Connection(currently_dragging)) = &current_graph.currently_dragging {
                                let start_index = match currently_dragging.index {
                                    DraggingIndex::Input(index) => index,
                                    _ => return,
                                };
                                let start_id = currently_dragging.from.read().id;
                                let edge = Signal::new(Edge::new(i, start_index, ty));
                                current_graph.graph.add_edge(current_node_id, start_id, edge);
                            }
                        }
                        graph.clear_dragging();
                    },
                    onmousemove: move |evt| {
                        let graph: VisualGraph = cx.consume_context().unwrap();
                        graph.update_mouse(&**evt);
                    },
                }
            }
        })
    }
}

fn CenterNodeUI(cx: Scope<NodeProps>) -> Element {
    let application = use_application_state(cx);
    let focused = &application.read().currently_focused == &Some(cx.props.node.clone());
    let node = cx.props.node;
    let current_node = node.read();
    let current_node_id = current_node.id;
    let name = &current_node.instance.metadata().name;
    let focused_class = if focused {
        "border-2 border-blue-500"
    } else {
        ""
    };

    render! {
        div {
            style: "-webkit-user-select: none; -ms-user-select: none; user-select: none; padding: {NODE_KNOB_SIZE*2.+2.}px;",
            class: "flex flex-col justify-center items-center w-full h-full border rounded-md overflow-scroll {focused_class}",
            div {
                button {
                    class: "fixed p-2 top-0 right-0",
                    onclick: move |_| {
                        application.write().remove(node.read().id)
                    },
                    Icon {
                        width: 15,
                        height: 15,
                        fill: "black",
                        icon: dioxus_free_icons::icons::io_icons::IoTrashOutline,
                    }
                }
                h1 {
                    class: "text-md",
                    "{name}"
                }
                if current_node.running {
                    rsx! { div { "Loading..." } }
                }
                else {
                    rsx! {
                        button {
                            class: "p-1 border rounded-md hover:bg-gray-200",
                            onclick: move |_| {
                                if application.read().graph.set_input_nodes(current_node_id) {
                                    let mut current_node = cx.props.node.write();
                                    let inputs = current_node.inputs.iter().map(|input| input.read().value.clone()).collect();
                                    log::trace!("Running node {:?} with inputs {:?}", current_node_id, inputs);
                                    current_node.running = true;
                                    current_node.queued = true;

                                    let fut = current_node.instance.run(inputs);
                                    let node = cx.props.node.clone();
                                    cx.spawn(async move {
                                        match fut.await.as_deref() {
                                            Some(Ok(result)) => {
                                                let current_node = node.read();
                                                for (out, current) in result.iter().zip(current_node.outputs.iter()) {
                                                    current.write().value = out.clone();
                                                }
                                            }
                                            Some(Err(err)) => {
                                                log::error!("Error running node {:?}: {:?}", current_node_id, err);
                                                let mut node_mut = node.write();
                                                node_mut.error = Some(err.to_string());
                                            }
                                            None => {}
                                        }
                                        let mut current_node = node.write();
                                        current_node.running = false;
                                        current_node.queued = false;
                                    });
                                }
                            },
                            "Run"
                        }
                    }
                }
                div { color: "red",
                    if let Some(error) = &current_node.error {
                        rsx! {
                            p { "{error}" }
                        }
                    }
                }
            }
        }
    }
}
