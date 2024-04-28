use dioxus::prelude::*;
use floneum_plugin::plugins::main::types::ValueType;

use crate::{
    edge::Connection, graph::CurrentlyDragging, node::NODE_KNOB_SIZE, CurrentlyDraggingProps,
    DraggingIndex, Node, VisualGraph,
};

#[component]
pub fn Input(node: Signal<Node>, index: usize) -> Element {
    let main_index = Connection {
        index,
        ty: crate::edge::ConnectionType::Single,
    };
    let node_read = node.read();

    rsx! {
        if let Some(ValueType::Many(_)) = node_read.input_type(main_index) {
            if !node_read.inputs[index].read().value.is_empty() {
                svg {
                    path {
                        d: "M 0 0 h-3v-3h-2v3h-3v2h3v3h2v-3h3z",
                        stroke: "black",
                        onmousedown: move |_| {
                            let node = node.read();
                            node.inputs[index].write_unchecked().push_default_value().throw();
                        },
                    }
                }
                svg {
                    path {
                        d: "M 0 0 h 8 v 2 h -8 Z",
                        stroke: "black",
                        onmousedown: move |_| {
                            let node = node.read();
                            node.inputs[index].write_unchecked().pop_value();
                        },
                    }
                }
                for element_index in 0..node_read.inputs[index].read().value.len() {
                    InputConnection {
                        node,
                        index: Connection { index, ty: crate::edge::ConnectionType::Element(element_index) },
                    }
                }
            }
        }
        else {
            InputConnection {
                node,
                index: main_index,
            }
        }
    }
}

#[component]
pub fn InputConnection(node: Signal<Node>, index: Connection) -> Element {
    let current_node = node.read();
    let current_node_id = current_node.id;
    let color = current_node.input_color(index);
    let is_list = current_node.input_is_list(index);

    rsx! {
        button {
            width: "{NODE_KNOB_SIZE}px",
            height: "{NODE_KNOB_SIZE}px",
            border_radius: "50%",
            background_color: "{color}",
            display: "inline-block",
            onmounted: move |mount| async move {
                let size = mount.get_client_rect().await.ok();
                node.with_mut(|node| node.inputs[index.index].write_unchecked().rendered_size = size);
            },
            onmousedown: move |evt| {
                let mut graph: VisualGraph = consume_context();
                let new_connection = Some(CurrentlyDragging::Connection(CurrentlyDraggingProps {
                    from: node,
                    index: DraggingIndex::Input(index),
                    from_pos: graph.scale_screen_pos(evt.page_coordinates()),
                    to: Signal::new(graph.scale_screen_pos(evt.page_coordinates())),
                }));
                graph.inner.write().currently_dragging = new_connection;
                evt.stop_propagation();
            },
            onmouseup: move |evt| {
                // Set this as the end of the connection if we're currently dragging and this is the right type of connection
                let mut graph: VisualGraph = consume_context();
                graph.finish_connection(current_node_id, DraggingIndex::Input(index));
                evt.stop_propagation();
            },
            onmousemove: move |evt| {
                let mut graph: VisualGraph = consume_context();
                graph.update_mouse(&evt);
            },
            if is_list {
                "+"
            }
        }
    }
}
