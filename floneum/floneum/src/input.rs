use dioxus::prelude::*;
use dioxus_signals::*;
use floneum_plugin::plugins::main::types::ValueType;

use crate::{
    edge::Connection,
    graph::CurrentlyDragging,
    node::{NODE_KNOB_SIZE, NODE_MARGIN},
    CurrentlyDraggingProps, DraggingIndex, Node, VisualGraph,
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
                // rsx! {
                //     path {
                //         d: "M {plus_pos.x + 12.} {plus_pos.y} h-3v-3h-2v3h-3v2h3v3h2v-3h3z",
                //         stroke: "black",
                //         onmousedown: move |_| {
                //             let node = props.node.read();
                //             node.inputs[index].write().push_default_value();
                //         },
                //     }
                //     path {
                //         d: "M {minus_pos.x} {minus_pos.y} h 8 v 2 h -8 Z",
                //         stroke: "black",
                //         onmousedown: move |_| {
                //             let node = props.node.read();
                //             node.inputs[index].write().pop_value();
                //         },
                //     }
                //     rect {
                //         x: box_x as f64,
                //         y: box_y as f64,
                //         width: box_width,
                //         height: box_height,
                //         stroke: "black",
                //         fill: "transparent",
                //         stroke_width: 1,
                //     }
                //     for element_index in 0..inputs_len {
                //         InputConnection {
                //             node: props.node,
                //             index: Connection { index, ty: crate::edge::ConnectionType::Element(element_index) },
                //         }
                //     }
                // }
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
    let pos = current_node.input_pos(index);
    let color = current_node.input_color(index);
    let is_list = current_node.input_is_list(index);

    rsx! {
        circle {
            cx: pos.x as f64 + NODE_KNOB_SIZE + NODE_MARGIN,
            cy: pos.y as f64,
            r: NODE_KNOB_SIZE,
            fill: "{color}",
            onmousedown: move |evt| {
                let graph: VisualGraph = consume_context();
                let new_connection = Some(CurrentlyDragging::Connection(CurrentlyDraggingProps {
                    from: node,
                    index: DraggingIndex::Input(index),
                    from_pos: graph.scale_screen_pos(evt.page_coordinates()),
                    to: Signal::new(graph.scale_screen_pos(evt.page_coordinates())),
                }));
                graph.inner.write().currently_dragging = new_connection;
            },
            onmouseup: move |_| {
                // Set this as the end of the connection if we're currently dragging and this is the right type of connection
                let graph: VisualGraph = consume_context();
                graph.finish_connection(current_node_id, DraggingIndex::Input(index));
            },
            onmousemove: move |evt| {
                let graph: VisualGraph = consume_context();
                graph.update_mouse(&evt);
            },
        }
        if is_list {
            circle {
                cx: pos.x as f64 + NODE_KNOB_SIZE + NODE_MARGIN,
                cy: pos.y as f64,
                r: NODE_KNOB_SIZE / 2.0,
                fill: "black",
                onmousedown: move |evt| {
                    let graph: VisualGraph = consume_context();
                    let new_connection = Some(CurrentlyDragging::Connection(CurrentlyDraggingProps {
                        from: node,
                        index: DraggingIndex::Input(index),
                        from_pos: graph.scale_screen_pos(evt.page_coordinates()),
                        to: Signal::new(graph.scale_screen_pos(evt.page_coordinates())),
                    }));
                    graph.inner.write().currently_dragging = new_connection;
                },
                onmouseup: move |_| {
                    // Set this as the end of the connection if we're currently dragging and this is the right type of connection
                    let graph: VisualGraph = consume_context();
                    graph.finish_connection(current_node_id, DraggingIndex::Input(index));
                },
                onmousemove: move |evt| {
                    let graph: VisualGraph = consume_context();
                    graph.update_mouse(&evt);
                },
            }
        }
    }
}
