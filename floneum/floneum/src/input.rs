use dioxus::prelude::*;
use dioxus_signals::*;
use floneum_plugin::plugins::main::types::ValueType;

use crate::{
    edge::Connection,
    graph::CurrentlyDragging,
    node::{NODE_KNOB_SIZE, NODE_MARGIN},
    CurrentlyDraggingProps, DraggingIndex, Node, VisualGraph,
};

#[inline_props]
pub fn Input(cx: Scope, node: Signal<Node>, index: usize) -> Element {
    let index = *index;
    let main_index = Connection {
        index,
        ty: crate::edge::ConnectionType::Single,
    };
    let node = node.read();

    render! {
        if let Some(ValueType::Many(_)) = node.input_type(main_index) {
            let inputs_len = node.inputs[index].read().value.len();

            if inputs_len == 0 {
                return None;
            }

            let plus_pos = node.input_array_add_element_pos(index);
            let minus_pos = node.input_array_remove_element_pos(index);

            let first_pos = node.input_pos(Connection { index, ty: crate::edge::ConnectionType::Element(0) });
            let last_pos = node.input_pos(Connection { index, ty: crate::edge::ConnectionType::Element(inputs_len - 1) });
            let box_width = {
                (last_pos.x as f64 - plus_pos.x as f64) + NODE_KNOB_SIZE * 2. + NODE_MARGIN * 2.
            };
            let box_height = {
                last_pos.y as f64 - first_pos.y as f64 + NODE_KNOB_SIZE * 2. + NODE_MARGIN * 2.
            };
            let box_x = plus_pos.x;
            let box_y = first_pos.y - 1. - NODE_MARGIN as f32 - NODE_KNOB_SIZE as f32;

            rsx! {
                path {
                    d: "M {plus_pos.x + 12.} {plus_pos.y} h-3v-3h-2v3h-3v2h3v3h2v-3h3z",
                    stroke: "black",
                    onmousedown: move |_| {
                        let node = cx.props.node.read();
                        node.inputs[index].write().push_default_value();
                    },
                }
                path {
                    d: "M {minus_pos.x} {minus_pos.y} h 8 v 2 h -8 Z",
                    stroke: "black",
                    onmousedown: move |_| {
                        let node = cx.props.node.read();
                        node.inputs[index].write().pop_value();
                    },
                }
                rect {
                    x: box_x as f64,
                    y: box_y as f64,
                    width: box_width,
                    height: box_height,
                    stroke: "black",
                    fill: "transparent",
                    stroke_width: 1,
                }
                for element_index in 0..inputs_len {
                    InputConnection {
                        node: cx.props.node,
                        index: Connection { index, ty: crate::edge::ConnectionType::Element(element_index) },
                    }
                }
            }
        }
        else {
            rsx! {
                InputConnection {
                    node: cx.props.node,
                    index: main_index,
                }
            }
        }
    }
}

#[inline_props]
pub fn InputConnection(cx: Scope, node: Signal<Node>, index: Connection) -> Element {
    let index = *index;
    let current_node = node.read();
    let current_node_id = current_node.id;
    let pos = current_node.input_pos(index);
    let color = current_node.input_color(index);
    let is_list = current_node.input_is_list(index);

    render! {
        circle {
            cx: pos.x as f64 + NODE_KNOB_SIZE + NODE_MARGIN,
            cy: pos.y as f64,
            r: NODE_KNOB_SIZE,
            fill: "{color}",
            onmousedown: move |evt| {
                let graph: VisualGraph = cx.consume_context().unwrap();
                let new_connection = Some(CurrentlyDragging::Connection(CurrentlyDraggingProps {
                    from: cx.props.node,
                    index: DraggingIndex::Input(index),
                    to: Signal::new(graph.scale_screen_pos(evt.page_coordinates())),
                }));
                graph.inner.write().currently_dragging = new_connection;
            },
            onmouseup: move |_| {
                // Set this as the end of the connection if we're currently dragging and this is the right type of connection
                let graph: VisualGraph = cx.consume_context().unwrap();
                graph.finish_connection(current_node_id, DraggingIndex::Input(index));
            },
            onmousemove: move |evt| {
                let graph: VisualGraph = cx.consume_context().unwrap();
                graph.update_mouse(&evt);
            },
        }
        if is_list {
            rsx! {
                circle {
                    cx: pos.x as f64 + NODE_KNOB_SIZE + NODE_MARGIN,
                    cy: pos.y as f64,
                    r: NODE_KNOB_SIZE / 2.0,
                    fill: "black",
                    onmousedown: move |evt| {
                        let graph: VisualGraph = cx.consume_context().unwrap();
                        let new_connection = Some(CurrentlyDragging::Connection(CurrentlyDraggingProps {
                            from: cx.props.node,
                            index: DraggingIndex::Input(index),
                            to: Signal::new(graph.scale_screen_pos(evt.page_coordinates())),
                        }));
                        graph.inner.write().currently_dragging = new_connection;
                    },
                    onmouseup: move |_| {
                        // Set this as the end of the connection if we're currently dragging and this is the right type of connection
                        let graph: VisualGraph = cx.consume_context().unwrap();
                        graph.finish_connection(current_node_id, DraggingIndex::Input(index));
                    },
                    onmousemove: move |evt| {
                        let graph: VisualGraph = cx.consume_context().unwrap();
                        graph.update_mouse(&evt);
                    },
                }
            }
        }
    }
}
