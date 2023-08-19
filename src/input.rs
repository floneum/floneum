use dioxus::{html::geometry::euclid::Point2D, prelude::*};
use dioxus_signals::*;
use floneum_plugin::exports::plugins::main::definitions::ValueType;

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
        InputConnection {
            node: cx.props.node,
            index: main_index,
        }
        if let Some(ValueType::Many(_)) = node.input_type(main_index) {
            let inputs_len = node.inputs[index].read().value.len();
            let plus_pos = node.input_pos(Connection { index, ty: crate::edge::ConnectionType::Element(inputs_len) });
            rsx! {
                for element_index in 0..inputs_len {
                    InputConnection {
                        node: cx.props.node,
                        index: Connection { index, ty: crate::edge::ConnectionType::Element(element_index) },
                    }
                }
                path {
                    d: "M {plus_pos.x as f64 + NODE_KNOB_SIZE + NODE_MARGIN} {plus_pos.y} h-5v-5h-4v5h-5v4h5v5h4v-5h5z",
                    stroke: "black",
                    onclick: move |_| {
                        let node = cx.props.node.read();
                        node.inputs[index].write().push_default_value();
                    },
                }
                path {
                    d: "M {plus_pos.x as f64 + (NODE_KNOB_SIZE * 2.) + NODE_MARGIN} {plus_pos.y} h 14 v 4 h -14 Z",
                    stroke: "black",
                    onclick: move |_| {
                        let node = cx.props.node.read();
                        node.inputs[index].write().pop_value();
                    },
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

    render! {
        circle {
            cx: pos.x as f64 + NODE_KNOB_SIZE + NODE_MARGIN,
            cy: pos.y as f64,
            r: NODE_KNOB_SIZE,
            fill: "{color}",
            onmousedown: move |evt| {
                let graph: VisualGraph = cx.consume_context().unwrap();
                graph.inner.write().currently_dragging = Some(CurrentlyDragging::Connection(CurrentlyDraggingProps {
                    from: cx.props.node,
                    index: DraggingIndex::Input(index),
                    to: Signal::new(Point2D::new(evt.page_coordinates().x as f32, evt.page_coordinates().y as f32)),
                }));
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
