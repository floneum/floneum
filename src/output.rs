use dioxus::{html::geometry::euclid::Point2D, prelude::*};
use dioxus_signals::*;

use crate::{
    edge::Connection,
    graph::CurrentlyDragging,
    node::{NODE_KNOB_SIZE, NODE_MARGIN},
    CurrentlyDraggingProps, DraggingIndex, Node, VisualGraph,
};

#[inline_props]
pub fn Output(cx: Scope, node: Signal<Node>, index: Connection) -> Element {
    let index = *index;
    let current_node = node.read();
    let current_node_id = current_node.id;
    let pos = current_node.output_pos(index);
    let color = current_node.output_color(index);

    render! {
        circle {
            cx: pos.x as f64 - NODE_KNOB_SIZE - NODE_MARGIN,
            cy: pos.y as f64,
            r: NODE_KNOB_SIZE,
            fill: "{color}",
            onmousedown: move |evt| {
                let graph: VisualGraph = cx.consume_context().unwrap();
                graph.inner.write().currently_dragging = Some(CurrentlyDragging::Connection(CurrentlyDraggingProps {
                    from: cx.props.node.clone(),
                    index: DraggingIndex::Output(index),
                    to: Signal::new(Point2D::new(evt.page_coordinates().x as f32, evt.page_coordinates().y as f32)),
                }));
            },
            onmouseup: move |_| {
                // Set this as the end of the connection if we're currently dragging and this is the right type of connection
                let graph: VisualGraph = cx.consume_context().unwrap();
                graph.finish_connection(current_node_id, DraggingIndex::Output(index));
            },
            onmousemove: move |evt| {
                let graph: VisualGraph = cx.consume_context().unwrap();
                graph.update_mouse(&**evt);
            },
        }
    }
}
