use dioxus::prelude::*;

use crate::{
    graph::CurrentlyDragging,
    node::{stop_dragging, NODE_KNOB_SIZE},
    CurrentlyDraggingProps, DraggingIndex, Node, VisualGraph,
};

#[component]
pub fn Output(node: Signal<Node>, index: usize) -> Element {
    let current_node = node.read();
    let current_node_id = current_node.id;
    let color = current_node.output_color(index);
    let is_list = current_node.output_is_list(index);

    rsx! {
        button {
            height: "{NODE_KNOB_SIZE}px",
            width: "{NODE_KNOB_SIZE}px",
            border_radius: "50%",
            background_color: "{color}",
            display: "inline-block",
            onmousedown: move |evt| {
                let mut graph: VisualGraph = consume_context();
                let scaled_pos = graph.scale_screen_pos(evt.page_coordinates());
                graph.inner.write().currently_dragging = Some(CurrentlyDragging::Connection(CurrentlyDraggingProps {
                    from: node,
                    from_pos: scaled_pos,
                    index: DraggingIndex::Output(index),
                    to: Signal::new(scaled_pos),
                }));
                evt.stop_propagation();
            },
            onmouseup: move |evt| {
                // Set this as the end of the connection if we're currently dragging and this is the right type of connection
                let mut graph: VisualGraph = consume_context();
                graph.finish_connection(current_node_id, DraggingIndex::Output(index));
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
