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
            onmounted: move |mount| async move {
                let size = mount.get_client_rect().await.ok();
                node.with_mut(|node| {
                    let pos = node.position;
                    node.outputs[index].write_unchecked().rendered_size = size.map(|mut size| {
                        size.origin += -node.offset();
                        size
                    });
                });
            },
            onmousedown: move |evt| {
                let mut graph: VisualGraph = consume_context();
                let scaled_pos = graph.scale_screen_pos(evt.page_coordinates());
                graph.inner.write().currently_dragging = Some(
                    CurrentlyDragging::Connection(CurrentlyDraggingProps {
                        from: node,
                        from_pos: scaled_pos,
                        index: DraggingIndex::Output(index),
                        to: Signal::new(scaled_pos),
                    }),
                );
                evt.stop_propagation();
            },
            onmouseup: move |evt| {
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
