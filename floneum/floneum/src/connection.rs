use dioxus::html::geometry::euclid::Point2D;
use dioxus::prelude::*;

#[component]
pub fn Connection(
    start_pos: Point2D<f32, f32>,
    end_pos: Point2D<f32, f32>,
    color: String,
) -> Element {
    let offset = (end_pos.x - start_pos.x) / 2.0;

    rsx! {
        path {
            d: "M{start_pos.x},{start_pos.y} C{start_pos.x + offset},{start_pos.y} {end_pos.x - offset},{end_pos.y} {end_pos.x},{end_pos.y}",
            fill: "none",
            stroke: "{color}",
            stroke_width: "2",
            pointer_events: "none"
        }
    }
}
