#![allow(non_snake_case)]

use dioxus::html::geometry::euclid::Point2D;

mod node;
pub use node::Node;
mod local_sub;
pub use local_sub::{LocalSubscription, UseLocalSubscription};
mod edge;
pub use edge::Edge;
mod graph;
pub use graph::{CurrentlyDraggingProps, DraggingIndex, FlowView, VisualGraph, VisualGraphInner};
mod connection;
pub use connection::Connection;

pub type Point = Point2D<f32, f32>;
