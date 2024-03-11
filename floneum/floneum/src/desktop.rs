use dioxus::{html::geometry::euclid::Point2D, prelude::*};
use dioxus::desktop::Config;
use dioxus_flow::{Edge, FlowView, Signal, Node, VisualGraph, VisualGraphInner};
use petgraph::Graph;

fn main() {
    const TAILWIND_CSS: &str = include_str!("../public/tailwind.css");
    dioxus::desktop::launch_cfg(
        app,
        Config::new().with_custom_index(
            format!(r#"
    <html>
        <head>
            <title>Flow</title>
            <style>
                html, body {
                    margin: 0;
                    padding: 0;
                    width: 100%;
                    height: 100%;
                }
            </style>
            <style>
            {TAILWIND_CSS}
            </style>
        </head>
        <body>
            <div id="main"></div>
            <script type="module">
                import init from "./pkg/dioxus::desktop.js";
                init();
            </script>
        </body>
    </html>
    "#
            )
            .into(),
        ),
    );
}

fn app() -> Element {
    // Create an undirected graph with `i32` nodes and edges with `()` associated data.
    let mut g: Graph<Signal<Node>, Signal<Edge>> = Graph::new();

    let n1 = g.add_node(Signal::new(Node {
        id: Default::default(),
        position: Point2D::new(0.0, 0.0),
        width: 100.0,
        height: 100.0,
        inputs: 1,
        outputs: 1,
    }));
    g[n1].write().id = n1;
    let n2 = g.add_node(Signal::new(Node {
        id: Default::default(),
        position: Point2D::new(100.0, 100.0),
        width: 100.0,
        height: 100.0,
        inputs: 1,
        outputs: 1,
    }));
    g[n2].write().id = n2;
    let n3 = g.add_node(Signal::new(Node {
        id: Default::default(),
        position: Point2D::new(200.0, 200.0),
        width: 100.0,
        height: 100.0,
        inputs: 1,
        outputs: 1,
    }));
    g[n3].write().id = n3;
    let n4 = g.add_node(Signal::new(Node {
        id: Default::default(),
        position: Point2D::new(300.0, 300.0),
        width: 100.0,
        height: 100.0,
        inputs: 1,
        outputs: 1,
    }));
    g[n4].write().id = n4;

    g.add_edge(n1, n2, Signal::new(Edge { start: 0, end: 0 }));
    g.add_edge(n2, n3, Signal::new(Edge { start: 0, end: 0 }));
    g.add_edge(n3, n4, Signal::new(Edge { start: 0, end: 0 }));

    let visual_graph = VisualGraph {
        inner: Signal::new(VisualGraphInner {
            graph: g,
            currently_dragging: None,
        }),
    };

    rsx! {
        div { width: "100%", height: "100%", FlowView { graph: visual_graph } }
    }
}
