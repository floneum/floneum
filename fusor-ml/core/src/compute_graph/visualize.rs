use rustc_hash::FxHashMap;

use super::queue::ComputeQueue;
use super::{
    NodeIndex, ComputeGraphInner, ComputeGraphNodeVariant, layout_pass,
};
use tabbycat::Graph;
use tabbycat::{Edge, GraphBuilder, GraphType, Identity, Stmt, StmtList};

#[derive(Default)]
struct GraphVisPass {
    queued: ComputeQueue,
    layout_pass: layout_pass::LayoutPass,
    identities: FxHashMap<NodeIndex, Identity>,
    statements: Vec<Stmt>,
}

impl GraphVisPass {
    fn visit_element_wise(&mut self, key: NodeIndex, graph: &ComputeGraphInner) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::ElementWise(op) => op,
            _ => panic!("Expected ElementWise node"),
        };
        let input = self.identities.get(&operation.value).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!(
            "{} ({}) #{:?}",
            operation.name(),
            output_layout,
            key
        ));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key, id.clone());
    }

    fn visit_pair_wise(&mut self, key: NodeIndex, graph: &ComputeGraphInner) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::PairWise(op) => op,
            _ => panic!("Expected PairWise node"),
        };
        let first = self.identities.get(&operation.first).unwrap();
        let second = self.identities.get(&operation.second).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!(
            "{} ({}) #{:?}",
            operation.function.name(),
            output_layout,
            key
        ));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(first.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.statements.push(Stmt::Edge(
            Edge::head_node(second.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key, id.clone());
    }

    fn visit_mat_mul(&mut self, key: NodeIndex, graph: &ComputeGraphInner) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::MatMul(op) => op,
            _ => panic!("Expected MatMul node"),
        };
        let first = self.identities.get(&operation.first).unwrap();
        let second = self.identities.get(&operation.second).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("matmul ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(first.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.statements.push(Stmt::Edge(
            Edge::head_node(second.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key, id.clone());
    }

    fn visit_q_mat_mul(&mut self, key: NodeIndex, graph: &ComputeGraphInner) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::QMatMul(op) => op,
            _ => panic!("Expected QMatMul node"),
        };
        let input = self.identities.get(&operation.input).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("qmatmul ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key, id.clone());
    }

    fn visit_reduce(&mut self, key: NodeIndex, graph: &ComputeGraphInner) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::Reduce(op) => op,
            _ => panic!("Expected Reduce node"),
        };
        let input = self.identities.get(&operation.value).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!(
            "{} ({}) #{:?}",
            operation.function.name(),
            output_layout,
            key
        ));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key, id.clone());
    }

    fn visit_map_layout(&mut self, key: NodeIndex, graph: &ComputeGraphInner) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::MapLayout(op) => op,
            _ => panic!("Expected MapLayout node"),
        };
        let input = self.identities.get(&operation.input).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("map_layout ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key, id.clone());
    }

    fn visit_resize(&mut self, key: NodeIndex, graph: &ComputeGraphInner) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::Resize(op) => op,
            _ => panic!("Expected Resize node"),
        };
        let input = self.identities.get(&operation.input).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("resize ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key, id.clone());
    }

    fn visit_slice_assign(&mut self, key: NodeIndex, graph: &ComputeGraphInner) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::SliceAssign(op) => op,
            _ => panic!("Expected SliceAssign node"),
        };
        let input = self.identities.get(&operation.input).unwrap();
        let value = self.identities.get(&operation.value).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("slice_assign ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.statements.push(Stmt::Edge(
            Edge::head_node(value.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key, id.clone());
    }

    fn visit_index_select(&mut self, key: NodeIndex, graph: &ComputeGraphInner) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::IndexSelect(op) => op,
            _ => panic!("Expected IndexSelect node"),
        };
        let input = self.identities.get(&operation.input).unwrap();
        let value = self.identities.get(&operation.indexes).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("index_select ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.statements.push(Stmt::Edge(
            Edge::head_node(value.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key, id.clone());
    }

    fn visit_dequantize(&mut self, key: NodeIndex, _: &ComputeGraphInner) {
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("dequantize ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.identities.insert(key, id.clone());
    }

    fn visit_tensor(&mut self, key: NodeIndex, _: &ComputeGraphInner) {
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("tensor ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.identities.insert(key, id.clone());
    }

    fn visit_custom(&mut self, key: NodeIndex, graph: &ComputeGraphInner) {
        let node = graph.nodes.nodes.node_weight(key).expect("Node not found");
        let operation = match &node.variant {
            ComputeGraphNodeVariant::Custom(op) => op,
            _ => panic!("Expected Custom node"),
        };
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("custom ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        operation.visit_dependencies(&mut |dep| {
            let dep_id = self.identities.get(&dep).unwrap();
            self.statements.push(Stmt::Edge(
                Edge::head_node(dep_id.clone(), None).arrow_to_node(id.clone(), None),
            ));
        });
        self.identities.insert(key, id.clone());
    }
}

impl ComputeGraphInner {
    pub(crate) fn graphvis(&self, root: NodeIndex) -> Graph {
        let mut layout_pass = layout_pass::LayoutPass::default();
        layout_pass.visit(self, root);
        let mut graph_vis_pass = GraphVisPass {
            layout_pass,
            ..Default::default()
        };
        graph_vis_pass.queued.push_back(root);
        while let Some(node) = graph_vis_pass.queued.pop_front() {
            if graph_vis_pass.identities.contains_key(&node) {
                continue;
            }
            if let Some(data) = self.cached_results.get(&node) {
                let id = Identity::quoted(format!("cached ({}) #{:?}", data.info(), node));
                graph_vis_pass.statements.push(Stmt::Node {
                    id: id.clone(),
                    port: None,
                    attr: None,
                });
                graph_vis_pass.identities.insert(node, id.clone());
                continue;
            }

            let mut dependencies = Vec::new();
            self.visit_dependencies(node, &mut |dependent_key| {
                dependencies.push(dependent_key);
            });
            dependencies.retain(|dependency| !graph_vis_pass.identities.contains_key(dependency));
            if !dependencies.is_empty() {
                // If there are dependencies that are not resolved, push them to the queue then
                // revisit this node
                for dependency in dependencies {
                    graph_vis_pass.queued.push_back(dependency);
                }
                graph_vis_pass.queued.push_back(node);
                continue;
            }

            let node_data = self.nodes.nodes.node_weight(node).expect("Node not found");
            match &node_data.variant {
                ComputeGraphNodeVariant::ElementWise(_) => graph_vis_pass.visit_element_wise(node, self),
                ComputeGraphNodeVariant::PairWise(_) => graph_vis_pass.visit_pair_wise(node, self),
                ComputeGraphNodeVariant::MatMul(_) => graph_vis_pass.visit_mat_mul(node, self),
                ComputeGraphNodeVariant::QMatMul(_) => graph_vis_pass.visit_q_mat_mul(node, self),
                ComputeGraphNodeVariant::Reduce(_) => graph_vis_pass.visit_reduce(node, self),
                ComputeGraphNodeVariant::MapLayout(_) => graph_vis_pass.visit_map_layout(node, self),
                ComputeGraphNodeVariant::Resize(_) => graph_vis_pass.visit_resize(node, self),
                ComputeGraphNodeVariant::SliceAssign(_) => graph_vis_pass.visit_slice_assign(node, self),
                ComputeGraphNodeVariant::Tensor(_) => graph_vis_pass.visit_tensor(node, self),
                ComputeGraphNodeVariant::Dequantize(_) => graph_vis_pass.visit_dequantize(node, self),
                ComputeGraphNodeVariant::IndexSelect(_) => graph_vis_pass.visit_index_select(node, self),
                ComputeGraphNodeVariant::Custom(_) => graph_vis_pass.visit_custom(node, self),
            }
        }

        GraphBuilder::default()
            .graph_type(GraphType::DiGraph)
            .strict(false)
            .id(Identity::quoted("ComputeGraph"))
            .stmts(StmtList::new().extend(graph_vis_pass.statements))
            .build()
            .unwrap()
    }
}