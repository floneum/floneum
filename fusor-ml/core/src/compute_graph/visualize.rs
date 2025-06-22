use rustc_hash::FxHashMap;

use crate::compute_graph::CustomComputeKey;

use super::dependency_map::visit_dependencies;
use super::queue::ComputeQueue;
use super::{
    AnyComputeKey, ComputeGraphInner, ComputeGraphNodes, DequantizeComputeKey,
    ElementWiseComputeNodeKey, IndexSelectComputeNodeKey, MapLayoutComputeNodeKey,
    MatMulComputeNodeKey, PairWiseComputeNodeKey, QMatMulComputeNodeKey, ReduceComputeNodeKey,
    ResizeComputeNodeKey, SliceAssignComputeNodeKey, TensorComputeNodeKey, layout_pass,
};
use tabbycat::Graph;
use tabbycat::{Edge, GraphBuilder, GraphType, Identity, Stmt, StmtList};

#[derive(Default)]
struct GraphVisPass {
    queued: ComputeQueue,
    layout_pass: layout_pass::LayoutPass,
    identities: FxHashMap<AnyComputeKey, Identity>,
    statements: Vec<Stmt>,
}

impl GraphVisPass {
    fn visit_element_wise(&mut self, key: ElementWiseComputeNodeKey, graph: &ComputeGraphNodes) {
        let operation = graph.element_wise.get(&key).unwrap();
        let input = self.identities.get(&operation.value).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!(
            "{} ({}) #{}",
            operation.name(),
            output_layout,
            key.0
        ));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key.into(), id.clone());
    }

    fn visit_pair_wise(&mut self, key: PairWiseComputeNodeKey, graph: &ComputeGraphNodes) {
        let operation = graph.pair_wise.get(&key).unwrap();
        let first = self.identities.get(&operation.first).unwrap();
        let second = self.identities.get(&operation.second).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!(
            "{} ({}) #{}",
            operation.function.name(),
            output_layout,
            key.0
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
        self.identities.insert(key.into(), id.clone());
    }

    fn visit_mat_mul(&mut self, key: MatMulComputeNodeKey, graph: &ComputeGraphNodes) {
        let operation = graph.mat_mul.get(&key).unwrap();
        let first = self.identities.get(&operation.first).unwrap();
        let second = self.identities.get(&operation.second).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("matmul ({}) #{}", output_layout, key.0));
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
        self.identities.insert(key.into(), id.clone());
    }

    fn visit_q_mat_mul(&mut self, key: QMatMulComputeNodeKey, graph: &ComputeGraphNodes) {
        let operation = graph.q_mat_mul.get(&key).unwrap();
        let input = self.identities.get(&operation.input).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("qmatmul ({}) #{}", output_layout, key.0));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key.into(), id.clone());
    }

    fn visit_reduce(&mut self, key: ReduceComputeNodeKey, graph: &ComputeGraphNodes) {
        let operation = graph.reduce.get(&key).unwrap();
        let input = self.identities.get(&operation.value).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!(
            "{} ({}) #{}",
            operation.function.name(),
            output_layout,
            key.0
        ));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key.into(), id.clone());
    }

    fn visit_map_layout(&mut self, key: MapLayoutComputeNodeKey, graph: &ComputeGraphNodes) {
        let operation = graph.map_layout.get(&key).unwrap();
        let input = self.identities.get(&operation.input).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("map_layout ({}) #{}", output_layout, key.0));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key.into(), id.clone());
    }

    fn visit_resize(&mut self, key: ResizeComputeNodeKey, graph: &ComputeGraphNodes) {
        let operation = graph.resize.get(&key).unwrap();
        let input = self.identities.get(&operation.input).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("resize ({}) #{}", output_layout, key.0));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key.into(), id.clone());
    }

    fn visit_slice_assign(&mut self, key: SliceAssignComputeNodeKey, graph: &ComputeGraphNodes) {
        let operation = graph.slice_assign.get(&key).unwrap();
        let input = self.identities.get(&operation.input).unwrap();
        let value = self.identities.get(&operation.value).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("slice_assign ({}) #{}", output_layout, key.0));
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
        self.identities.insert(key.into(), id.clone());
    }

    fn visit_index_select(&mut self, key: IndexSelectComputeNodeKey, graph: &ComputeGraphNodes) {
        let operation = graph.index_select.get(&key).unwrap();
        let input = self.identities.get(&operation.input).unwrap();
        let value = self.identities.get(&operation.indexes).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("index_select ({}) #{}", output_layout, key.0));
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
        self.identities.insert(key.into(), id.clone());
    }

    fn visit_dequantize(&mut self, key: DequantizeComputeKey, _: &ComputeGraphNodes) {
        let output_layout = self.layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("dequantize ({}) #{}", output_layout, key.0));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.identities.insert(key.into(), id.clone());
    }

    fn visit_tensor(&mut self, key: TensorComputeNodeKey, _: &ComputeGraphNodes) {
        let output_layout = self.layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("tensor ({}) #{}", output_layout, key.0));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.identities.insert(key.into(), id.clone());
    }

    fn visit_custom(&mut self, key: CustomComputeKey, graph: &ComputeGraphNodes) {
        let operation = graph.custom.get(&key).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("custom ({}) #{}", output_layout, key.0));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        operation.visit_dependencies(&mut |dep| {
            let id = self.identities.get(&dep).unwrap();
            self.statements.push(Stmt::Edge(
                Edge::head_node(id.clone(), None).arrow_to_node(id.clone(), None),
            ));
        });
        self.identities.insert(key.into(), id.clone());
    }
}

impl ComputeGraphInner {
    pub(crate) fn graphvis(&self, root: AnyComputeKey) -> Graph {
        let mut layout_pass = layout_pass::LayoutPass::default();
        layout_pass.visit(self, root);
        let mut graph_vis_pass = GraphVisPass::default();
        graph_vis_pass.layout_pass = layout_pass;
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
            visit_dependencies(&self.nodes, node, |dependent_key| {
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
            let nodes = &self.nodes;
            match node {
                AnyComputeKey::ElementWise(key) => graph_vis_pass.visit_element_wise(key, nodes),
                AnyComputeKey::PairWise(key) => graph_vis_pass.visit_pair_wise(key, nodes),
                AnyComputeKey::MatMul(key) => graph_vis_pass.visit_mat_mul(key, nodes),
                AnyComputeKey::QMatMul(key) => graph_vis_pass.visit_q_mat_mul(key, nodes),
                AnyComputeKey::Reduce(key) => graph_vis_pass.visit_reduce(key, nodes),
                AnyComputeKey::MapLayout(key) => graph_vis_pass.visit_map_layout(key, nodes),
                AnyComputeKey::Resize(key) => graph_vis_pass.visit_resize(key, nodes),
                AnyComputeKey::SliceAssign(key) => graph_vis_pass.visit_slice_assign(key, nodes),
                AnyComputeKey::Tensor(key) => graph_vis_pass.visit_tensor(key, nodes),
                AnyComputeKey::Dequantize(key) => graph_vis_pass.visit_dequantize(key, nodes),
                AnyComputeKey::IndexSelect(key) => graph_vis_pass.visit_index_select(key, nodes),
                AnyComputeKey::Custom(key) => graph_vis_pass.visit_custom(key, nodes),
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
