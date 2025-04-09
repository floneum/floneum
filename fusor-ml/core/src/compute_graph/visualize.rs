use std::collections::{HashMap, HashSet, VecDeque};

use super::{
    AnyComputeKey, ComputeGraphNodes, DequantizeComputeKey, ElementWiseComputeNodeKey,
    IndexSelectComputeNodeKey, MapLayoutComputeNodeKey, MatMulComputeNodeKey,
    PairWiseComputeNodeKey, QMatMulComputeNodeKey, ReduceComputeNodeKey, ResizeComputeNodeKey,
    SliceAssignComputeNodeKey, TensorComputeNodeKey, layout_pass,
};
use tabbycat::Graph;
use tabbycat::{Edge, GraphBuilder, GraphType, Identity, Stmt, StmtList};

#[derive(Default)]
struct GraphVisPass {
    pub(crate) queued_nodes: VecDeque<AnyComputeKey>,
    pub(crate) queued_nodes_set: HashSet<AnyComputeKey>,
    layout_pass: layout_pass::LayoutPass,
    identities: HashMap<AnyComputeKey, Identity>,
    statements: Vec<Stmt>,
}

impl GraphVisPass {
    fn push_back(&mut self, key: AnyComputeKey) {
        if self.queued_nodes_set.insert(key) {
            self.queued_nodes.push_back(key);
        }
    }

    fn visit_element_wise(&mut self, key: ElementWiseComputeNodeKey, graph: &ComputeGraphNodes) {
        let operation = graph.element_wise.get(&key).unwrap();
        let Some(input) = self.identities.get(&operation.value.into()) else {
            self.push_back(operation.value);
            self.push_back(key.into());
            return;
        };
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

    fn visit_pair_wise(&mut self, key: PairWiseComputeNodeKey, graph: &ComputeGraphNodes) {
        let operation = graph.pair_wise.get(&key).unwrap();
        let Some(first) = self.identities.get(&operation.first.into()) else {
            self.push_back(operation.first);
            self.push_back(key.into());
            return;
        };
        let Some(second) = self.identities.get(&operation.second.into()) else {
            self.push_back(operation.second);
            self.push_back(key.into());
            return;
        };
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
        let Some(first) = self.identities.get(&operation.first.into()) else {
            self.push_back(operation.first);
            self.push_back(key.into());
            return;
        };
        let Some(second) = self.identities.get(&operation.second.into()) else {
            self.push_back(operation.second);
            self.push_back(key.into());
            return;
        };
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
        let Some(input) = self.identities.get(&operation.input.into()) else {
            self.push_back(operation.input);
            self.push_back(key.into());
            return;
        };
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
        let Some(input) = self.identities.get(&operation.value.into()) else {
            self.push_back(operation.value);
            self.push_back(key.into());
            return;
        };
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
    }

    fn visit_map_layout(
        &mut self,
        key: MapLayoutComputeNodeKey,
        graph: &ComputeGraphNodes,
    ) {
        let operation = graph.map_layout.get(&key).unwrap();
        let Some(input) = self.identities.get(&operation.input.into()) else {
            self.push_back(operation.input);
            self.push_back(key.into());
            return;
        };
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

    fn visit_slice(&mut self, key: MapLayoutComputeNodeKey, graph: &ComputeGraphNodes) {
        let operation = graph.map_layout.get(&key).unwrap();
        let Some(input) = self.identities.get(&operation.input.into()) else {
            self.push_back(operation.input);
            self.push_back(key.into());
            return;
        };
        let output_layout = self.layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("slice ({}) #{}", output_layout, key.0));
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
        let Some(input) = self.identities.get(&operation.input.into()) else {
            self.push_back(operation.input);
            self.push_back(key.into());
            return;
        };
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
        let Some(input) = self.identities.get(&operation.input.into()) else {
            self.push_back(operation.input);
            self.push_back(key.into());
            return;
        };
        let Some(value) = self.identities.get(&operation.value.into()) else {
            self.push_back(operation.value);
            self.push_back(key.into());
            return;
        };
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
        let Some(input) = self.identities.get(&key.into()) else {
            self.push_back(operation.input);
            self.push_back(key.into());
            return;
        };
        let Some(value) = self.identities.get(&key.into()) else {
            self.push_back(operation.indexes);
            self.push_back(key.into());
            return;
        };
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
}

impl ComputeGraphNodes {
    pub(crate) fn graphvis(&self, root: AnyComputeKey) -> Graph {
        let mut layout_pass = layout_pass::LayoutPass::default();
        layout_pass.visit(self, root);
        let mut graph_vis_pass = GraphVisPass::default();
        graph_vis_pass.layout_pass = layout_pass;
        graph_vis_pass.push_back(root);
        while let Some(node) = graph_vis_pass.queued_nodes.pop_front() {
            graph_vis_pass.queued_nodes_set.remove(&node);
            if graph_vis_pass.identities.contains_key(&node) {
                continue;
            }
            match node {
                AnyComputeKey::ElementWise(key) => graph_vis_pass.visit_element_wise(key, self),
                AnyComputeKey::PairWise(key) => graph_vis_pass.visit_pair_wise(key, self),
                AnyComputeKey::MatMul(key) => graph_vis_pass.visit_mat_mul(key, self),
                AnyComputeKey::QMatMul(key) => graph_vis_pass.visit_q_mat_mul(key, self),
                AnyComputeKey::Reduce(key) => graph_vis_pass.visit_reduce(key, self),
                AnyComputeKey::MapLayout(key) => graph_vis_pass.visit_map_layout(key, self),
                AnyComputeKey::Resize(key) => graph_vis_pass.visit_resize(key, self),
                AnyComputeKey::SliceAssign(key) => graph_vis_pass.visit_slice_assign(key, self),
                AnyComputeKey::Tensor(key) => graph_vis_pass.visit_tensor(key, self),
                AnyComputeKey::Dequantize(key) => graph_vis_pass.visit_dequantize(key, self),
                AnyComputeKey::IndexSelect(key) => graph_vis_pass.visit_index_select(key, self),
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
