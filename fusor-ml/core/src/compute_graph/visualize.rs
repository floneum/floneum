use std::collections::HashMap;

use super::visit::VisitComputeGraph;
use super::{
    AnyComputeKey, ComputeGraphNodes, ElementWiseComputeNodeKey, IndexSelectComputeNodeKey,
    MapLayoutComputeNodeKey, MatMulComputeNodeKey, PairWiseComputeNodeKey, QMatMulComputeNodeKey,
    ReduceComputeNodeKey, ResizeComputeNodeKey, SliceAssignComputeNodeKey, TensorComputeNodeKey,
    layout_pass,
};
use tabbycat::Graph;
use tabbycat::{Edge, GraphBuilder, GraphType, Identity, Stmt, StmtList};

impl ComputeGraphNodes {
    pub(crate) fn graphvis(&self, root: AnyComputeKey) -> Graph {
        let mut layout_pass = layout_pass::LayoutPass::default();
        layout_pass.visit(self, root);
        let mut statements = Vec::new();
        let mut identities = HashMap::new();
        self.add_node_to_graph(&mut statements, root, &layout_pass, &mut identities);
        GraphBuilder::default()
            .graph_type(GraphType::DiGraph)
            .strict(false)
            .id(Identity::quoted("ComputeGraph"))
            .stmts(StmtList::new().extend(statements))
            .build()
            .unwrap()
    }

    fn add_node_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: AnyComputeKey,
        layout_pass: &layout_pass::LayoutPass,
        identities: &mut HashMap<AnyComputeKey, Identity>,
    ) -> Identity {
        if let Some(id) = identities.get(&key) {
            return id.clone();
        }
        let id = match key {
            AnyComputeKey::ElementWise(element_wise_compute_node_key) => self
                .add_element_wise_to_graph(
                    graph,
                    element_wise_compute_node_key,
                    layout_pass,
                    identities,
                ),
            AnyComputeKey::PairWise(pair_wise_compute_node_key) => self.add_pair_wise_to_graph(
                graph,
                pair_wise_compute_node_key,
                layout_pass,
                identities,
            ),
            AnyComputeKey::MatMul(mat_mul_compute_node_key) => {
                self.add_mat_mul_to_graph(graph, mat_mul_compute_node_key, layout_pass, identities)
            }
            AnyComputeKey::QMatMul(quantized_mat_mul_compute_node_key) => self
                .add_q_mat_mul_to_graph(
                    graph,
                    quantized_mat_mul_compute_node_key,
                    layout_pass,
                    identities,
                ),
            AnyComputeKey::Reduce(reduce_compute_node_key) => {
                self.add_reduce_to_graph(graph, reduce_compute_node_key, layout_pass, identities)
            }
            AnyComputeKey::Tensor(tensor_compute_node_key) => {
                self.add_tensor_to_graph(graph, tensor_compute_node_key, layout_pass, identities)
            }
            AnyComputeKey::MapLayout(slice_compute_node_key) => {
                self.add_slice_to_graph(graph, slice_compute_node_key, layout_pass, identities)
            }
            AnyComputeKey::Resize(resize_compute_node_key) => {
                self.add_resize_to_graph(graph, resize_compute_node_key, layout_pass, identities)
            }
            AnyComputeKey::SliceAssign(slice_assign_compute_node_key) => self
                .add_slice_assign_to_graph(
                    graph,
                    slice_assign_compute_node_key,
                    layout_pass,
                    identities,
                ),
            AnyComputeKey::IndexSelect(index_select_compute_node_key) => self
                .add_index_select_to_graph(
                    graph,
                    index_select_compute_node_key,
                    layout_pass,
                    identities,
                ),
        };
        identities.insert(key, id.clone());
        id
    }

    fn add_element_wise_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: ElementWiseComputeNodeKey,
        layout_pass: &layout_pass::LayoutPass,
        identities: &mut HashMap<AnyComputeKey, Identity>,
    ) -> Identity {
        let operation = self.element_wise.get(&key).unwrap();
        let input = self.add_node_to_graph(graph, operation.value, layout_pass, identities);
        let output_layout = layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!(
            "{} ({}) #{}",
            operation.function.name(),
            output_layout,
            key.0
        ));
        graph.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        graph.push(Stmt::Edge(
            Edge::head_node(input, None).arrow_to_node(id.clone(), None),
        ));
        id
    }

    fn add_pair_wise_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: PairWiseComputeNodeKey,
        layout_pass: &layout_pass::LayoutPass,
        identities: &mut HashMap<AnyComputeKey, Identity>,
    ) -> Identity {
        let operation = self.pair_wise.get(&key).unwrap();
        let first = self.add_node_to_graph(graph, operation.first, layout_pass, identities);
        let second = self.add_node_to_graph(graph, operation.second, layout_pass, identities);
        let output_layout = layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!(
            "{} ({}) #{}",
            operation.function.name(),
            output_layout,
            key.0
        ));
        graph.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        graph.push(Stmt::Edge(
            Edge::head_node(first, None).arrow_to_node(id.clone(), None),
        ));
        graph.push(Stmt::Edge(
            Edge::head_node(second, None).arrow_to_node(id.clone(), None),
        ));
        id
    }

    fn add_mat_mul_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: MatMulComputeNodeKey,
        layout_pass: &layout_pass::LayoutPass,
        identities: &mut HashMap<AnyComputeKey, Identity>,
    ) -> Identity {
        let operation = self.mat_mul.get(&key).unwrap();
        let first = self.add_node_to_graph(graph, operation.first, layout_pass, identities);
        let second = self.add_node_to_graph(graph, operation.second, layout_pass, identities);
        let output_layout = layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("matmul ({}) #{}", output_layout, key.0));
        graph.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        graph.push(Stmt::Edge(
            Edge::head_node(first, None).arrow_to_node(id.clone(), None),
        ));
        graph.push(Stmt::Edge(
            Edge::head_node(second, None).arrow_to_node(id.clone(), None),
        ));
        id
    }

    fn add_q_mat_mul_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: QMatMulComputeNodeKey,
        layout_pass: &layout_pass::LayoutPass,
        identities: &mut HashMap<AnyComputeKey, Identity>,
    ) -> Identity {
        let operation = self.q_mat_mul.get(&key).unwrap();
        let input = self.add_node_to_graph(graph, operation.input, layout_pass, identities);
        let output_layout = layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("qmatmul ({}) #{}", output_layout, key.0));
        graph.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        graph.push(Stmt::Edge(
            Edge::head_node(input, None).arrow_to_node(id.clone(), None),
        ));
        id
    }

    fn add_reduce_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: ReduceComputeNodeKey,
        layout_pass: &layout_pass::LayoutPass,
        identities: &mut HashMap<AnyComputeKey, Identity>,
    ) -> Identity {
        let operation = self.reduce.get(&key).unwrap();
        let input = self.add_node_to_graph(graph, operation.value, layout_pass, identities);
        let output_layout = layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!(
            "{} ({}) #{}",
            operation.function.name(),
            output_layout,
            key.0
        ));
        graph.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        graph.push(Stmt::Edge(
            Edge::head_node(input, None).arrow_to_node(id.clone(), None),
        ));
        id
    }

    fn add_slice_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: MapLayoutComputeNodeKey,
        layout_pass: &layout_pass::LayoutPass,
        identities: &mut HashMap<AnyComputeKey, Identity>,
    ) -> Identity {
        let operation = self.map_layout.get(&key).unwrap();
        let input = self.add_node_to_graph(graph, operation.input, layout_pass, identities);
        let output_layout = layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("slice ({}) #{}", output_layout, key.0));
        graph.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        graph.push(Stmt::Edge(
            Edge::head_node(input, None).arrow_to_node(id.clone(), None),
        ));
        id
    }

    fn add_resize_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: ResizeComputeNodeKey,
        layout_pass: &layout_pass::LayoutPass,
        identities: &mut HashMap<AnyComputeKey, Identity>,
    ) -> Identity {
        let operation = self.resize.get(&key).unwrap();
        let input = self.add_node_to_graph(graph, operation.input, layout_pass, identities);
        let output_layout = layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("resize ({}) #{}", output_layout, key.0));
        graph.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        graph.push(Stmt::Edge(
            Edge::head_node(input, None).arrow_to_node(id.clone(), None),
        ));
        id
    }

    fn add_slice_assign_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: SliceAssignComputeNodeKey,
        layout_pass: &layout_pass::LayoutPass,
        identities: &mut HashMap<AnyComputeKey, Identity>,
    ) -> Identity {
        let operation = self.slice_assign.get(&key).unwrap();
        let input = self.add_node_to_graph(graph, operation.input, layout_pass, identities);
        let value = self.add_node_to_graph(graph, operation.value, layout_pass, identities);
        let output_layout = layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("slice_assign ({}) #{}", output_layout, key.0));
        graph.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        graph.push(Stmt::Edge(
            Edge::head_node(input, None).arrow_to_node(id.clone(), None),
        ));
        graph.push(Stmt::Edge(
            Edge::head_node(value, None).arrow_to_node(id.clone(), None),
        ));
        id
    }

    fn add_index_select_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: IndexSelectComputeNodeKey,
        layout_pass: &layout_pass::LayoutPass,
        identities: &mut HashMap<AnyComputeKey, Identity>,
    ) -> Identity {
        let operation = self.index_select.get(&key).unwrap();
        let input = self.add_node_to_graph(graph, operation.input, layout_pass, identities);
        let value = self.add_node_to_graph(graph, operation.indexes, layout_pass, identities);
        let output_layout = layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("index_select ({}) #{}", output_layout, key.0));
        graph.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        graph.push(Stmt::Edge(
            Edge::head_node(input, None).arrow_to_node(id.clone(), None),
        ));
        graph.push(Stmt::Edge(
            Edge::head_node(value, None).arrow_to_node(id.clone(), None),
        ));
        id
    }

    fn add_tensor_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: TensorComputeNodeKey,
        layout_pass: &layout_pass::LayoutPass,
        _: &mut HashMap<AnyComputeKey, Identity>,
    ) -> Identity {
        let output_layout = layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("tensor ({}) #{}", output_layout, key.0));
        graph.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        id
    }
}
