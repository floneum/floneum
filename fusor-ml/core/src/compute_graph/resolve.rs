use std::fmt::Write;
use std::sync::Arc;

use petgraph::algo::toposort;
use petgraph::stable_graph::StableGraph;
use rustc_hash::{FxHashMap, FxHashSet};
use wgpu::CommandEncoder;

use crate::{
    ElementWiseFunctions, ElementWiseOperation,
    mir::{
        inputs::{KernelInputValue, MirValue},
        kernel::GenericKernel,
        operation::Operation,
        workgroup_shape::{self, WorkgroupShapeConstraints},
    },
    nary_wise::{NaryExpr, NaryOperation},
    quantized::matmul::QMatMulOperation,
    tensor::TensorData,
};

use super::{ComputeGraphInner, ComputeGraphNodeVariant, NodeIndex};

pub(crate) struct ResolverResult {
    pub(crate) data: TensorData,
    pub(crate) total_kernels: usize,
}

#[derive(Debug, Clone)]
struct ExecutionNode {
    inner_idx: NodeIndex,
    variant: ComputeGraphNodeVariant,
}

type ExecutionGraph = StableGraph<ExecutionNode, ()>;
type ExecutionNodeIndex = petgraph::graph::NodeIndex;

pub(crate) struct Resolver<'a> {
    command_encoder: &'a mut CommandEncoder,
    execution_graph: ExecutionGraph,
    node_mapping: FxHashMap<NodeIndex, ExecutionNodeIndex>,
    target: NodeIndex,
    resolved_set: FxHashSet<NodeIndex>,
}

impl<'a> Resolver<'a> {
    pub(crate) fn new(
        graph: &mut ComputeGraphInner,
        target: NodeIndex,
        command_encoder: &'a mut CommandEncoder,
    ) -> Self {
        let resolved_set = graph
            .nodes
            .nodes
            .node_indices()
            .filter(|&idx| {
                graph
                    .nodes
                    .nodes
                    .node_weight(idx)
                    .map(|n| n.cached.is_some())
                    .unwrap_or(false)
            })
            .collect();
        Self {
            command_encoder,
            target,
            execution_graph: Default::default(),
            node_mapping: Default::default(),
            resolved_set,
        }
    }

    pub(crate) fn run(&mut self, graph: &mut ComputeGraphInner) -> ResolverResult {
        let limits = graph.device.limits();

        // Pass 1: Build execution graph
        self.build_execution_graph(graph, self.target);

        // Pass 2: Apply Rewrite Rules
        self.optimize(graph);

        // Pass 3: Topological Sort
        let sorted_nodes = toposort(&self.execution_graph, None)
            .unwrap_or_else(|_| panic!("Cycle detected in execution graph"));

        // Pass 4: Execution
        // Extract operations in order
        let mut queued_operations = Vec::with_capacity(sorted_nodes.len());
        for idx in sorted_nodes {
            let node = &self.execution_graph[idx];
            // Handle Tensor caching explicitly here
            if let ComputeGraphNodeVariant::Tensor(data) = &node.variant {
                graph.set_cached_result(node.inner_idx, data.clone());
                continue;
            }

            if let Some(op) = self.lower_node(node) {
                queued_operations.push((node.inner_idx, op));
            }
        }

        // Find runs of compatible dispatch shapes
        let mut current_constraints = WorkgroupShapeConstraints::new();
        let mut pending_operations = Vec::new();
        let mut inputs = Vec::new();
        let mut all_input_values = Vec::new();
        let mut kernel = GenericKernel::new();
        let mut total_kernels = 0;

        for (node, operation) in queued_operations {
            let new_inputs = operation.inputs(graph);
            let constraint = operation.workgroup_shape_constraints(&graph.device);
            let mut new_merged = current_constraints.clone();
            new_merged.merge(&constraint);
            let old_best = current_constraints.solve(&limits).unwrap_or_else(|| {
                panic!(
                    "Failed to find a valid workgroup shape for constraints {current_constraints:?}"
                )
            });
            let mut extend = self.should_extend_kernel(new_inputs.clone(), &inputs);
            extend &= new_merged.solve(&limits).is_some();
            if extend {
                current_constraints = new_merged;
            } else {
                if !pending_operations.is_empty() {
                    total_kernels += 1;
                    self.flush_operations(
                        graph,
                        &mut kernel,
                        &pending_operations,
                        &inputs,
                        &all_input_values,
                        old_best,
                    );
                    pending_operations.clear();
                    all_input_values.clear();
                    inputs.clear();
                    kernel.clear();
                }
                current_constraints = constraint;
            }
            // Map layout isn't really a kernel. Resolve it immediately
            let map_layout = if let Some(node_data) = graph.nodes.nodes.node_weight(node) {
                match &node_data.variant {
                    ComputeGraphNodeVariant::MapLayout(map_layout) => Some(map_layout.clone()),
                    ComputeGraphNodeVariant::Resize(resize) => resize.lower(graph),
                    _ => None,
                }
            } else {
                None
            };
            if let Some(map_layout) = map_layout {
                let result = map_layout.run(graph);
                // Cache the result
                graph.set_cached_result(node, result);
            } else {
                self.push_operation(
                    graph,
                    new_inputs,
                    &mut kernel,
                    node,
                    operation,
                    &mut inputs,
                    &mut all_input_values,
                    &mut pending_operations,
                );
            };
        }

        if !pending_operations.is_empty() {
            let old_best = current_constraints.solve(&limits).unwrap_or_else(|| {
                panic!(
                    "Failed to find a valid workgroup shape for constraints {current_constraints:?}"
                )
            });
            total_kernels += 1;
            self.flush_operations(
                graph,
                &mut kernel,
                &pending_operations,
                &inputs,
                &all_input_values,
                old_best,
            );
        }

        let data = graph
            .get_result(self.target)
            .expect("Target result not cached");
        ResolverResult {
            data,
            total_kernels,
        }
    }

    fn build_execution_graph(
        &mut self,
        graph: &ComputeGraphInner,
        node: NodeIndex,
    ) -> Option<ExecutionNodeIndex> {
        if self.resolved_set.contains(&node) {
            return None;
        }
        if let Some(&idx) = self.node_mapping.get(&node) {
            return Some(idx);
        }

        let node_data = graph
            .nodes
            .nodes
            .node_weight(node)
            .expect("Node not found in graph");
        let variant = node_data.variant.clone();

        // Add to execution graph
        let exec_idx = self.execution_graph.add_node(ExecutionNode {
            inner_idx: node,
            variant: variant.clone(),
        });
        self.node_mapping.insert(node, exec_idx);

        // Find dependencies
        let mut dependencies = Vec::new();
        variant.visit_dependencies(&mut |dependency| {
            dependencies.push(dependency);
        });

        for dependency in dependencies {
            if let Some(dep_exec_idx) = self.build_execution_graph(graph, dependency) {
                self.execution_graph.add_edge(dep_exec_idx, exec_idx, ());
            }
        }

        Some(exec_idx)
    }

    fn lower_node(&self, node: &ExecutionNode) -> Option<Arc<dyn Operation>> {
        match &node.variant {
            ComputeGraphNodeVariant::ElementWise(op) => {
                let inputs = vec![op.value];
                // Construct NaryExpr for simple unary chain
                let expression =
                    self.wrap_with_element_wise_functions(NaryExpr::Input(0), &op.functions);

                let shape: Box<[_]> = op.shape().into();
                let final_output_datatype = op.functions.out_datatype();
                let nary = NaryOperation {
                    inputs,
                    expression,
                    shape,
                    output_datatype: final_output_datatype,
                };
                Some(Arc::new(nary))
            }
            ComputeGraphNodeVariant::PairWise(op) => {
                let inputs = vec![op.first, op.second];
                let ty = op.function.datatype;
                let expression = NaryExpr::Op {
                    children: vec![NaryExpr::Input(0), NaryExpr::Input(1)],
                    function: op.function.to_nary_function(ty, ty),
                };

                let shape: Box<[_]> = op.shape().into();
                let final_output_datatype = op.function.datatype;
                let nary = NaryOperation {
                    inputs,
                    expression,
                    shape,
                    output_datatype: final_output_datatype,
                };
                Some(Arc::new(nary))
            }
            ComputeGraphNodeVariant::Nary(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::MatMul(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::Reduce(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::MapLayout(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::Resize(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::SliceAssign(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::IndexSelect(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::QMatMul(op) => Some(Arc::new(QMatMulOperation::new(
                op.input_datatype,
                &op.in_shape,
                op.input,
                op.matrix.clone(),
            ))),
            ComputeGraphNodeVariant::Dequantize(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::Tensor(_) => None, // Handled in execution loop
            ComputeGraphNodeVariant::Custom(op) => Some(op.clone()),
        }
    }

    // --- Rewrite Engine ---

    fn optimize(&mut self, graph: &mut ComputeGraphInner) {
        let mut changed = true;
        while changed {
            changed = false;
            let nodes: Vec<_> = self.execution_graph.node_indices().collect();

            for node_idx in nodes {
                if !self.execution_graph.contains_node(node_idx) {
                    continue;
                }

                if self.try_fuse_into_nary(graph, node_idx) {
                    changed = true;
                    continue;
                }
                if self.try_convert_pairwise_to_elementwise(graph, node_idx) {
                    changed = true;
                    continue;
                }
                if self.try_fuse_element_wise_chain(graph, node_idx) {
                    changed = true;
                    continue;
                }
                if self.try_fuse_into_reduce(graph, node_idx) {
                    changed = true;
                    continue;
                }
                if self.try_fuse_into_matmul(graph, node_idx) {
                    changed = true;
                    continue;
                }
                if self.try_fuse_into_dequantize(graph, node_idx) {
                    changed = true;
                    continue;
                }
                if self.try_fuse_into_index_select(graph, node_idx) {
                    changed = true;
                    continue;
                }
            }
        }
    }

    // Helpers
    fn add_physical_dependencies(
        &self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
        inputs: &[NodeIndex],
    ) {
        let inner_idx = self.execution_graph[node_idx].inner_idx;
        for &input in inputs {
            graph.nodes.nodes.add_edge(input, inner_idx, ());
        }
    }

    fn get_input_node_in_exec_graph(&self, inner_input: NodeIndex) -> Option<ExecutionNodeIndex> {
        self.node_mapping.get(&inner_input).copied()
    }

    fn check_cached(&self, graph: &ComputeGraphInner, inner_idx: NodeIndex) -> bool {
        graph.get_cached_result(inner_idx).is_some()
    }

    fn remove_node_if_dead(&mut self, node_idx: ExecutionNodeIndex) {
        if !self.execution_graph.contains_node(node_idx) {
            return;
        }
        if self
            .execution_graph
            .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
            .count()
            == 0
        {
            // Collect incoming neighbors before removing
            let incoming: Vec<_> = self
                .execution_graph
                .neighbors_directed(node_idx, petgraph::Direction::Incoming)
                .collect();
            self.execution_graph.remove_node(node_idx);
            // Recursively check if dependencies are now dead
            for dep in incoming {
                self.remove_node_if_dead(dep);
            }
        }
    }

    // Rules

    /// Fuse chains of PairWise and ElementWise operations into a single Nary operation.
    /// This allows combining operations like `(a + b) * c` into one kernel.
    fn try_fuse_into_nary(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();

        match &node_variant {
            ComputeGraphNodeVariant::PairWise(op) => {
                self.try_fuse_pairwise_into_nary(graph, node_idx, op.clone())
            }
            ComputeGraphNodeVariant::ElementWise(op) => {
                self.try_fuse_elementwise_into_nary(graph, node_idx, op.clone())
            }
            _ => false,
        }
    }

    fn try_fuse_pairwise_into_nary(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
        op: crate::PairWiseOperation,
    ) -> bool {
        let (first, second, shape, output_datatype, function) = (
            op.first,
            op.second,
            op.shape(),
            op.function.datatype,
            op.function.clone(),
        );

        // Check if at least one input can be fused
        let first_can_fuse = self.can_fuse_into_nary(graph, first);
        let second_can_fuse = self.can_fuse_into_nary(graph, second);

        if !first_can_fuse && !second_can_fuse {
            return false;
        }

        // Collect all inputs and build expression tree
        let mut inputs = Vec::new();
        let left_expr = self.collect_nary_expr_recursive(graph, first, &mut inputs);
        let right_expr = self.collect_nary_expr_recursive(graph, second, &mut inputs);

        let expression = NaryExpr::Op {
            children: vec![left_expr, right_expr],
            function: function.to_nary_function(output_datatype, output_datatype),
        };

        let nary = NaryOperation {
            inputs: inputs.clone(),
            expression,
            shape: shape.into(),
            output_datatype,
        };

        // Update the node to be an Nary operation
        self.execution_graph[node_idx].variant = ComputeGraphNodeVariant::Nary(nary.clone());

        // Remove edges from old inputs
        if let Some(first_exec) = self.get_input_node_in_exec_graph(first) {
            if let Some(edge) = self.execution_graph.find_edge(first_exec, node_idx) {
                self.execution_graph.remove_edge(edge);
            }
        }
        if let Some(second_exec) = self.get_input_node_in_exec_graph(second) {
            if let Some(edge) = self.execution_graph.find_edge(second_exec, node_idx) {
                self.execution_graph.remove_edge(edge);
            }
        }

        // Add edges from new inputs (the leaves of the expression tree)
        for input in &inputs {
            if let Some(input_exec) = self.get_input_node_in_exec_graph(*input) {
                self.execution_graph.add_edge(input_exec, node_idx, ());
            }
        }

        self.add_physical_dependencies(graph, node_idx, &nary.inputs);

        // Remove dead intermediate nodes
        if let Some(first_exec) = self.get_input_node_in_exec_graph(first) {
            self.remove_node_if_dead(first_exec);
        }
        if let Some(second_exec) = self.get_input_node_in_exec_graph(second) {
            self.remove_node_if_dead(second_exec);
        }

        true
    }

    fn try_fuse_elementwise_into_nary(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
        op: crate::ElementWiseOperation,
    ) -> bool {
        let input_inner = op.value;

        // Check if the input can be fused (is a PairWise or ElementWise that we can absorb)
        if !self.can_fuse_into_nary(graph, input_inner) {
            return false;
        }

        // Collect all inputs and build expression tree
        let mut inputs = Vec::new();
        let child_expr = self.collect_nary_expr_recursive(graph, input_inner, &mut inputs);
        let expression = self.wrap_with_element_wise_functions(child_expr, &op.functions);

        let shape: Box<[_]> = op.shape().into();
        let output_datatype = op.functions.out_datatype();

        let nary = NaryOperation {
            inputs: inputs.clone(),
            expression,
            shape,
            output_datatype,
        };

        // Update the node to be an Nary operation
        self.execution_graph[node_idx].variant = ComputeGraphNodeVariant::Nary(nary.clone());

        // Remove edge from old input
        if let Some(input_exec) = self.get_input_node_in_exec_graph(input_inner) {
            if let Some(edge) = self.execution_graph.find_edge(input_exec, node_idx) {
                self.execution_graph.remove_edge(edge);
            }
        }

        // Add edges from new inputs (the leaves of the expression tree)
        for input in &inputs {
            if let Some(input_exec) = self.get_input_node_in_exec_graph(*input) {
                self.execution_graph.add_edge(input_exec, node_idx, ());
            }
        }

        self.add_physical_dependencies(graph, node_idx, &nary.inputs);

        // Remove dead intermediate nodes
        if let Some(input_exec) = self.get_input_node_in_exec_graph(input_inner) {
            self.remove_node_if_dead(input_exec);
        }

        true
    }

    /// Check if a node can be fused into a parent nary operation
    fn can_fuse_into_nary(&self, graph: &ComputeGraphInner, inner_idx: NodeIndex) -> bool {
        // Don't fuse cached results
        if self.check_cached(graph, inner_idx) {
            return false;
        }

        // Get the execution node
        let exec_idx = match self.get_input_node_in_exec_graph(inner_idx) {
            Some(idx) => idx,
            None => return false,
        };

        // Note: We allow nodes with multiple consumers to be fused.
        // If a node is used multiple times in the expression tree, the subexpression
        // will be computed multiple times. This is typically fine for elementwise ops
        // as the kernel launch overhead savings outweigh the extra computation.

        let variant = &self.execution_graph[exec_idx].variant;
        matches!(
            variant,
            ComputeGraphNodeVariant::PairWise(_) | ComputeGraphNodeVariant::ElementWise(_)
        )
    }

    /// Recursively collect expression tree from execution graph
    fn collect_nary_expr_recursive(
        &self,
        graph: &ComputeGraphInner,
        inner_idx: NodeIndex,
        inputs: &mut Vec<NodeIndex>,
    ) -> NaryExpr {
        // If this is cached or not in execution graph, it's a leaf input
        if self.check_cached(graph, inner_idx) {
            return self.add_nary_leaf_input(inner_idx, inputs);
        }

        let exec_idx = match self.get_input_node_in_exec_graph(inner_idx) {
            Some(idx) => idx,
            None => return self.add_nary_leaf_input(inner_idx, inputs),
        };

        // Note: We don't check consumer count here. Even if a node has multiple consumers,
        // we allow it to be inlined. The subexpression will be computed multiple times
        // in the kernel, but this is typically faster than launching a separate kernel.

        let variant = self.execution_graph[exec_idx].variant.clone();

        match variant {
            ComputeGraphNodeVariant::PairWise(op) => {
                let left_expr = self.collect_nary_expr_recursive(graph, op.first, inputs);
                let right_expr = self.collect_nary_expr_recursive(graph, op.second, inputs);

                let ty = op.function.datatype;
                NaryExpr::Op {
                    children: vec![left_expr, right_expr],
                    function: op.function.to_nary_function(ty, ty),
                }
            }
            ComputeGraphNodeVariant::ElementWise(op) => {
                let child_expr = self.collect_nary_expr_recursive(graph, op.value, inputs);
                self.wrap_with_element_wise_functions(child_expr, &op.functions)
            }
            _ => self.add_nary_leaf_input(inner_idx, inputs),
        }
    }

    /// Add a leaf input to the inputs list and return an Input expression
    fn add_nary_leaf_input(&self, inner_idx: NodeIndex, inputs: &mut Vec<NodeIndex>) -> NaryExpr {
        // Check if this input already exists (deduplication)
        if let Some(idx) = inputs.iter().position(|&n| n == inner_idx) {
            return NaryExpr::Input(idx);
        }

        let idx = inputs.len();
        inputs.push(inner_idx);
        NaryExpr::Input(idx)
    }

    fn try_convert_pairwise_to_elementwise(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();

        if let ComputeGraphNodeVariant::PairWise(op) = node_variant {
            let ty = op.function.datatype;
            let expression = NaryExpr::Op {
                children: vec![NaryExpr::Input(0), NaryExpr::Input(1)],
                function: op.function.to_nary_function(ty, ty),
            };

            let inputs = vec![op.first, op.second];
            let shape: Box<[_]> = op.shape().into();
            let final_output_datatype = op.function.datatype;
            let nary = NaryOperation {
                inputs,
                expression,
                shape,
                output_datatype: final_output_datatype,
            };

            if let Some(funcs) = nary.try_into_elementwise_op() {
                let new_op = ElementWiseOperation::from_element_wise(
                    funcs.value,
                    funcs.functions,
                    op.shape(),
                );
                self.execution_graph[node_idx].variant =
                    ComputeGraphNodeVariant::ElementWise(new_op.clone());
                self.add_physical_dependencies(graph, node_idx, &[new_op.value]);
                return true;
            }
        }
        false
    }

    fn try_fuse_element_wise_chain(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();
        let (input_inner, funcs) = if let ComputeGraphNodeVariant::ElementWise(op) = node_variant {
            (op.value, op.functions.clone())
        } else {
            return false;
        };

        if let Some(input_exec_idx) = self.get_input_node_in_exec_graph(input_inner) {
            if self.check_cached(graph, input_inner) {
                return false;
            }

            let input_variant = self.execution_graph[input_exec_idx].variant.clone();
            if let ComputeGraphNodeVariant::ElementWise(input_op) = input_variant {
                let mut new_funcs = input_op.functions.functions.clone();
                new_funcs.extend(funcs.functions.iter().cloned());
                let new_functions =
                    ElementWiseFunctions::new(new_funcs, input_op.functions.input_datatype());

                let new_op = ElementWiseOperation::from_element_wise(
                    input_op.value,
                    new_functions,
                    input_op.shape(),
                );

                self.execution_graph[node_idx].variant =
                    ComputeGraphNodeVariant::ElementWise(new_op.clone());

                let input_of_input_inner = input_op.value;
                if let Some(input_of_input_exec) =
                    self.get_input_node_in_exec_graph(input_of_input_inner)
                {
                    self.execution_graph
                        .add_edge(input_of_input_exec, node_idx, ());
                }

                if let Some(edge) = self.execution_graph.find_edge(input_exec_idx, node_idx) {
                    self.execution_graph.remove_edge(edge);
                }
                self.add_physical_dependencies(graph, node_idx, &[input_of_input_inner]);
                self.remove_node_if_dead(input_exec_idx);
                return true;
            }
        }
        false
    }

    fn try_fuse_into_reduce(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();
        let (input_inner, funcs) = if let ComputeGraphNodeVariant::ElementWise(op) = node_variant {
            (op.value, op.functions.clone())
        } else {
            return false;
        };

        if let Some(input_exec_idx) = self.get_input_node_in_exec_graph(input_inner) {
            if self.check_cached(graph, input_inner) {
                return false;
            }

            let input_variant = self.execution_graph[input_exec_idx].variant.clone();
            if let ComputeGraphNodeVariant::Reduce(reduce_op) = input_variant {
                let mut new_reduce = reduce_op.clone();
                let mut existing_post = new_reduce.post_element_wise.functions.clone();
                existing_post.extend(funcs.functions.iter().cloned());
                new_reduce.post_element_wise = ElementWiseFunctions::new(
                    existing_post,
                    reduce_op.post_element_wise.input_datatype(),
                );

                self.execution_graph[node_idx].variant =
                    ComputeGraphNodeVariant::Reduce(new_reduce.clone());

                let reduce_input_inner = reduce_op.value;
                if let Some(reduce_input_exec) =
                    self.get_input_node_in_exec_graph(reduce_input_inner)
                {
                    self.execution_graph
                        .add_edge(reduce_input_exec, node_idx, ());
                }

                if let Some(edge) = self.execution_graph.find_edge(input_exec_idx, node_idx) {
                    self.execution_graph.remove_edge(edge);
                }
                self.add_physical_dependencies(graph, node_idx, &[reduce_input_inner]);
                self.remove_node_if_dead(input_exec_idx);
                return true;
            }
        }
        false
    }

    fn try_fuse_into_matmul(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();

        // Post-op
        if let ComputeGraphNodeVariant::ElementWise(op) = &node_variant {
            let input_inner = op.value;
            let funcs = op.functions.clone();

            if let Some(input_exec_idx) = self.get_input_node_in_exec_graph(input_inner) {
                if !self.check_cached(graph, input_inner) {
                    let input_variant = self.execution_graph[input_exec_idx].variant.clone();
                    if let ComputeGraphNodeVariant::MatMul(matmul_op) = input_variant {
                        let mut new_matmul = matmul_op.clone();
                        let mut existing_post = new_matmul.post_element_wise.functions.clone();
                        existing_post.extend(funcs.functions.iter().cloned());
                        new_matmul.post_element_wise = ElementWiseFunctions::new(
                            existing_post,
                            matmul_op.post_element_wise.input_datatype(),
                        );

                        self.execution_graph[node_idx].variant =
                            ComputeGraphNodeVariant::MatMul(new_matmul.clone());

                        let first_inner = matmul_op.first;
                        let second_inner = matmul_op.second;

                        if let Some(idx) = self.get_input_node_in_exec_graph(first_inner) {
                            self.execution_graph.add_edge(idx, node_idx, ());
                        }
                        if let Some(idx) = self.get_input_node_in_exec_graph(second_inner) {
                            self.execution_graph.add_edge(idx, node_idx, ());
                        }

                        if let Some(edge) = self.execution_graph.find_edge(input_exec_idx, node_idx)
                        {
                            self.execution_graph.remove_edge(edge);
                        }
                        self.add_physical_dependencies(
                            graph,
                            node_idx,
                            &[first_inner, second_inner],
                        );
                        self.remove_node_if_dead(input_exec_idx);
                        return true;
                    }
                }
            }
        }

        // Pre-op
        if let ComputeGraphNodeVariant::MatMul(matmul_op) = &node_variant {
            let first_inner = matmul_op.first;
            let second_inner = matmul_op.second;
            let mut changed_any = false;
            let mut new_matmul = matmul_op.clone();

            if let Some(first_exec_idx) = self.get_input_node_in_exec_graph(first_inner) {
                if !self.check_cached(graph, first_inner) {
                    let first_variant = self.execution_graph[first_exec_idx].variant.clone();
                    if let ComputeGraphNodeVariant::ElementWise(el_op) = first_variant {
                        new_matmul.first = el_op.value;
                        let mut new_funcs = el_op.functions.functions.clone();
                        new_funcs.extend(new_matmul.pre_element_wise[0].functions.iter().cloned());
                        new_matmul.pre_element_wise[0] =
                            ElementWiseFunctions::new(new_funcs, el_op.input_datatype());
                        changed_any = true;
                    }
                }
            }

            if let Some(second_exec_idx) = self.get_input_node_in_exec_graph(second_inner) {
                if !self.check_cached(graph, second_inner) {
                    let second_variant = self.execution_graph[second_exec_idx].variant.clone();
                    if let ComputeGraphNodeVariant::ElementWise(el_op) = second_variant {
                        new_matmul.second = el_op.value;
                        let mut new_funcs = el_op.functions.functions.clone();
                        new_funcs.extend(new_matmul.pre_element_wise[1].functions.iter().cloned());
                        new_matmul.pre_element_wise[1] =
                            ElementWiseFunctions::new(new_funcs, el_op.input_datatype());
                        changed_any = true;
                    }
                }
            }

            if changed_any {
                self.execution_graph[node_idx].variant =
                    ComputeGraphNodeVariant::MatMul(new_matmul.clone());

                if new_matmul.first != matmul_op.first {
                    let old_first_exec =
                        self.get_input_node_in_exec_graph(matmul_op.first).unwrap();
                    if let Some(edge) = self.execution_graph.find_edge(old_first_exec, node_idx) {
                        self.execution_graph.remove_edge(edge);
                    }
                    if let Some(new_first_exec) =
                        self.get_input_node_in_exec_graph(new_matmul.first)
                    {
                        self.execution_graph.add_edge(new_first_exec, node_idx, ());
                    }
                    self.remove_node_if_dead(old_first_exec);
                }

                if new_matmul.second != matmul_op.second {
                    let old_second_exec =
                        self.get_input_node_in_exec_graph(matmul_op.second).unwrap();
                    if let Some(edge) = self.execution_graph.find_edge(old_second_exec, node_idx) {
                        self.execution_graph.remove_edge(edge);
                    }
                    if let Some(new_second_exec) =
                        self.get_input_node_in_exec_graph(new_matmul.second)
                    {
                        self.execution_graph.add_edge(new_second_exec, node_idx, ());
                    }
                    self.remove_node_if_dead(old_second_exec);
                }
                self.add_physical_dependencies(
                    graph,
                    node_idx,
                    &[new_matmul.first, new_matmul.second],
                );
                return true;
            }
        }

        false
    }

    fn try_fuse_into_dequantize(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();
        let (input_inner, funcs) = if let ComputeGraphNodeVariant::ElementWise(op) = node_variant {
            (op.value, op.functions.clone())
        } else {
            return false;
        };

        if let Some(input_exec_idx) = self.get_input_node_in_exec_graph(input_inner) {
            if self.check_cached(graph, input_inner) {
                return false;
            }

            let input_variant = self.execution_graph[input_exec_idx].variant.clone();
            if let ComputeGraphNodeVariant::Dequantize(deq_op) = input_variant {
                let mut new_deq = deq_op.clone();
                let mut existing_post = new_deq.post_dequantize.functions.clone();
                existing_post.extend(funcs.functions.iter().cloned());
                new_deq.post_dequantize = ElementWiseFunctions::new(existing_post, deq_op.datatype);

                self.execution_graph[node_idx].variant =
                    ComputeGraphNodeVariant::Dequantize(new_deq);

                if let Some(edge) = self.execution_graph.find_edge(input_exec_idx, node_idx) {
                    self.execution_graph.remove_edge(edge);
                }
                self.remove_node_if_dead(input_exec_idx);
                return true;
            }
        }
        false
    }

    fn try_fuse_into_index_select(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();

        if let ComputeGraphNodeVariant::IndexSelect(op) = &node_variant {
            let mut new_op = op.clone();
            let mut changed = false;

            if let Some(input_exec) = self.get_input_node_in_exec_graph(op.input) {
                if !self.check_cached(graph, op.input) {
                    let input_variant = self.execution_graph[input_exec].variant.clone();
                    if let ComputeGraphNodeVariant::ElementWise(el_op) = input_variant {
                        new_op.input = el_op.value;
                        let mut new_funcs = el_op.functions.functions.clone();
                        new_funcs.extend(new_op.pre_element_wise_input.functions.iter().cloned());
                        new_op.pre_element_wise_input =
                            ElementWiseFunctions::new(new_funcs, el_op.input_datatype());
                        changed = true;
                    }
                }
            }

            if let Some(indexes_exec) = self.get_input_node_in_exec_graph(op.indexes) {
                if !self.check_cached(graph, op.indexes) {
                    let indexes_variant = self.execution_graph[indexes_exec].variant.clone();
                    if let ComputeGraphNodeVariant::ElementWise(el_op) = indexes_variant {
                        new_op.indexes = el_op.value;
                        let mut new_funcs = el_op.functions.functions.clone();
                        new_funcs.extend(new_op.pre_element_wise_indexes.functions.iter().cloned());
                        new_op.pre_element_wise_indexes =
                            ElementWiseFunctions::new(new_funcs, el_op.input_datatype());
                        changed = true;
                    }
                }
            }

            if changed {
                self.execution_graph[node_idx].variant =
                    ComputeGraphNodeVariant::IndexSelect(new_op.clone());
                if new_op.input != op.input {
                    let old = self.get_input_node_in_exec_graph(op.input).unwrap();
                    if let Some(edge) = self.execution_graph.find_edge(old, node_idx) {
                        self.execution_graph.remove_edge(edge);
                    }
                    if let Some(new) = self.get_input_node_in_exec_graph(new_op.input) {
                        self.execution_graph.add_edge(new, node_idx, ());
                    }
                    self.remove_node_if_dead(old);
                }
                if new_op.indexes != op.indexes {
                    let old = self.get_input_node_in_exec_graph(op.indexes).unwrap();
                    if let Some(edge) = self.execution_graph.find_edge(old, node_idx) {
                        self.execution_graph.remove_edge(edge);
                    }
                    if let Some(new) = self.get_input_node_in_exec_graph(new_op.indexes) {
                        self.execution_graph.add_edge(new, node_idx, ());
                    }
                    self.remove_node_if_dead(old);
                }
                self.add_physical_dependencies(graph, node_idx, &[new_op.input, new_op.indexes]);
                return true;
            }
        }

        if let ComputeGraphNodeVariant::ElementWise(el_op) = &node_variant {
            if let Some(input_exec) = self.get_input_node_in_exec_graph(el_op.value) {
                if !self.check_cached(graph, el_op.value) {
                    let input_variant = self.execution_graph[input_exec].variant.clone();
                    if let ComputeGraphNodeVariant::IndexSelect(idx_op) = input_variant {
                        let mut new_idx_op = idx_op.clone();
                        let mut new_funcs = new_idx_op.pre_element_wise_input.functions.clone();
                        new_funcs.extend(el_op.functions.functions.iter().cloned());
                        new_idx_op.pre_element_wise_input =
                            ElementWiseFunctions::new(new_funcs, idx_op.input_datatype());

                        self.execution_graph[node_idx].variant =
                            ComputeGraphNodeVariant::IndexSelect(new_idx_op.clone());

                        if let Some(idx) = self.get_input_node_in_exec_graph(idx_op.input) {
                            self.execution_graph.add_edge(idx, node_idx, ());
                        }
                        if let Some(idx) = self.get_input_node_in_exec_graph(idx_op.indexes) {
                            self.execution_graph.add_edge(idx, node_idx, ());
                        }

                        if let Some(edge) = self.execution_graph.find_edge(input_exec, node_idx) {
                            self.execution_graph.remove_edge(edge);
                        }
                        self.add_physical_dependencies(
                            graph,
                            node_idx,
                            &[idx_op.input, idx_op.indexes],
                        );
                        self.remove_node_if_dead(input_exec);
                        return true;
                    }
                }
            }
        }

        false
    }

    fn should_extend_kernel(&mut self, _: Vec<MirValue>, _: &[Vec<MirValue>]) -> bool {
        // TODO: Restore with better testing. This passes all tests in fusor, but breaks rbert and rwhisper
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn push_operation(
        &mut self,
        graph: &mut ComputeGraphInner,
        new_inputs: Vec<MirValue>,
        kernel: &mut GenericKernel,
        key: NodeIndex,
        operation: Arc<dyn Operation>,
        inputs: &mut Vec<Vec<MirValue>>,
        all_input_values: &mut Vec<KernelInputValue>,
        queued_operations: &mut Vec<(NodeIndex, Arc<dyn Operation>)>,
    ) {
        for input in &new_inputs {
            input.visit_input_values(|value| {
                if let Some(index) = all_input_values.iter().position(|x| *x == value) {
                    kernel.pre_register_binding(index as _);
                } else {
                    kernel.pre_register_binding(all_input_values.len() as _);
                    all_input_values.push(value.clone());
                }
            });
        }
        let result = operation.output(graph, &new_inputs);
        let MirValue::Tensor(resolved) = result else {
            panic!("Kernel input value is not a tensor");
        };
        // Cache the result
        graph.set_cached_result(key, resolved);
        inputs.push(new_inputs);
        queued_operations.push((key, operation));
    }

    fn flush_operations(
        &mut self,
        graph: &mut ComputeGraphInner,
        mut kernel: &mut GenericKernel,
        queued_operations: &[(NodeIndex, Arc<dyn Operation>)],
        inputs: &[Vec<MirValue>],
        all_input_values: &[KernelInputValue],
        workgroup_shape: workgroup_shape::WorkgroupShape,
    ) {
        let mut max_dispatch_size = [0; 3];
        for ((key, operation), inputs) in queued_operations.iter().zip(inputs) {
            // Map layout isn't really a kernel. Skip it
            if let Some(node) = graph.nodes.nodes.node_weight(*key)
                && matches!(node.variant, ComputeGraphNodeVariant::MapLayout(_))
            {
                continue;
            }

            let dispatch_size = operation.dispatch_size(&workgroup_shape, inputs);
            for (new, max) in dispatch_size.iter().zip(max_dispatch_size.iter_mut()) {
                *max = (*max).max(*new);
            }
            if cfg!(debug_assertions) {
                writeln!(&mut kernel, "{{ // start {}", operation.name()).unwrap();
            } else {
                writeln!(&mut kernel, "{{").unwrap();
            }
            operation.build_kernel(graph, &workgroup_shape, inputs, kernel);
            let name = kernel.name_mut();
            if !name.is_empty() {
                *name += "->";
            }
            *name += &operation.name();
            if cfg!(debug_assertions) {
                writeln!(&mut kernel, "}} // end {}", operation.name()).unwrap();
            } else {
                writeln!(&mut kernel, "}}").unwrap();
            }
            // Check if that makes any of this node's dependencies dead
            let mut dependencies = Vec::new();
            graph.visit_dependencies(*key, &mut |dependent_key| {
                dependencies.push(dependent_key);
            });
            for dependency in dependencies {
                graph.check_life(dependency);
            }
        }
        kernel.set_workgroup_size(workgroup_shape);
        kernel.run(
            &graph.device,
            all_input_values,
            self.command_encoder,
            max_dispatch_size,
        );
    }

    /// Wrap an expression with element-wise functions (each becomes a unary Op node)
    fn wrap_with_element_wise_functions(
        &self,
        mut expr: NaryExpr,
        funcs: &ElementWiseFunctions,
    ) -> NaryExpr {
        let mut current_input_type = funcs.input_datatype();
        for func in funcs.functions.iter() {
            expr = NaryExpr::Op {
                children: vec![expr],
                function: func.to_nary_function(current_input_type),
            };
            current_input_type = func.datatype;
        }
        expr
    }
}
