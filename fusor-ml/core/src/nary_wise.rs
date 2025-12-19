use std::fmt::Write;

use crate::{
    ElementWiseFunction, ElementWiseOperation,
    compute_graph::{ComputeGraphInner, NodeIndex},
    layout::TILE_SIZE,
    mir::{function::Function, inputs::MirValue, kernel::GenericKernel, operation::Operation},
    tensor::{DataTypeEnum, TensorData},
    visit_tiled::{
        MaybeQData, build_visit_tiled_kernel, titled_map_dispatch_size,
        titled_map_workgroup_size_constraints,
    },
};

/// A function that can be applied in the expression tree.
/// Supports any arity (unary, binary, etc.)
#[derive(Clone, Debug)]
pub(crate) struct NaryFunction {
    pub(crate) name: Option<String>,
    /// WGSL code, e.g. "let output = a + b;" or "let output = sin(input);"
    pub(crate) operation: String,
    /// Input parameter names, e.g. ["a", "b"] for binary, ["input"] for unary
    pub(crate) input_names: Vec<String>,
    pub(crate) input_types: Vec<DataTypeEnum>,
    pub(crate) output_type: DataTypeEnum,
}

impl NaryFunction {
    pub fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("op")
    }
}


/// Expression tree node supporting any arity operations
#[derive(Clone, Debug)]
pub(crate) enum NaryExpr {
    /// Operation with N children (supports unary, binary, or more)
    Op {
        children: Vec<NaryExpr>,
        function: NaryFunction,
    },
    /// Index into input tensor using computed index expression
    /// The index expression should evaluate to an array of indices (one per dimension)
    IndexedInput {
        /// Which input to access (index into inputs array)
        input_idx: usize,
        /// Expression that computes the index (should evaluate to array of indices)
        index: Box<NaryExpr>,
    },
    /// Get current output index as array of all dimension indices
    /// For a 3D tensor, evaluates to vec3<u32>(dim_0, dim_1, dim_2)
    Dim,
    /// Extract single component from array expression
    /// Used to get a specific dimension index from Dim
    Component {
        /// Which component to extract (0-indexed)
        component: usize,
        /// Expression that evaluates to an array
        expr: Box<NaryExpr>,
    },
}

impl NaryExpr {
    /// Create an input expression that accesses at the current dimension indices
    pub fn input(input_idx: usize, _output_type: DataTypeEnum) -> Self {
        NaryExpr::IndexedInput {
            input_idx,
            index: Box::new(NaryExpr::Dim),
        }
    }

    /// Create an expression that gets a single dimension index
    pub fn dim_index(dim_idx: usize) -> Self {
        NaryExpr::Component {
            component: dim_idx,
            expr: Box::new(NaryExpr::Dim),
        }
    }

    /// Create a select expression (ternary operator)
    /// Semantics: condition != 0 ? on_true : on_false
    pub fn select(
        condition: NaryExpr,
        on_true: NaryExpr,
        on_false: NaryExpr,
        condition_type: DataTypeEnum,
        output_type: DataTypeEnum,
    ) -> NaryExpr {
        NaryExpr::Op {
            children: vec![condition, on_true, on_false],
            function: NaryFunction {
                name: Some("select".to_string()),
                operation: format!(
                    "let output = select(on_false, on_true, condition != {}(0));",
                    condition_type
                ),
                input_names: vec![
                    "condition".to_string(),
                    "on_true".to_string(),
                    "on_false".to_string(),
                ],
                input_types: vec![condition_type, output_type, output_type],
                output_type,
            },
        }
    }

    /// Create a multiplication expression: a * b
    pub fn mul(a: NaryExpr, b: NaryExpr, datatype: DataTypeEnum) -> NaryExpr {
        NaryExpr::Op {
            children: vec![a, b],
            function: NaryFunction {
                name: Some("mul".to_string()),
                operation: "let output = a * b;".to_string(),
                input_names: vec!["a".to_string(), "b".to_string()],
                input_types: vec![datatype, datatype],
                output_type: datatype,
            },
        }
    }

    /// Create an addition expression: a + b
    pub fn add(a: NaryExpr, b: NaryExpr, datatype: DataTypeEnum) -> NaryExpr {
        NaryExpr::Op {
            children: vec![a, b],
            function: NaryFunction {
                name: Some("add".to_string()),
                operation: "let output = a + b;".to_string(),
                input_names: vec!["a".to_string(), "b".to_string()],
                input_types: vec![datatype, datatype],
                output_type: datatype,
            },
        }
    }

    /// Create a negation expression: -a
    pub fn neg(a: NaryExpr, datatype: DataTypeEnum) -> NaryExpr {
        NaryExpr::Op {
            children: vec![a],
            function: NaryFunction {
                name: Some("neg".to_string()),
                operation: "let output = -input;".to_string(),
                input_names: vec!["input".to_string()],
                input_types: vec![datatype],
                output_type: datatype,
            },
        }
    }

    /// Create a custom unary operation
    pub fn unary_op(
        a: NaryExpr,
        name: &str,
        operation: impl Into<String>,
        input_type: DataTypeEnum,
        output_type: DataTypeEnum,
    ) -> NaryExpr {
        NaryExpr::Op {
            children: vec![a],
            function: NaryFunction {
                name: Some(name.to_string()),
                operation: operation.into(),
                input_names: vec!["input".to_string()],
                input_types: vec![input_type],
                output_type,
            },
        }
    }

    /// Create an index vector (vec2, vec3, or vec4) from dimension components
    pub fn make_index(components: Vec<NaryExpr>) -> NaryExpr {
        let rank = components.len();
        let (operation, input_names) = match rank {
            2 => (
                "let output = vec2<u32>(a, b);".to_string(),
                vec!["a".to_string(), "b".to_string()],
            ),
            3 => (
                "let output = vec3<u32>(a, b, c);".to_string(),
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
            ),
            4 => (
                "let output = vec4<u32>(a, b, c, d);".to_string(),
                vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()],
            ),
            _ => panic!("Unsupported rank {} for make_index", rank),
        };

        NaryExpr::Op {
            children: components,
            function: NaryFunction {
                name: Some(format!("make_vec{}", rank)),
                operation,
                input_names,
                input_types: vec![DataTypeEnum::U32; rank],
                output_type: DataTypeEnum::U32,
            },
        }
    }

    /// Create an indexed input access with a custom index expression
    pub fn indexed_input(input_idx: usize, index: NaryExpr) -> NaryExpr {
        NaryExpr::IndexedInput {
            input_idx,
            index: Box::new(index),
        }
    }

    /// Get the name of the expression for debugging
    pub fn name(&self) -> String {
        match self {
            NaryExpr::Op { children, function } => {
                let child_names: Vec<_> = children.iter().map(|c| c.name()).collect();
                format!("{}({})", function.name(), child_names.join(","))
            }
            NaryExpr::IndexedInput { input_idx, index } => {
                format!("input_{}[{}]", input_idx, index.name())
            }
            NaryExpr::Dim => "dim".to_string(),
            NaryExpr::Component { component, expr } => {
                format!("{}.{}", expr.name(), component)
            }
        }
    }
}

/// N-ary operation combining multiple inputs with arbitrary operations.
/// Can fuse chains of element-wise and pair-wise operations into a single kernel.
#[derive(Clone, Debug)]
pub(crate) struct NaryOperation {
    /// Input tensors (leaves of expression tree)
    pub(crate) inputs: Vec<NodeIndex>,
    /// Expression tree describing computation (includes all operations)
    pub(crate) expression: NaryExpr,
    pub(crate) shape: Box<[usize]>,
    pub(crate) output_datatype: DataTypeEnum,
}

impl NaryOperation {
    /// Generate WGSL code for evaluating the expression tree
    fn generate_expr_code(
        &self,
        expr: &NaryExpr,
        kernel: &mut GenericKernel,
        input_values: &[String],
        input_tensors: &[crate::visit_tiled::MaybeQTensorInput],
        current_dims: &[String],
        temp_counter: &mut usize,
        functions_cache: &mut Vec<(String, Function)>,
    ) -> String {
        match expr {
            NaryExpr::Op { children, function } => {
                // Recursively evaluate all children
                let child_values: Vec<String> = children
                    .iter()
                    .map(|child| {
                        self.generate_expr_code(
                            child,
                            kernel,
                            input_values,
                            input_tensors,
                            current_dims,
                            temp_counter,
                            functions_cache,
                        )
                    })
                    .collect();

                // Check if we already have this function cached
                let func = if let Some((_, cached_func)) = functions_cache
                    .iter()
                    .find(|(op, _)| *op == function.operation)
                {
                    cached_func.clone()
                } else {
                    // Create the function with proper input names and types
                    let func = kernel.add_function(
                        function.output_type,
                        function.operation.clone(),
                        function
                            .input_names
                            .iter()
                            .zip(&function.input_types)
                            .map(|(name, ty)| (name.clone(), ty.to_string())),
                    );
                    functions_cache.push((function.operation.clone(), func.clone()));
                    func
                };

                // Generate temp variable for result
                let temp_name = format!("tmp_{}", *temp_counter);
                *temp_counter += 1;

                // Call function with child values
                writeln!(kernel, "let {temp_name} = {};", func.call(child_values)).unwrap();

                temp_name
            }
            NaryExpr::IndexedInput { input_idx, index } => {
                use crate::visit_tiled::MaybeQTensorInput;

                // Check if index is just Dim (equivalent to old Normal mapping)
                if matches!(index.as_ref(), NaryExpr::Dim) {
                    // Normal mapping - just use the pre-computed value
                    input_values[*input_idx].clone()
                } else {
                    // Get the rank of the input tensor
                    let input_rank = match &input_tensors[*input_idx] {
                        MaybeQTensorInput::Tensor(t) => t.rank() as usize,
                        MaybeQTensorInput::QTensor(q) => q.rank as usize,
                    };

                    // Extract dimension expressions from the index tree
                    // If it's an Op that makes a vec, we can directly use its children
                    let dims: Vec<String> = self.extract_index_dimensions(
                        index,
                        input_rank,
                        kernel,
                        input_values,
                        input_tensors,
                        current_dims,
                        temp_counter,
                        functions_cache,
                    );

                    // Generate strided index call with computed dimension expressions
                    let custom_idx_var = format!("custom_idx_{}", *temp_counter);
                    *temp_counter += 1;

                    write!(kernel, "let {} = ", custom_idx_var).unwrap();
                    match &input_tensors[*input_idx] {
                        MaybeQTensorInput::Tensor(t) => {
                            t.strided_index(kernel, dims);
                        }
                        MaybeQTensorInput::QTensor(_) => {
                            panic!("Custom indexing not supported for quantized tensors");
                        }
                    }
                    writeln!(kernel, ";").unwrap();

                    // Return the custom indexed value
                    format!("{}[{}]", input_tensors[*input_idx], custom_idx_var)
                }
            }
            NaryExpr::Dim => {
                // Dim evaluates to an array of all current dimension indices
                // We need to store it in a temp variable because WGSL doesn't allow
                // indexing directly into constructor expressions
                let rank = current_dims.len();
                let temp_name = format!("dim_vec_{}", *temp_counter);
                *temp_counter += 1;

                if rank <= 4 {
                    // Use WGSL vec types for small ranks
                    writeln!(
                        kernel,
                        "let {} = vec{}<u32>({});",
                        temp_name,
                        rank,
                        current_dims.join(", ")
                    )
                    .unwrap();
                } else {
                    // Use array for larger ranks
                    writeln!(
                        kernel,
                        "let {} = array<u32, {}>({});",
                        temp_name,
                        rank,
                        current_dims.join(", ")
                    )
                    .unwrap();
                }

                temp_name
            }
            NaryExpr::Component { component, expr } => {
                // First evaluate the expression to get the array
                let array_val = self.generate_expr_code(
                    expr,
                    kernel,
                    input_values,
                    input_tensors,
                    current_dims,
                    temp_counter,
                    functions_cache,
                );

                // Then extract the component
                format!("{}[{}u]", array_val, component)
            }
        }
    }

    /// Extract dimension expressions from an index expression tree.
    /// This avoids creating intermediate vec types by directly extracting each dimension
    /// as a separate WGSL expression.
    fn extract_index_dimensions(
        &self,
        index: &NaryExpr,
        expected_rank: usize,
        kernel: &mut GenericKernel,
        input_values: &[String],
        input_tensors: &[crate::visit_tiled::MaybeQTensorInput],
        current_dims: &[String],
        temp_counter: &mut usize,
        functions_cache: &mut Vec<(String, Function)>,
    ) -> Vec<String> {
        match index {
            // If the index is an Op that constructs a vec (name starts with "make_vec"),
            // directly use its children as dimension expressions
            NaryExpr::Op { children, function }
                if function.name.as_ref().is_some_and(|n| n.starts_with("make_vec")) =>
            {
                assert_eq!(
                    children.len(),
                    expected_rank,
                    "Index vec has wrong number of dimensions: expected {}, got {}",
                    expected_rank,
                    children.len()
                );
                children
                    .iter()
                    .map(|child| {
                        self.generate_expr_code(
                            child,
                            kernel,
                            input_values,
                            input_tensors,
                            current_dims,
                            temp_counter,
                            functions_cache,
                        )
                    })
                    .collect()
            }
            // If the index is Dim, use the current dimension variables directly
            NaryExpr::Dim => current_dims.to_vec(),
            // For other cases, evaluate the index and extract each component
            _ => {
                let index_val = self.generate_expr_code(
                    index,
                    kernel,
                    input_values,
                    input_tensors,
                    current_dims,
                    temp_counter,
                    functions_cache,
                );
                (0..expected_rank)
                    .map(|i| format!("{}[{}u]", index_val, i))
                    .collect()
            }
        }
    }

    /// Attempt to convert this NaryOperation into an ElementwiseOperation. This will only succeed
    /// if there is only a single input to the operation
    pub(crate) fn try_into_elementwise_op(&self) -> Option<ElementWiseOperation> {
        if self.inputs.len() == 1 {
            let output_datatype = self.output_datatype;
            let value = self.inputs[0];
            let input_datatype = match &self.expression {
                NaryExpr::Op { function, .. } => function.input_types[0],
                NaryExpr::IndexedInput { .. } => {
                    // For ElementWise, we need the datatype from the first input
                    // This is a limitation - we can't easily get it without graph access
                    output_datatype
                }
                NaryExpr::Dim => {
                    panic!("Dim cannot be the root expression for ElementWise conversion");
                }
                NaryExpr::Component { .. } => {
                    panic!("Component cannot be the root expression for ElementWise conversion");
                }
            };

            fn collect_functions(
                expr: &NaryExpr,
                function_body: &mut String,
                out_id: &mut usize,
            ) -> std::fmt::Result {
                let this_output = *out_id;
                match expr {
                    NaryExpr::Op { children, function } => {
                        let mut inputs = Vec::new();
                        for child in children {
                            *out_id += 1;
                            inputs.push(*out_id);
                            collect_functions(child, function_body, out_id)?;
                        }
                        // TODO: this isn't a great way to handle this. Refactor once we get rid of elementwise and pairwise ops
                        let default_value = match function.output_type {
                            DataTypeEnum::F32 => "0.0",
                            DataTypeEnum::F16 => "f16(0.0)",
                            DataTypeEnum::U32 => "0u",
                        };
                        writeln!(function_body, "var output_{this_output} = {default_value};",)?;
                        writeln!(function_body, "{{",)?;
                        for (i, input_id) in inputs.iter().enumerate() {
                            writeln!(
                                function_body,
                                "    let {} = output_{};",
                                function.input_names[i], input_id
                            )?;
                        }
                        writeln!(function_body, "{}", function.operation)?;
                        writeln!(function_body, "    output_{} = output;", this_output,)?;
                        writeln!(function_body, "}}",)?;
                        Ok(())
                    }
                    NaryExpr::IndexedInput { index, .. } => {
                        // Only index = Dim (normal mapping) can be converted to ElementWise
                        if matches!(index.as_ref(), NaryExpr::Dim) {
                            writeln!(function_body, "let output_{this_output} = input;")
                        } else {
                            // Custom indexing cannot be converted to ElementWise
                            // (requires multi-dimensional indexing)
                            panic!(
                                "IndexedInput with custom index cannot be converted to ElementWise operation"
                            );
                        }
                    }
                    NaryExpr::Dim => {
                        // Dim cannot be converted to ElementWise
                        // (operates on indices, not tensor values)
                        panic!("Dim cannot be converted to ElementWise operation");
                    }
                    NaryExpr::Component { .. } => {
                        // Component cannot be converted to ElementWise
                        // (operates on indices, not tensor values)
                        panic!("Component cannot be converted to ElementWise operation");
                    }
                }
            }

            let mut function_body = String::new();
            let mut out_id = 0;
            collect_functions(&self.expression, &mut function_body, &mut out_id).unwrap();
            writeln!(function_body, "let output = output_0;").unwrap();
            let functions = ElementWiseFunction::new(function_body, output_datatype);

            let shape = self.shape.clone();
            Some(ElementWiseOperation::new(
                input_datatype,
                value,
                functions,
                shape,
            ))
        } else {
            None
        }
    }
}

impl Operation for NaryOperation {
    fn workgroup_shape_constraints(
        &self,
        device: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        titled_map_workgroup_size_constraints(&self.shape, device)
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _inputs: &[MirValue],
    ) -> [u32; 3] {
        titled_map_dispatch_size(TILE_SIZE, *workgroup_shape, &self.shape)
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        for input in &self.inputs {
            f(*input);
        }
    }

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue> {
        let mut mir_inputs: Vec<MirValue> = self
            .inputs
            .iter()
            .map(|idx| nodes.get_result_or_qmatrix(*idx).unwrap().into())
            .collect();

        // Check if we can reuse an input allocation for output
        let reuse_index = mir_inputs.iter().enumerate().find_map(|(i, input)| {
            if let Ok(data) = std::convert::TryInto::<MaybeQData>::try_into(input.clone())
                && data.datatype() == self.output_datatype.into()
                && data.owned()
                && !data.layout().allocation_overlaps()
            {
                return Some(i);
            }
            None
        });

        if reuse_index.is_none() {
            // Need to allocate a new output tensor
            let first_input: MaybeQData = mir_inputs[0].clone().try_into().unwrap();
            let output_tensor =
                TensorData::new_for_shape(first_input.device(), &self.shape, self.output_datatype);
            mir_inputs.push(output_tensor.into());
        }

        mir_inputs
    }

    fn output(&self, _nodes: &ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        // Check if we reused an input allocation
        let reuse_index = inputs[..self.inputs.len()]
            .iter()
            .enumerate()
            .find_map(|(i, input)| {
                if let Ok(data) = std::convert::TryInto::<MaybeQData>::try_into(input.clone())
                    && data.datatype() == self.output_datatype.into()
                    && data.owned()
                    && !data.layout().allocation_overlaps()
                {
                    return Some(i);
                }
                None
            });

        if let Some(idx) = reuse_index {
            inputs[idx].clone()
        } else {
            // Output is the last input (newly allocated)
            inputs.last().unwrap().clone()
        }
    }

    fn build_kernel(
        &self,
        graph: &ComputeGraphInner,
        _workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
        kernel: &mut GenericKernel,
    ) {
        // Determine output tensor index
        let reuse_index = inputs[..self.inputs.len()]
            .iter()
            .enumerate()
            .find_map(|(i, input)| {
                if let Ok(data) = std::convert::TryInto::<MaybeQData>::try_into(input.clone())
                    && data.datatype() == self.output_datatype.into()
                    && data.owned()
                    && !data.layout().allocation_overlaps()
                {
                    return Some(i);
                }
                None
            });

        let output_tensor_index = reuse_index.unwrap_or(self.inputs.len());

        // Collect datatypes and ranks for all inputs
        let datatypes: Vec<_> = inputs
            .iter()
            .filter_map(|input| {
                let data: MaybeQData = input.clone().try_into().ok()?;
                let datatype = data.datatype();
                let input_rank = data.layout().shape().len() as u32;
                let output_rank = self.shape.len() as u32;

                // Use DequantizedWithRank if the input has a different rank than the output
                if input_rank != output_rank {
                    // Extract DataTypeEnum from VisitTiledInputType
                    match datatype {
                        crate::visit_tiled::VisitTiledInputType::Dequantized(dt) => {
                            Some(crate::visit_tiled::VisitTiledInputType::DequantizedWithRank {
                                datatype: dt,
                                rank: input_rank,
                            })
                        }
                        _ => Some(datatype), // For quantized types, use the original
                    }
                } else {
                    Some(datatype)
                }
            })
            .collect();

        let mut functions_cache: Vec<(String, Function)> = Vec::new();

        build_visit_tiled_kernel(
            &graph.device,
            &self.shape,
            TILE_SIZE,
            datatypes,
            |kernel, indexes, tensors, values| {
                let input_values: Vec<_> = values[..self.inputs.len()].to_vec();
                let input_tensors = &tensors[..self.inputs.len()];
                let output_index = &indexes[output_tensor_index];
                let output_tensor = &tensors[output_tensor_index];

                let mut temp_counter = 0;

                // Extract dimension variables (dim_0, dim_1, ..., dim_N)
                let rank = self.shape.len();
                let current_dims: Vec<String> = (0..rank).map(|i| format!("dim_{}", i)).collect();

                // Generate expression tree evaluation
                let result = self.generate_expr_code(
                    &self.expression,
                    kernel,
                    &input_values,
                    input_tensors,
                    &current_dims,
                    &mut temp_counter,
                    &mut functions_cache,
                );

                format!("{output_tensor}[{output_index}] = {result};")
            },
            kernel,
        );
    }

    fn name(&self) -> String {
        format!(
            "nary_{}_{}",
            self.expression.name(),
            self.shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
    }
}
