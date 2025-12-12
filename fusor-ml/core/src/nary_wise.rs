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
    /// Leaf node - index into inputs array
    Input(usize),
    /// Operation with N children (supports unary, binary, or more)
    Op {
        children: Vec<NaryExpr>,
        function: NaryFunction,
    },
}

impl NaryExpr {
    /// Get the name of the expression for debugging
    pub fn name(&self) -> String {
        match self {
            NaryExpr::Input(idx) => format!("input_{}", idx),
            NaryExpr::Op { children, function } => {
                let child_names: Vec<_> = children.iter().map(|c| c.name()).collect();
                format!("{}({})", function.name(), child_names.join(","))
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
        temp_counter: &mut usize,
        functions_cache: &mut Vec<(String, Function)>,
    ) -> String {
        match expr {
            NaryExpr::Input(idx) => input_values[*idx].clone(),
            NaryExpr::Op { children, function } => {
                // Recursively evaluate all children
                let child_values: Vec<String> = children
                    .iter()
                    .map(|child| {
                        self.generate_expr_code(
                            child,
                            kernel,
                            input_values,
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
        }
    }

    /// Attempt to convert this NaryOperation into an ElementwiseOperation. This will only succeed
    /// if there is only a single input to the operation
    pub(crate) fn try_into_elementwise_op(&self) -> Option<ElementWiseOperation> {
        if self.inputs.len() == 1 {
            let output_datatype = self.output_datatype;
            let value = self.inputs[0];
            let input_datatype = match self.expression {
                NaryExpr::Input(_) => output_datatype,
                NaryExpr::Op { ref function, .. } => function.input_types[0],
            };

            fn collect_functions(
                expr: &NaryExpr,
                function_body: &mut String,
                out_id: &mut usize,
            ) -> std::fmt::Result {
                let this_output = *out_id;
                match expr {
                    NaryExpr::Input(_) => {
                        writeln!(function_body, "let output_{this_output} = input;",)
                    }
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

        // Collect datatypes for all inputs
        let datatypes: Vec<_> = inputs
            .iter()
            .filter_map(|input| {
                let data: MaybeQData = input.clone().try_into().ok()?;
                Some(data.datatype())
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
                let output_index = &indexes[output_tensor_index];
                let output_tensor = &tensors[output_tensor_index];

                let mut temp_counter = 0;

                // Generate expression tree evaluation
                let result = self.generate_expr_code(
                    &self.expression,
                    kernel,
                    &input_values,
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
