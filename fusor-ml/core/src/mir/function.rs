#[derive(Clone, Debug)]
pub(crate) struct Function {
    id: u32,
    ty: String,
    body: String,
    inputs: Vec<(String, String)>,
}

impl Function {
    pub(crate) fn new(id: u32, ty: String, body: String, inputs: Vec<(String, String)>) -> Self {
        Self {
            id,
            ty,
            body,
            inputs,
        }
    }

    pub(crate) fn function_definition(&self) -> String {
        let name = self.function_name();
        let inputs = &self.inputs;
        let mut inputs_string = String::new();
        for (name, ty) in inputs {
            inputs_string.push_str(&format!("{name}: {ty}, "));
        }
        for _ in 0..2 {
            inputs_string.pop();
        }
        let body = &self.body;
        let ty = &self.ty;
        format!("fn {name}({inputs_string}) -> {ty} {{ {body} return output; }}")
    }

    fn function_name(&self) -> String {
        format!("f_{}", self.id)
    }

    pub(crate) fn call(&self, inputs: Vec<String>) -> String {
        format!("{}({})", self.function_name(), inputs.join(", "))
    }

    #[allow(dead_code)]
    pub(crate) fn call_inlined(&self, inputs: Vec<String>) -> String {
        let mut output = String::new();
        output.push_str("{\n");
        for (i_value, (i_name, ty)) in inputs.iter().zip(&self.inputs) {
            output.push_str(&format!("let {i_name}: {ty} = {i_value};\n"));
        }
        output.push_str("}\n");
        output
    }
}
