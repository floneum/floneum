use std::fmt::Display;

use crate::mir::{
    expression::SharedExpression, kernel::GenericKernel, workgroup_shape::WorkgroupShape,
};

pub struct DispatchIndex {
    index: usize,
}

struct Dispatch {
    indexes: Vec<SharedExpression>,
}

impl Dispatch {
    fn new() -> Self {
        Self {
            indexes: Vec::new(),
        }
    }

    fn push(&mut self, index: impl Into<SharedExpression>) -> IndexBinding {
        let index_binding = self.indexes.len();
        self.indexes.push(index.into());
        IndexBinding::new(index_binding)
    }

    fn dispatch(&self, writer: &mut impl std::fmt::Write, kernel: &mut GenericKernel) {
        let global_id = kernel.global_id();
        let mut index = 0;
        for (i, size) in self.indexes.iter().enumerate() {
            let size_const = size.eval();
            let dispatch_index = IndexBinding::new(i);
            if size_const == 1 {
                writeln!(writer, "let {dispatch_index} = 0;").unwrap();
                continue;
            }
            let dim = ["x", "y"].get(index);
            index += 1;
            if let Some(dim) = dim {
                writeln!(writer, "let {dispatch_index} = {global_id}.{dim};").unwrap();
                continue;
            }
            if index == 3 {
                writeln!(writer, "var remaining_z = {global_id}.z;").unwrap();
            }
            if i == self.indexes.len() - 1 {
                writeln!(writer, "let {dispatch_index} = remaining_z;").unwrap();
                continue;
            }
            writeln!(writer, "let {dispatch_index} = remaining_z % {size};").unwrap();
            writeln!(writer, "remaining_z = remaining_z / {size};").unwrap();
        }
    }

    fn dispatch_size(&self, workgroup_shape: WorkgroupShape) -> [u32; 3] {
        std::array::from_fn(|i| {
            let raw = if i < 2 {
                self.indexes.get(i).map(|e| e.eval()).unwrap_or(1)
            } else {
                self.indexes
                    .iter()
                    .skip(2)
                    .map(|e| e.eval())
                    .product::<u32>()
            };
            raw / workgroup_shape.component(i)
        })
    }
}

pub struct IndexBinding {
    index: usize,
}

impl IndexBinding {
    pub fn new(index: usize) -> Self {
        Self { index }
    }
}

impl Display for IndexBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "dispatch_index_{}", self.index)
    }
}

#[test]
fn test_dispatch_visit() {
    let mut dispatch = Dispatch::new();
    dispatch.push(SharedExpression::constant(1));
    dispatch.push(SharedExpression::constant(10));
    dispatch.push(SharedExpression::constant(100));
    dispatch.push(SharedExpression::constant(1000));
    let mut writer = String::new();
    let mut kernel = GenericKernel::new();
    dispatch.dispatch(&mut writer, &mut kernel);
    println!("{writer}")
}
