//! Schedule independent expression operands without changing their arithmetic.
//! Larger temporary trees run first so their scalar results, rather than every
//! small sibling operand, remain live while the rest of an expression is built.
use fusor_ir::ir::kernel::{ElementType, TileExpr, TileExprKind as K};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

#[derive(Clone, Copy)]
struct Pressure {
    peak: u32,
    result: u32,
}

impl Pressure {
    fn priority(self) -> u32 {
        self.peak.saturating_sub(self.result)
    }
}

/// Per-emission, bounded estimates over the shared expression DAG. Collectives
/// and cooperative operations are ordering boundaries, including descendants.
#[derive(Default)]
pub(super) struct ExpressionOrder {
    estimates: FxHashMap<TileExpr, Option<Pressure>>,
}

impl ExpressionOrder {
    pub(super) fn operands<const N: usize>(&mut self, operands: [&TileExpr; N]) -> [usize; N] {
        let mut order = std::array::from_fn(|i| i);
        let mut priorities = [0; N];
        for (i, operand) in operands.into_iter().enumerate() {
            let Some(pressure) = self.estimate(operand, 0) else {
                return order;
            };
            priorities[i] = pressure.priority();
        }
        // Stable ties preserve source order. Handles are placed back in their
        // original argument positions; even noncommutative operators are safe.
        order.sort_by_key(|i| std::cmp::Reverse(priorities[*i]));
        order
    }

    fn estimate(&mut self, value: &TileExpr, depth: u32) -> Option<Pressure> {
        if let Some(found) = self.estimates.get(value) {
            return *found;
        }
        if depth >= 128 || self.estimates.len() >= 16_384 {
            return None;
        }
        let estimate = (|| {
            if matches!(
                value.kind(),
                K::Reduce { .. } | K::CoopLoad { .. } | K::CoopMma { .. }
            ) {
                return None;
            }
            let result = match value.element() {
                ElementType::Scalar(_) => 1,
                ElementType::Vector { lanes, .. } => lanes,
                ElementType::CoopMatrix { .. } => return None,
            };
            if matches!(value.kind(), K::Literal(_)) {
                return Some(Pressure { peak: 0, result: 0 });
            }
            let mut children = SmallVec::<[Pressure; 4]>::new();
            let mut independent = true;
            value.kind().visit_children(&mut |child| {
                if let Some(pressure) = self.estimate(child, depth + 1) {
                    children.push(pressure);
                } else {
                    independent = false;
                }
            });
            if !independent {
                return None;
            }
            children.sort_by_key(|p| std::cmp::Reverse(p.priority()));
            let mut peak = result;
            let mut live = 0u32;
            for child in children {
                peak = peak.max(live.saturating_add(child.peak));
                live = live.saturating_add(child.result);
            }
            Some(Pressure {
                peak: peak.max(live),
                result,
            })
        })();
        self.estimates.insert(value.clone(), estimate);
        estimate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor_ir::ir::kernel::{ReduceKind, ScalarElement, TileReduceOp};
    use fusor_tile::build::Kernel;

    #[test]
    fn collective_descendants_preserve_operand_order() {
        let b = Kernel::new();
        let x = b.load_local(b.local(ScalarElement::F32.element()));
        let collective = b.reduce(TileReduceOp::Sum, ReduceKind::Subgroup, x.clone());
        let nested = b.add(b.f32(1.), collective);
        let mut order = ExpressionOrder::default();
        assert_eq!(order.operands([&x, &nested]), [0, 1]);
        assert_eq!(order.operands([&nested, &x]), [0, 1]);
    }
}
