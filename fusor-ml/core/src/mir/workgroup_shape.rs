// The shape of the workgroup. [x, y, z] where their product is <= 256.
//
// Kernels can be fused if their workgroup shape can be coerced. Coercion can happen if
// the biggest linearized workgroup shape is a multiple of all smaller workgroup shapes.

use lru::LruCache;
use parking_lot::RwLock;
use rustc_hash::FxBuildHasher;
use std::{num::NonZeroUsize, sync::OnceLock};

use crate::mir::kernel::GenericKernel;

const MAX_WORKGROUP_SIZE: u32 = 256;

#[derive(Debug, Clone, Copy)]
pub struct WorkgroupShape {
    shape: [u32; 3],
}

impl From<[u32; 3]> for WorkgroupShape {
    fn from(shape: [u32; 3]) -> Self {
        Self { shape }
    }
}

impl WorkgroupShape {
    pub(crate) fn new(x: u32, y: u32, z: u32) -> Self {
        assert!(
            x > 0 && y > 0 && z > 0,
            "Workgroup shape dimensions must be greater than zero"
        );
        assert!(
            x * y * z <= 256,
            "Workgroup shape dimensions must be less than or equal to 256"
        );
        Self { shape: [x, y, z] }
    }

    pub(crate) fn linearized(&self) -> u32 {
        self.shape.iter().product()
    }

    pub(crate) fn x(&self) -> u32 {
        self.shape[0]
    }

    pub(crate) fn y(&self) -> u32 {
        self.shape[1]
    }

    pub(crate) fn z(&self) -> u32 {
        self.shape[2]
    }

    pub(crate) fn shape(&self) -> [u32; 3] {
        self.shape
    }

    pub(crate) fn component(&self, i: usize) -> u32 {
        assert!(i < 3, "Index must be 0, 1, or 2");
        self.shape[i]
    }

    pub(crate) fn linearized_workgroup_index(&self, kernel: &mut GenericKernel) -> String {
        let mut merged = "0".to_string();
        let mut product = 1;
        for (component, real_size) in ["x", "y", "z"].iter().zip(self.shape()) {
            merged += &format!(
                " + {}.{} * {}",
                kernel.workgroup_index(),
                component,
                product
            );
            product *= real_size;
        }

        merged
    }
}

impl IntoIterator for WorkgroupShape {
    type Item = u32;
    type IntoIter = std::array::IntoIter<u32, 3>;

    fn into_iter(self) -> Self::IntoIter {
        self.shape.into_iter()
    }
}

#[derive(Default, Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct WorkgroupShapeConstraints {
    shape: [Vec<Constraint>; 3],
}

impl WorkgroupShapeConstraints {
    #[track_caller]
    pub(crate) fn new() -> Self {
        println!(
            "WorkgroupShapeConstraints::new called from {:?}",
            std::panic::Location::caller()
        );
        Self::default()
    }

    pub(crate) fn add_constraint(&mut self, dimension: usize, constraint: Constraint) {
        assert!(dimension < 3, "Dimension must be 0, 1, or 2");
        self.shape[dimension].push(constraint);
    }

    fn is_valid(&self, shape: &WorkgroupShape) -> bool {
        self.shape.iter().enumerate().all(|(i, constraints)| {
            constraints
                .iter()
                .all(|constraint| constraint.fits(shape.shape[i]))
        })
    }

    fn possible(&self) -> impl Iterator<Item = WorkgroupShape> {
        possible_workgroup_shapes().filter(move |shape| self.is_valid(shape))
    }

    pub(crate) fn solve(&self, limits: &wgpu::Limits) -> Option<WorkgroupShape> {
        static CACHE: OnceLock<
            RwLock<LruCache<WorkgroupShapeConstraints, Option<WorkgroupShape>, FxBuildHasher>>,
        > = OnceLock::new();
        let cache = CACHE.get_or_init(|| {
            RwLock::new(LruCache::with_hasher(
                const { NonZeroUsize::new(2048).unwrap() },
                Default::default(),
            ))
        });
        let mut write = cache.write();
        *write.get_or_insert_ref(self, || {
            // Find the smallest valid shape that matches the max subgroup size
            self.possible().min_by_key(|shape| {
                let linearized = shape.linearized();
                (linearized as i64)
                    + if shape.x() % limits.max_subgroup_size == 0 {
                        0
                    } else {
                        1024
                    }
            })
        })
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        for (i, constraints) in other.shape.iter().enumerate() {
            self.shape[i].extend(constraints.clone());
        }
    }
}

fn possible_workgroup_shapes() -> impl Iterator<Item = WorkgroupShape> {
    (1..=MAX_WORKGROUP_SIZE).flat_map(move |x| {
        (1..=(MAX_WORKGROUP_SIZE / x)).flat_map(move |y| {
            (1..=(MAX_WORKGROUP_SIZE / (x * y))).map(move |z| WorkgroupShape::new(x, y, z))
        })
    })
}

#[test]
fn test_all_possible_workgroup_shapes() {
    assert_eq!(possible_workgroup_shapes().count(), 5136);
}

#[cfg(test)]
fn test_limits() -> wgpu::Limits {
    let mut limits = wgpu::Limits::default();
    limits.max_subgroup_size = 64;
    limits
}

#[test]
fn test_workgroup_shape_constraints() {
    let mut constraints = WorkgroupShapeConstraints::new();
    constraints.add_constraint(0, Constraint::Equals(4));
    constraints.add_constraint(1, Constraint::LessThan(3));

    let valid_shapes: Vec<_> = constraints.possible().collect();
    println!("Valid shapes: {valid_shapes:#?}");
    for shape in valid_shapes {
        assert_eq!(shape.shape[0], 4);
        assert!(shape.shape[1] < 3);
    }

    let valid_shape = constraints.solve(&test_limits());
    assert_eq!(valid_shape.unwrap().shape, [4, 1, 16]);
    assert_eq!(valid_shape.unwrap().linearized(), 64);
}

#[test]
fn test_many_workgroup_shape_constraints() {
    let mut constraints = WorkgroupShapeConstraints::new();
    constraints.add_constraint(0, Constraint::Equals(4));
    constraints.add_constraint(1, Constraint::LessThan(3));

    let mut constraints2 = WorkgroupShapeConstraints::new();
    constraints2.add_constraint(1, Constraint::Equals(2));

    let mut merged = WorkgroupShapeConstraints::new();
    merged.merge(&constraints);
    merged.merge(&constraints2);
    let valid_shapes: Vec<_> = merged.possible().collect();
    println!("Valid shapes: {valid_shapes:#?}");
    for shape in valid_shapes {
        assert_eq!(shape.shape[0], 4);
        assert!(shape.shape[1] < 3);
    }

    let valid_shape = merged.solve(&test_limits());
    assert_eq!(valid_shape.unwrap().shape, [4, 2, 1]);
    assert_eq!(valid_shape.unwrap().linearized(), 8);
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) enum Constraint {
    Equals(u32),
    LessThan(u32),
    Not(Box<Constraint>),
}

impl Constraint {
    pub(crate) fn equals(value: u32) -> Self {
        Constraint::Equals(value)
    }

    pub(crate) fn less_than(value: u32) -> Self {
        Constraint::LessThan(value)
    }

    pub(crate) fn more_than_or_equals(value: u32) -> Self {
        Constraint::Not(Box::new(Constraint::LessThan(value)))
    }

    pub(crate) fn less_than_or_equals(value: u32) -> Self {
        Constraint::LessThan(value + 1)
    }

    fn fits(&self, shape: u32) -> bool {
        match self {
            Constraint::Equals(value) => shape == *value,
            Constraint::LessThan(value) => shape < *value,
            Constraint::Not(inner) => !inner.fits(shape),
        }
    }
}
