// The shape of the workgroup. [x, y, z] where their product is <= 256.
//
// Kernels can be fused if their workgroup shape can be coerced. Coercion can happen if
// the biggest linearized workgroup shape is a multiple of all smaller workgroup shapes.

use crate::mir::kernel::GenericKernel;

const MAX_WORKGROUP_SIZE: u32 = 256;

struct RealWorkgroupShape {
    size: WorkgroupShape,
}

struct MappedWorkgroupShape {
    real_size: RealWorkgroupShape,
    virtual_size: WorkgroupShape,
}

impl MappedWorkgroupShape {
    fn new(real_size: RealWorkgroupShape, virtual_size: WorkgroupShape) -> Option<Self> {
        // The virtual size must be a multiple of the real size
        (real_size.size.linearized() % virtual_size.linearized() == 0).then_some(Self {
            real_size,
            virtual_size,
        })
    }

    fn workgroup_scale(&self) -> u32 {
        self.real_size.size.linearized() / self.virtual_size.linearized()
    }

    fn component_linearized(&self, i: usize) -> bool {
        self.real_size.size.shape[i] != self.virtual_size.shape[i] * self.workgroup_scale()
    }

    fn linearized_components(&self) -> [bool; 3] {
        std::array::from_fn(|i| self.component_linearized(i))
    }

    fn linearized_component(&self, kernel: &mut GenericKernel) -> String {
        let mut merged = "0".to_string();
        let mut product = 1;
        for ((component, real_size), linearized) in ["x", "y", "z"]
            .iter()
            .zip(self.real_size.size.shape)
            .zip(self.linearized_components())
        {
            if linearized {
                merged += &format!(" + {}.{} * {}", kernel.global_id(), component, product);
            }
            product *= real_size;
        }

        merged
    }

    fn get_global_id(&self, kernel: &mut GenericKernel, linearized: &str, i: usize) -> String {
        if self.component_linearized(i) {
            format!(
                "({linearized} / {}) % {}",
                self.virtual_size.shape[..i].iter().product::<u32>(),
                self.virtual_size.shape[i]
            )
        } else {
            let global_id = kernel.global_id();
            let component = ["x", "y", "z"][i];
            format!(
                "{global_id}.{component} % {}",
                self.real_size.size.shape[i] / self.virtual_size.shape[i]
            )
        }
    }
}

#[test]
fn test_one_to_one_mapped_work_groups() {
    let real_size = RealWorkgroupShape {
        size: WorkgroupShape::new(4, 2, 8),
    };
    assert_eq!(real_size.size.linearized(), 64);
    let virtual_size = WorkgroupShape::new(16, 2, 2);
    assert_eq!(virtual_size.linearized(), 64);
    let mapped_size = MappedWorkgroupShape::new(real_size, virtual_size).unwrap();

    assert_eq!(mapped_size.linearized_components(), [true, false, true]);
    let linearized = mapped_size.linearized_component(&mut GenericKernel::new());
    assert_eq!(linearized, "0 + global_id.x * 1 + global_id.z * 8");

    let x_index = mapped_size.get_global_id(&mut GenericKernel::new(), "linearized", 0);
    assert_eq!(x_index, "(linearized / 1) % 16");

    let y_index = mapped_size.get_global_id(&mut GenericKernel::new(), "linearized", 1);
    assert_eq!(y_index, "global_id.y % 1");

    let z_index = mapped_size.get_global_id(&mut GenericKernel::new(), "linearized", 2);
    assert_eq!(z_index, "(linearized / 32) % 2");
}

#[test]
fn test_many_to_one_mapped_work_groups() {
    let real_size = RealWorkgroupShape {
        size: WorkgroupShape::new(4, 2, 8),
    };
    assert_eq!(real_size.size.linearized(), 64);
    let virtual_size = WorkgroupShape::new(8, 2, 2);
    assert_eq!(virtual_size.linearized(), 32);
    let mapped_size = MappedWorkgroupShape::new(real_size, virtual_size).unwrap();

    assert_eq!(mapped_size.linearized_components(), [true, true, true]);
    let linearized = mapped_size.linearized_component(&mut GenericKernel::new());
    assert_eq!(
        linearized,
        "0 + global_id.x * 1 + global_id.y * 4 + global_id.z * 8"
    );

    let x_index = mapped_size.get_global_id(&mut GenericKernel::new(), "linearized", 0);
    assert_eq!(x_index, "(linearized / 1) % 8");
    let y_index = mapped_size.get_global_id(&mut GenericKernel::new(), "linearized", 1);
    assert_eq!(y_index, "(linearized / 8) % 2");
    let z_index = mapped_size.get_global_id(&mut GenericKernel::new(), "linearized", 2);
    assert_eq!(z_index, "(linearized / 16) % 2");
}

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
}

#[derive(Default, Debug, Clone)]
pub(crate) struct WorkgroupShapeConstraints {
    shape: [Vec<Constraint>; 3],
}

impl WorkgroupShapeConstraints {
    pub(crate) fn new() -> Self {
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

    pub(crate) fn solve(&self) -> Option<WorkgroupShape> {
        self.possible().max_by_key(|shape| shape.linearized())
    }

    fn possible_all(
        constraints: impl IntoIterator<Item = Self>,
    ) -> impl Iterator<Item = WorkgroupShape> {
        let all = constraints.into_iter().collect::<Vec<_>>();
        possible_workgroup_shapes()
            .filter(move |shape| all.iter().all(|constraint| constraint.is_valid(shape)))
    }

    fn solve_all(constraints: impl IntoIterator<Item = Self>) -> Option<WorkgroupShape> {
        Self::possible_all(constraints).max_by_key(|shape| shape.linearized())
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

#[test]
fn test_workgroup_shape_constraints() {
    let mut constraints = WorkgroupShapeConstraints::new();
    constraints.add_constraint(0, Constraint::Equals(4));
    constraints.add_constraint(1, Constraint::LessThan(3));
    constraints.add_constraint(2, Constraint::MultipleOf(2));

    let valid_shapes: Vec<_> = constraints.possible().collect();
    println!("Valid shapes: {:#?}", valid_shapes);
    for shape in valid_shapes {
        assert_eq!(shape.shape[0], 4);
        assert!(shape.shape[1] < 3);
        assert_eq!(shape.shape[2] % 2, 0);
    }

    let valid_shape = constraints.solve();
    assert_eq!(valid_shape.unwrap().shape, [4, 2, 32]);
    assert_eq!(valid_shape.unwrap().linearized(), 256);
}

#[test]
fn test_many_workgroup_shape_constraints() {
    let mut constraints = WorkgroupShapeConstraints::new();
    constraints.add_constraint(0, Constraint::Equals(4));
    constraints.add_constraint(1, Constraint::LessThan(3));
    constraints.add_constraint(2, Constraint::MultipleOf(2));

    let mut constraints2 = WorkgroupShapeConstraints::new();
    constraints2.add_constraint(0, Constraint::MultipleOf(2));
    constraints2.add_constraint(1, Constraint::Equals(2));
    constraints2.add_constraint(2, Constraint::MultipleOf(8));

    let valid_shapes: Vec<_> =
        WorkgroupShapeConstraints::possible_all([constraints.clone(), constraints2.clone()])
            .collect();
    println!("Valid shapes: {:#?}", valid_shapes);
    for shape in valid_shapes {
        assert_eq!(shape.shape[0], 4);
        assert!(shape.shape[1] < 3);
        assert_eq!(shape.shape[2] % 8, 0);
    }

    let valid_shape = WorkgroupShapeConstraints::solve_all([constraints, constraints2]);
    assert_eq!(valid_shape.unwrap().shape, [4, 2, 32]);
    assert_eq!(valid_shape.unwrap().linearized(), 256);
}

#[derive(Debug, Clone)]
pub(crate) enum Constraint {
    Equals(u32),
    LessThan(u32),
    MultipleOf(u32),
    Or(Box<Constraint>, Box<Constraint>),
    And(Box<Constraint>, Box<Constraint>),
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

    pub(crate) fn multiple_of(value: u32) -> Self {
        Constraint::MultipleOf(value)
    }

    pub(crate) fn or(left: Constraint, right: Constraint) -> Self {
        Constraint::Or(Box::new(left), Box::new(right))
    }

    pub(crate) fn and(left: Constraint, right: Constraint) -> Self {
        Constraint::And(Box::new(left), Box::new(right))
    }

    pub(crate) fn not(inner: Constraint) -> Self {
        Constraint::Not(Box::new(inner))
    }

    fn fits(&self, shape: u32) -> bool {
        match self {
            Constraint::Equals(value) => shape == *value,
            Constraint::LessThan(value) => shape < *value,
            Constraint::MultipleOf(value) => shape % value == 0,
            Constraint::Or(left, right) => left.fits(shape) || right.fits(shape),
            Constraint::And(left, right) => left.fits(shape) && right.fits(shape),
            Constraint::Not(inner) => !inner.fits(shape),
        }
    }
}
