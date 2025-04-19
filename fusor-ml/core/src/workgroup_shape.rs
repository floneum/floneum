// The shape of the workgroup. [x, y, z] where their product is <= 256.
//
// Kernels can be fused if their workgroup shape can be cohered. Coercion can happen if
// the biggest linearized workgroup shape is a multiple of all smaller workgroup shapes.

use crate::kernel::GenericKernel;

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

struct WorkgroupShape {
    shape: [u32; 3],
}

impl WorkgroupShape {
    fn new(x: u32, y: u32, z: u32) -> Self {
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

    fn linearized(&self) -> u32 {
        self.shape.iter().product()
    }
}
