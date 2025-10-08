use std::fmt::Display;
use std::fmt::Write;

/// A binding for a tensor input
pub struct InfoBinding {
    binding: u32,
}

impl Display for InfoBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i_{}", self.binding)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TensorInput {
    pub(crate) tensor_binding: u32,
    pub(crate) info_binding: u32,
    pub(crate) rank: u32,
}

impl TensorInput {
    pub(crate) fn get_info_binding(&self) -> u32 {
        self.info_binding
    }

    fn info_binding(&self) -> InfoBinding {
        InfoBinding {
            binding: self.get_info_binding(),
        }
    }

    pub(crate) fn offset_binding(&self) -> impl Display {
        struct OffsetBinding {
            binding: InfoBinding,
        }

        impl Display for OffsetBinding {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}.offset", self.binding)
            }
        }

        OffsetBinding {
            binding: self.info_binding(),
        }
    }

    pub(crate) fn stride_binding(&self, rank: u32) -> impl Display {
        struct StrideBinding {
            binding: InfoBinding,
            rank: u32,
        }

        impl Display for StrideBinding {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}.stride_{}", self.binding, self.rank)
            }
        }

        StrideBinding {
            binding: self.info_binding(),
            rank,
        }
    }

    pub(crate) fn shape_binding(&self, rank: u32) -> impl Display {
        struct ShapeBinding {
            binding: InfoBinding,
            rank: u32,
        }

        impl Display for ShapeBinding {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}.shape_{}", self.binding, self.rank)
            }
        }

        ShapeBinding {
            binding: self.info_binding(),
            rank,
        }
    }

    pub(crate) fn check_bounds<W: Write, O, D: Display>(
        &self,
        write: &mut W,
        indexes: impl IntoIterator<Item = D>,
        in_bounds: impl FnOnce(&mut W) -> O,
    ) -> O {
        write!(write, "if true ").unwrap();
        for (i, index) in indexes.into_iter().enumerate().take(self.rank as usize) {
            let stride = self.shape_binding(i as u32);
            write!(write, "&& {index} < {stride} ").unwrap();
        }
        write!(write, "{{").unwrap();
        let out = in_bounds(write);
        write!(write, "}}").unwrap();
        out
    }

    pub(crate) fn strided_index<D: Display>(
        &self,
        write: &mut impl Write,
        indexes: impl IntoIterator<Item = D>,
    ) {
        let offset = self.offset_binding();
        write!(write, "{offset}").unwrap();
        if self.rank > 0 {
            write!(write, " + ").unwrap();
        }
        for (i, index) in indexes.into_iter().enumerate().take(self.rank as usize) {
            let stride = self.stride_binding(i as u32);
            write!(write, "({index})*{stride}").unwrap();
            if i < (self.rank - 1) as usize {
                write!(write, " + ").unwrap();
            }
        }
    }

    pub fn rank(&self) -> u32 {
        self.rank
    }
}

impl Display for TensorInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i_{}", self.tensor_binding)
    }
}
