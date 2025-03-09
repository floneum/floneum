use std::fmt::Write;

use wgpu::CommandEncoder;

use crate::{
    DataTypeEnum, PerformanceQueries, TensorData,
    kernel::{GenericKernel, TensorInput},
};

pub(crate) struct VisitTiledKernel {
    rank: u32,
    contiguous: bool,
    tile_size: u32,
    kernel: GenericKernel,
}

impl VisitTiledKernel {
    pub(crate) fn new(
        rank: u32,
        tile_size: u32,
        contiguous: bool,
        datatypes: Vec<DataTypeEnum>,
        modify_data: impl FnMut(&mut GenericKernel, &[String], &[TensorInput]) -> String,
    ) -> Self {
        let mut kernel = GenericKernel::new();
        let kernel_text = Self::build_tiled_map_kernel(
            rank,
            tile_size,
            contiguous,
            datatypes,
            &mut kernel,
            modify_data,
        );
        kernel.set_body(kernel_text);
        let blocksize = Self::blocksize_raw(contiguous, rank);
        let workgroup_size = if contiguous {
            [blocksize, 1, 1]
        } else {
            std::array::from_fn(|i| if rank as usize > i { blocksize } else { 1 })
        };
        kernel.set_workgroup_size(workgroup_size);
        Self {
            rank,
            contiguous,
            kernel,
            tile_size,
        }
    }

    fn blocksize_raw(contiguous: bool, rank: u32) -> u32 {
        if contiguous {
            256
        } else {
            // max_blocksize^R = 256
            (256f64.powf(1. / rank as f64)).floor() as u32
        }
    }

    fn blocksize(&self) -> u32 {
        Self::blocksize_raw(self.contiguous, self.rank)
    }

    fn build_tiled_map_kernel(
        rank: u32,
        tile_size: u32,
        contiguous: bool,
        datatypes: Vec<DataTypeEnum>,
        kernel: &mut GenericKernel,
        mut modify_data: impl FnMut(&mut GenericKernel, &[String], &[TensorInput]) -> String,
    ) -> String {
        assert!(rank <= 3, "TensorLayout only supports up to 3 rank tensors");

        let mut kernel_body = String::new();
        let global_id = kernel.global_id();
        let tensors = datatypes
            .iter()
            .map(|ty| kernel.add_tensor_input(rank, true, *ty))
            .collect::<Vec<_>>();

        if contiguous {
            for local_index in 0..tile_size {
                let index = format!("index_{local_index}");
                writeln!(
                    &mut kernel_body,
                    "let {index} = {global_id}.x * {tile_size} + {local_index};"
                )
                .unwrap();
                tensors[0].check_bounds_contiguous(
                    &mut kernel_body,
                    index.clone(),
                    |kernel_body| {
                        let indexes = (0..datatypes.len())
                            .map(|_| index.clone())
                            .collect::<Vec<_>>();
                        let modify_data = modify_data(kernel, &indexes, &tensors);
                        writeln!(kernel_body, "{modify_data}").unwrap();
                    },
                );
            }
        } else {
            for i in 0..rank as usize {
                let index = ["x", "y", "z"][i];
                writeln!(
                    &mut kernel_body,
                    "let tile_index_{i} = {global_id}.{index} * {tile_size};"
                )
                .unwrap();
            }
            writeln!(&mut kernel_body, "\n").unwrap();

            for i in 0..rank {
                writeln!(&mut kernel_body, "for (var local_index_{i} = 0u; local_index_{i} < {tile_size}; local_index_{i}++) {{").unwrap();
            }

            for i in 0..rank {
                writeln!(
                    &mut kernel_body,
                    "let merged_index_{i} = tile_index_{i} + local_index_{i};"
                )
                .unwrap();
            }

            tensors[0].check_bounds(
                &mut kernel_body,
                (0..).map(|i| format!("merged_index_{i}")),
                |kernel_body| {
                    for (index, tensor) in tensors.iter().enumerate() {
                        writeln!(kernel_body, "let index_{index} = ",).unwrap();
                        tensor
                            .strided_index(kernel_body, (0..).map(|i| format!("merged_index_{i}")));
                        writeln!(kernel_body, ";").unwrap();
                    }
                    let indexes = (0..datatypes.len())
                        .map(|i| format!("index_{i}"))
                        .collect::<Vec<_>>();
                    let modify_data = modify_data(kernel, &indexes, &tensors);
                    writeln!(kernel_body, "{modify_data}").unwrap();
                },
            );

            for _ in 0..rank {
                writeln!(&mut kernel_body, "}}").unwrap();
            }
        }

        kernel_body
    }

    pub(crate) fn run_with_query<'a>(
        &self,
        tensors: impl IntoIterator<Item = &'a TensorData>,
        query: Option<&PerformanceQueries>,
        command_encoder: &mut CommandEncoder,
    ) {
        let tensors = tensors.into_iter().collect::<Vec<_>>();
        let layout = tensors[0].layout();
        let shape = layout.shape();
        let max_blocksize = self.blocksize();
        let workgroup_dispatch_size = if self.contiguous {
            [
                shape
                    .iter()
                    .map(|x| *x as u32)
                    .product::<u32>()
                    .div_ceil(self.tile_size * max_blocksize),
                1,
                1,
            ]
        } else {
            let workgroup_size_x = shape
                .get(0)
                .map(|x| (*x as u32).div_ceil(self.tile_size * max_blocksize))
                .unwrap_or(1);
            let workgroup_size_y = shape
                .get(1)
                .map(|x| (*x as u32).div_ceil(self.tile_size * max_blocksize))
                .unwrap_or(1);
            let workgroup_size_z = shape
                .get(2)
                .map(|x| (*x as u32).div_ceil(self.tile_size * max_blocksize))
                .unwrap_or(1);
            [workgroup_size_x, workgroup_size_y, workgroup_size_z]
        };

        let device = tensors[0].device();
        self.kernel.run_with_query(
            device,
            tensors.iter().map(|x| (*x).clone()),
            query,
            command_encoder,
            workgroup_dispatch_size,
        );
    }
}
