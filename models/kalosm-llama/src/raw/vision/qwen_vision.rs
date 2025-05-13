use candle_core::{Tensor, WithDType};

fn pad(tensor: &Tensor, shape: &[usize], value: impl WithDType) -> candle_core::Result<Tensor> {
    let new_tensor = Tensor::full(value, shape, tensor.device())?;

    let ranges = tensor
        .shape()
        .dims()
        .iter()
        .map(|old| 0..*old)
        .collect::<Vec<_>>();
    new_tensor.slice_assign(&ranges, tensor)
}

#[test]
fn test_pad() {
    let tensor = Tensor::arange(0u32, 12, &candle_core::Device::Cpu)
        .unwrap()
        .reshape((3, 4))
        .unwrap();
    let value = u32::MAX;

    let padded_tensor = pad(&tensor, &[4, 5], value).unwrap();

    println!("Original Tensor: {:?}", tensor);
    println!("Padded Tensor: {:?}", padded_tensor);
    assert_eq!(
        padded_tensor.to_vec2::<u32>().unwrap(),
        vec![
            vec![0, 1, 2, 3, u32::MAX],
            vec![4, 5, 6, 7, u32::MAX],
            vec![8, 9, 10, 11, u32::MAX],
            vec![u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX]
        ]
    );
}

fn get_window_index(
    grid_thw: Vec<(usize, usize, usize)>,
    window_size: usize,
    spatial_merge_size: usize,
    spatial_merge_unit: usize,
    device: &candle_core::Device,
) -> candle_core::Result<(Tensor, Vec<u32>)> {
    let mut window_index = vec![];
    let mut cu_window_seqlens = vec![0];
    let mut window_index_id = 0;
    let vit_merger_window_size = window_size / spatial_merge_size;

    for (grid_t, grid_h, grid_w) in grid_thw {
        let llm_grid_h = grid_h / spatial_merge_size;
        let llm_grid_w = grid_w / spatial_merge_size;
        let index = Tensor::arange(0, (grid_t * llm_grid_h * llm_grid_w) as u32, device)?
            .reshape((grid_t, llm_grid_h, llm_grid_w))?;
        let pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size;
        let pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size;
        let num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
        let num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;
        let index_padded = {
            pad(
                &index,
                &[grid_t, llm_grid_h + pad_h, llm_grid_w + pad_w],
                u32::MAX,
            )?
            .reshape((
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            ))?
            .permute([0, 1, 3, 2, 4])?
            .reshape((
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            ))?
        };
        let seqlens = index_padded.ne(u32::MAX)?.sum([2, 3])?.reshape(((),))?;
        let index_padded = index_padded.reshape(((),))?;
        let index_new = index_padded.ne(u32::MAX)?;
        window_index.push((index_new + window_index_id as f64)?);
        let cu_seqlens_tmp = ((seqlens.cumsum(0)? * spatial_merge_unit as f64)?
            + *cu_window_seqlens.last().unwrap() as f64)?;
        cu_window_seqlens.extend(cu_seqlens_tmp.to_vec1::<u32>()?);
        window_index_id += (grid_t * llm_grid_h * llm_grid_w) as usize;
    }
    let window_index = Tensor::cat(&window_index, 0)?;

    Ok((window_index, cu_window_seqlens))
}

#[test]
fn test_get_window_index() {
    let grid_thw = vec![(1, 4, 4)];
    let window_size = 2;
    let spatial_merge_size = 2;
    let spatial_merge_unit = 1;
    let device = candle_core::Device::Cpu;

    let (window_index, cu_window_seqlens) = get_window_index(
        grid_thw,
        window_size,
        spatial_merge_size,
        spatial_merge_unit,
        &device,
    )
    .unwrap();

    println!("Window Index: {:?}", window_index);
    println!("CU Window Seqlens: {:?}", cu_window_seqlens);
}
