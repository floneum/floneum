//! One benchmark's size sweep on the native device: burn and fusor medians
//! side by side at every size, which is what shows whether either side's
//! number tracks the problem or sits at a fixed floor.

use fusor::Device;
use fusor_conformance::bench::{BenchmarkConfig, sweep};

fn main() {
    pollster::block_on(async {
        let device = Device::gpu().await.expect("a gpu device");
        let case = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "elementwise_add_square".to_string());
        let points = sweep::run_sweep(&case, &device, BenchmarkConfig::new(2, 3, 5), |_| {})
            .await
            .expect("the sweep");
        println!(
            "{:<14} {:>10} {:>12} {:>12}",
            "size", "elements", "burn ms", "fusor ms"
        );
        for point in &points {
            println!(
                "{:<14} {:>10} {:>12.3} {:>12.3}",
                point.label,
                point.value,
                point.burn.as_ref().map(|r| r.median_ms).unwrap_or(f64::NAN),
                point
                    .webgpu
                    .as_ref()
                    .map(|r| r.median_ms)
                    .unwrap_or(f64::NAN),
            );
        }
    });
}
