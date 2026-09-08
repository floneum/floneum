//! Run the benchmark registry on the native device and print each case's
//! fusor-versus-burn ratio, so the comparison the browser page shows can be
//! checked without a browser.

use fusor::Device;
use fusor_conformance::bench::{BenchmarkConfig, BenchmarkEvent, BenchmarkReport, registry};

fn main() {
    pollster::block_on(async {
        let device = Device::gpu().await.expect("a gpu device");
        let filter = std::env::args().nth(1);
        let cases = registry::cases()
            .into_iter()
            .filter(|c| {
                filter
                    .as_ref()
                    .is_none_or(|f| c.name().contains(f.as_str()))
            })
            .collect::<Vec<_>>();
        let mut reports: Vec<BenchmarkReport> = Vec::new();
        registry::run_cases(&device, BenchmarkConfig::default(), cases, |event| {
            if let BenchmarkEvent::Finished(report) = event {
                println!("{:<52} {:>9.3} ms", report.name, report.mean_ms);
                reports.push(report);
            }
        })
        .await
        .expect("the benchmark suite");

        println!(
            "\n{:<44} {:>9} {:>9} {:>10}",
            "case", "burn ms", "fusor ms", "ratio"
        );
        for report in &reports {
            let Some((suite, case)) = report.name.split_once("::") else {
                continue;
            };
            if suite != "webgpu" {
                continue;
            }
            let Some(burn) = reports.iter().find(|r| r.name == format!("burn::{case}")) else {
                continue;
            };
            // Medians, for the reason the page's own comparison gives: a
            // first-round warmup outlier drags a mean into fiction.
            let ratio = burn.median_ms / report.median_ms;
            let word = if ratio >= 1.0 { "faster" } else { "slower" };
            let shown = if ratio >= 1.0 { ratio } else { 1.0 / ratio };
            println!(
                "{case:<44} {:>9.3} {:>9.3} {shown:>7.2}x {word}",
                burn.median_ms, report.median_ms
            );
        }
    });
}
