//! Browser-runnable WebGPU benchmarks.
//!
//! The conformance suite collects cases and lets each runner decide how to
//! execute them. Benchmarks follow the same shape, but each case returns
//! timing data instead of a pass/fail tensor comparison. Every case body is
//! async because a benchmark's fence is `Device::wait_async`, which on the
//! web completes on the browser's event loop.

pub mod registry;
pub mod sweep;
pub mod webgpu;

#[cfg(feature = "burn-bench")]
pub mod burn;

use std::{future::Future, pin::Pin};

use fusor::Device;

use web_time::{Duration, Instant};

pub type BenchmarkError = Box<dyn std::error::Error>;
pub type BenchmarkResult<T> = Result<T, BenchmarkError>;

type CaseFuture<'a> = Pin<Box<dyn Future<Output = BenchmarkResult<BenchmarkReport>> + 'a>>;
type BenchmarkRunner = dyn for<'a> FnOnce(&'a Device, BenchmarkConfig) -> CaseFuture<'a>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchmarkConfig {
    pub warmups: usize,
    pub iterations: usize,
    pub samples: usize,
}

impl BenchmarkConfig {
    pub const fn new(warmups: usize, iterations: usize, samples: usize) -> Self {
        Self {
            warmups,
            iterations,
            samples,
        }
    }

    pub const fn smoke() -> Self {
        Self::new(0, 1, 1)
    }

    fn sanitized(self) -> Self {
        Self {
            warmups: self.warmups,
            iterations: self.iterations.max(1),
            samples: self.samples.max(1),
        }
    }
}

impl Default for BenchmarkConfig {
    /// The iteration count here is only calibration's starting guess (see
    /// [`time_samples`]); nine rounds is what the reported median rests on,
    /// which holds the run-to-run spread near ten percent for about eight
    /// seconds of page time.
    fn default() -> Self {
        Self::new(2, 10, 9)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkReport {
    pub name: String,
    pub warmups: usize,
    pub iterations: usize,
    pub samples: usize,
    pub total_iterations: usize,
    pub sample_mean_ms: Vec<f64>,
    pub total_ms: f64,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub stddev_ms: f64,
    pub detail: String,
}

/// What one case's timed rounds produced: the rounds themselves and how many
/// iterations each of them held, which calibration decides rather than the
/// config (see [`time_samples`]).
pub(crate) struct TimedRounds {
    pub(crate) rounds: Vec<Duration>,
    pub(crate) iterations: usize,
}

impl BenchmarkReport {
    pub(crate) fn new(
        name: impl Into<String>,
        config: BenchmarkConfig,
        timed: TimedRounds,
        detail: impl Into<String>,
    ) -> Self {
        let config = config.sanitized();
        let iterations = timed.iterations.max(1);
        let samples = timed.rounds;
        let mut sample_mean_ms = samples
            .iter()
            .map(|elapsed| elapsed.as_secs_f64() * 1000.0 / iterations as f64)
            .collect::<Vec<_>>();
        let total_ms = samples
            .iter()
            .map(|elapsed| elapsed.as_secs_f64() * 1000.0)
            .sum::<f64>();
        let mean_ms = sample_mean_ms.iter().copied().sum::<f64>() / sample_mean_ms.len() as f64;
        sample_mean_ms.sort_by(f64::total_cmp);
        let median_ms = median(&sample_mean_ms);
        let min_ms = sample_mean_ms.first().copied().unwrap_or(0.0);
        let max_ms = sample_mean_ms.last().copied().unwrap_or(0.0);
        let stddev_ms = stddev(&sample_mean_ms, mean_ms);
        let total_iterations = iterations * samples.len();
        Self {
            name: name.into(),
            warmups: config.warmups,
            iterations,
            samples: config.samples,
            total_iterations,
            sample_mean_ms,
            total_ms,
            mean_ms,
            median_ms,
            min_ms,
            max_ms,
            stddev_ms,
            detail: detail.into(),
        }
    }
}

fn median(sorted: &[f64]) -> f64 {
    match sorted.len() {
        0 => 0.0,
        len if len % 2 == 1 => sorted[len / 2],
        len => (sorted[len / 2 - 1] + sorted[len / 2]) * 0.5,
    }
}

fn stddev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
}

#[derive(Clone, Debug, PartialEq)]
pub enum BenchmarkEvent {
    Started(String),
    Finished(BenchmarkReport),
}

pub struct BenchmarkCase {
    name: String,
    run: Box<BenchmarkRunner>,
}

impl BenchmarkCase {
    pub fn new(
        name: impl Into<String>,
        run: impl for<'a> FnOnce(&'a Device, BenchmarkConfig) -> CaseFuture<'a> + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            run: Box::new(run),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub async fn run(
        self,
        device: &Device,
        config: BenchmarkConfig,
    ) -> BenchmarkResult<BenchmarkReport> {
        (self.run)(device, config.sanitized()).await
    }
}

/// Time `samples` rounds of `iterations` computations, each round ending in
/// one `flush`.
///
/// The flush is per round, never per iteration, and that is the whole point.
/// Retrieving a result costs each library a fixed latency that has nothing to
/// do with the kernels: burn's is about 35 ms, so a suite that downloaded
/// every iteration reported that latency for every case and the same number
/// came back for a 128x128 add as for a 2048x2048 matmul. Amortized over a
/// round, what is left is the work. `run_once` therefore only has to get an
/// iteration *started*; `flush` makes the round's results real, and holds
/// every one of them alive until it does, so neither library can skip the
/// results nobody asked for.
/// How long a timed round should take. A browser clamps `performance.now()`
/// to 100 microseconds, so a round of a few hundred microseconds is only a
/// handful of ticks and quantization alone moved these cases by 3x between
/// runs. Twenty milliseconds is two hundred ticks, which puts the clock's
/// contribution near a half percent.
const TARGET_ROUND_MS: f64 = 20.0;

/// Ceiling on a calibrated round. A round holds every iteration's output
/// alive until it flushes, so the count is also how many results sit on the
/// device at once: at a megabyte an output, a few hundred is already a few
/// hundred megabytes. The cheapest cases hit this and settle for a shorter
/// round, which still clears the clock by a wide margin.
const MAX_ITERATIONS: usize = 256;

/// Time `samples` rounds, each of enough iterations to outrun the clock, and
/// close every round with one `flush`.
///
/// **Rounds, not iterations, are timed.** Retrieving a result costs each
/// library a fixed latency unrelated to the kernels — burn's is about 35 ms —
/// so a suite that downloaded every iteration reported that latency for every
/// case, and a 128x128 add measured the same as a 2048x2048 matmul.
///
/// **The iteration count is calibrated, not configured.** `config.iterations`
/// is the starting guess; one unrecorded round measures the case and the
/// count is rescaled so a timed round lands near [`TARGET_ROUND_MS`]. Without
/// it the cheap cases sat under the browser's timer resolution and their
/// numbers were noise.
///
/// `run_once` only has to get an iteration started; `flush` makes the round's
/// results real and holds every one of them alive until it does, so neither
/// library can skip the results nobody asked for.
pub(crate) async fn time_samples<F, Fut, T, G, GFut>(
    config: BenchmarkConfig,
    mut run_once: F,
    mut flush: G,
) -> BenchmarkResult<TimedRounds>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = BenchmarkResult<T>>,
    G: FnMut(Vec<T>) -> GFut,
    GFut: Future<Output = BenchmarkResult<()>>,
{
    let config = config.sanitized();
    let round = async |n: usize, run_once: &mut F, flush: &mut G| -> BenchmarkResult<Duration> {
        let started = Instant::now();
        let mut held = Vec::with_capacity(n);
        for _ in 0..n {
            held.push(run_once().await?);
        }
        flush(held).await?;
        Ok(started.elapsed())
    };

    for _ in 0..config.warmups {
        round(config.iterations, &mut run_once, &mut flush).await?;
    }

    // Calibrate off a round nobody records: whatever the case costs, the
    // timed rounds should be long enough to measure.
    let probe = round(config.iterations, &mut run_once, &mut flush).await?;
    let per_iteration_ms = probe.as_secs_f64() * 1000.0 / config.iterations as f64;
    let iterations = if per_iteration_ms > 0.0 {
        ((TARGET_ROUND_MS / per_iteration_ms).ceil() as usize)
            .clamp(config.iterations, MAX_ITERATIONS)
    } else {
        MAX_ITERATIONS
    };

    let mut rounds = Vec::with_capacity(config.samples);
    for _ in 0..config.samples {
        rounds.push(round(iterations, &mut run_once, &mut flush).await?);
    }
    Ok(TimedRounds { rounds, iterations })
}
