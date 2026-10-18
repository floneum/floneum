//! Debugging switches read from `FUSOR_*` environment variables, parsed once.

use std::path::PathBuf;
use std::sync::OnceLock;

pub(crate) struct Flags {
    /// `FUSOR_GAPSTEP`: one resolve-phase stopwatch line per resolve.
    pub gapstep: bool,
    /// Bounded diagnostic override of the native completion polling window.
    pub poll_spin_ms: Option<u64>,
    /// Optional bounded sleep between native nonblocking completion polls.
    pub poll_interval_us: u64,
    /// Bounded F32 arithmetic probe for Q4 contractions: 0/1/2/3.
    pub q4_variant: u8,
    /// `FUSOR_COLDLIST`: under `gapstep`, one line per launch the probe missed.
    pub coldlist: bool,
    /// `FUSOR_VERIFY_ARTIFACT_CACHE`: relower every cache hit and compare
    /// (compiler-test builds only).
    pub verify_artifact_cache: bool,
    /// `FUSOR_MISMATCH_DUMP`: where a failed cache verification dumps the body.
    pub mismatch_dump: Option<PathBuf>,
    /// `FUSOR_NO_PIPELINE_SHARE`: compile every body, sharing no pipeline.
    pub no_pipeline_share: bool,
    /// `FUSOR_WGSL_DUMP`: a directory every emitted module is written into.
    pub wgsl_dump: Option<PathBuf>,
    /// `FUSOR_COMPILE_THREADS`: cap on parallel lower-and-compile workers.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub compile_threads: Option<usize>,
    /// `FUSOR_TIME_PLAN`: print each plan's GPU span as `TPLAN <us>`.
    pub time_plan: bool,
    /// `FUSOR_TIME_RANGE`: time live dispatches from this index as `TSPAN`.
    pub time_range: Option<usize>,
    /// Rotate through one in N dispatches on successive resolves.
    pub time_sample_stride: Option<usize>,
    /// `FUSOR_TRACE_DISPATCH`: one dispatch per submission, each waited on.
    pub trace_dispatch: bool,
    /// `FUSOR_CACHE_STATS`: retained-object counts every 64 resolves.
    pub cache_stats: bool,
    /// `FUSOR_PASS_SIZE`: dispatches per compute pass.
    pub pass_size: Option<usize>,
    /// Diagnostic submission chunk size; zero is rejected.
    pub submit_size: Option<usize>,
    /// Larger preparation chunks after the first early submission; bounded probes.
    pub stream_submit_rest: Option<usize>,
    /// `FUSOR_NO_TILE_ALIAS`: give every workgroup tile its own allocation.
    pub no_tile_alias: bool,
    /// `FUSOR_DUMP_CONTRACT`: the family, point and shape of each contraction.
    pub dump_contract: bool,
    /// Diagnostic SGEMV accumulation chains per column; default preserves order.
    pub gemv_chains: usize,
    /// Exact integer weight decoding through normal half-bit encodings; F32 math.
    pub q4_half_unpack: bool,
    /// Exact signed Q6 integers through F32 mantissa encodings.
    pub q6_float_unpack: bool,
    pub packed_q4_splits: u32,
    pub packed_q4_split_max_n: Option<usize>,
    pub packed_q4_split_max_k: Option<usize>,
    pub q6_splits: u32,
    pub q6_split_max_n: Option<usize>,
    pub gemv_contiguous_n: bool,
    pub scatter_single_linear: Option<u32>,
    pub time_passes: bool,
    pub no_debug_groups: bool,
    pub inflight_chunks: Option<usize>,
    /// Additional in-flight limit for large plans, keeping prefill transients bounded.
    pub large_plan_inflight_chunks: Option<usize>,
    pub large_plan_submit_size: Option<usize>,
    /// Scope extra prefill resource limits independently of normal pass chunking.
    pub large_plan_min_dispatches: Option<usize>,
    /// `FUSOR_POOL_DEBUG`: log pool allocation failures.
    pub pool_debug: bool,
    /// Optional lower ceiling for memory-bounded experiments; never raises it.
    pub memory_budget_bytes: Option<u64>,
}

pub(crate) fn flags() -> &'static Flags {
    static FLAGS: OnceLock<Flags> = OnceLock::new();
    FLAGS.get_or_init(|| {
        let on = |name| {
            std::env::var(name).is_ok_and(|v| !matches!(v.as_str(), "0" | "false" | "off" | "no"))
        };
        let path = |name| std::env::var_os(name).map(PathBuf::from);
        let number = |name| std::env::var(name).ok().and_then(|v| v.parse().ok());
        Flags {
            gapstep: on("FUSOR_GAPSTEP"),
            q4_variant: std::env::var("FUSOR_PROFILE_Q4_VARIANT")
                .ok()
                .map(|v| {
                    let n = v.parse().expect("Q4 variant must be an integer");
                    assert!(n <= 3, "Q4 variant must be 0..=3");
                    n
                })
                .unwrap_or(0),
            poll_spin_ms: std::env::var("FUSOR_PROFILE_POLL_SPIN_MS").ok().map(|v| {
                let n = v
                    .parse()
                    .expect("FUSOR_PROFILE_POLL_SPIN_MS must be an integer");
                assert!(n <= 64, "FUSOR_PROFILE_POLL_SPIN_MS must be 0..=64");
                n
            }),
            poll_interval_us: std::env::var("FUSOR_PROFILE_POLL_INTERVAL_US")
                .ok()
                .map(|v| {
                    let n = v.parse().expect("poll interval must be an integer");
                    assert!(n <= 1000, "poll interval must be 0..=1000 us");
                    n
                })
                .unwrap_or(0),
            coldlist: on("FUSOR_COLDLIST"),
            verify_artifact_cache: cfg!(feature = "compiler-tests")
                && on("FUSOR_VERIFY_ARTIFACT_CACHE"),
            mismatch_dump: path("FUSOR_MISMATCH_DUMP"),
            no_pipeline_share: on("FUSOR_NO_PIPELINE_SHARE"),
            wgsl_dump: path("FUSOR_WGSL_DUMP"),
            compile_threads: Some(number("FUSOR_COMPILE_THREADS").unwrap_or(1)),
            time_plan: on("FUSOR_TIME_PLAN"),
            time_range: number("FUSOR_TIME_RANGE"),
            time_sample_stride: number("FUSOR_TIME_SAMPLE_STRIDE").inspect(|n| {
                assert!(*n > 0, "FUSOR_TIME_SAMPLE_STRIDE must be positive");
            }),
            trace_dispatch: on("FUSOR_TRACE_DISPATCH"),
            cache_stats: on("FUSOR_CACHE_STATS"),
            pass_size: number("FUSOR_PASS_SIZE"),
            submit_size: std::env::var("FUSOR_PROFILE_SUBMIT_SIZE").ok().map(|v| {
                let n = v
                    .parse()
                    .expect("FUSOR_PROFILE_SUBMIT_SIZE must be an integer");
                assert!(n > 0, "FUSOR_PROFILE_SUBMIT_SIZE must be positive");
                n
            }),
            stream_submit_rest: std::env::var("FUSOR_PROFILE_STREAM_SUBMIT_REST")
                .ok()
                .map(|v| {
                    let n = v.parse().expect("stream submit rest must be an integer");
                    assert!(
                        (1..=4096).contains(&n),
                        "stream submit rest must be 1..=4096"
                    );
                    n
                }),
            no_tile_alias: on("FUSOR_NO_TILE_ALIAS"),
            dump_contract: on("FUSOR_DUMP_CONTRACT"),
            gemv_chains: std::env::var("FUSOR_PROFILE_GEMV_CHAINS")
                .ok()
                .map(|v| {
                    let n = v
                        .parse()
                        .expect("FUSOR_PROFILE_GEMV_CHAINS must be an integer");
                    assert!(
                        matches!(n, 1 | 2 | 4 | 8),
                        "FUSOR_PROFILE_GEMV_CHAINS must be 1, 2, 4, or 8"
                    );
                    n
                })
                .unwrap_or(1),
            q4_half_unpack: on("FUSOR_PROFILE_Q4_HALF_UNPACK"),
            q6_float_unpack: on("FUSOR_PROFILE_Q6_FLOAT_UNPACK"),
            packed_q4_splits: std::env::var("FUSOR_PROFILE_PACKED_Q4_SPLITS")
                .ok()
                .map(|s| {
                    let n: u32 = s
                        .parse()
                        .expect("FUSOR_PROFILE_PACKED_Q4_SPLITS must be an integer");
                    assert!(
                        matches!(n, 1 | 2 | 4 | 8),
                        "packed Q4 splits must be 1, 2, 4 or 8"
                    );
                    n
                })
                .unwrap_or(1),
            packed_q4_split_max_n: number("FUSOR_PROFILE_PACKED_Q4_SPLIT_MAX_N"),
            packed_q4_split_max_k: number("FUSOR_PROFILE_PACKED_Q4_SPLIT_MAX_K"),
            q6_splits: std::env::var("FUSOR_PROFILE_Q6_SPLITS")
                .ok()
                .map(|s| {
                    let n = s.parse().expect("Q6 splits must be an integer");
                    assert!(
                        matches!(n, 1 | 2 | 4 | 8),
                        "Q6 splits must be 1, 2, 4, or 8"
                    );
                    n
                })
                .unwrap_or(1),
            q6_split_max_n: number("FUSOR_PROFILE_Q6_SPLIT_MAX_N"),
            gemv_contiguous_n: on("FUSOR_PROFILE_GEMV_CONTIGUOUS_N"),
            scatter_single_linear: std::env::var("FUSOR_PROFILE_SCATTER_SINGLE_LINEAR")
                .ok()
                .map(|s| {
                    let n = s.parse().expect("scatter linear width must be an integer");
                    assert!(
                        matches!(n, 1 | 2 | 4 | 8 | 16),
                        "scatter linear width must be 1,2,4,8,16"
                    );
                    n
                }),
            time_passes: on("FUSOR_TIME_PASSES"),
            no_debug_groups: on("FUSOR_PROFILE_NO_DEBUG_GROUPS"),
            inflight_chunks: number("FUSOR_PROFILE_INFLIGHT_CHUNKS"),
            large_plan_inflight_chunks: number("FUSOR_PROFILE_LARGE_PLAN_INFLIGHT_CHUNKS")
                .inspect(|n| assert!(*n > 0, "large-plan in-flight limit must be positive")),
            large_plan_submit_size: number("FUSOR_PROFILE_LARGE_PLAN_SUBMIT_SIZE")
                .inspect(|n| assert!(*n > 0, "large-plan submit size must be positive")),
            large_plan_min_dispatches: number("FUSOR_PROFILE_LARGE_PLAN_MIN_DISPATCHES")
                .inspect(|n| assert!(*n > 0, "large-plan threshold must be positive")),
            pool_debug: on("FUSOR_POOL_DEBUG"),
            memory_budget_bytes: std::env::var("FUSOR_PROFILE_MEMORY_BUDGET_BYTES")
                .ok()
                .map(|v| {
                    let bytes = v.parse().expect("memory budget must be an integer");
                    assert!(bytes > 0, "memory budget must be positive");
                    bytes
                }),
        }
    })
}
