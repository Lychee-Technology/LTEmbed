// benches/inference.rs
//
// Warm-invocation benchmarks for LTEmbed ZeroVecEngine.
//
// Run with:
//   cargo bench
//   RUSTFLAGS="-C target-cpu=native" cargo bench
//
// Requires: assets/model.safetensors (download separately)
// Skips gracefully if the model file is absent.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ltembed::{
    benchmarking::{
        gemm_microbenchmark_scenarios, padded_seq_len, projection_kernel_shapes,
        scenario_token_lengths, BENCHMARK_MAX_LENGTH,
    },
    engine::ZeroVecEngine,
    models::bert::{gelu, layer_norm_rows, masked_softmax, softmax_unmasked},
    traits::{pooling::MeanPooling, tokenizer::HFTokenizer},
};
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::path::Path;
use std::thread;

#[cfg(all(
    feature = "vendored-blas",
    target_arch = "aarch64",
    target_os = "linux"
))]
use ltembed::benchmarking::dense_backend_name;
#[cfg(all(
    feature = "vendored-blas",
    target_arch = "aarch64",
    target_os = "linux"
))]
use openblas_src as _;

// ── Test inputs ───────────────────────────────────────────────────────────────

const SHORT: &str = "query: Hello, world!";
const MEDIUM: &str =
    "query: What is the impact of large language models on software engineering productivity?";

fn long_input() -> String {
    "passage: ".to_string() + &"The quick brown fox jumps over the lazy dog. ".repeat(30)
}

// ── Asset guard ───────────────────────────────────────────────────────────────

fn assets_available() -> bool {
    Path::new("assets/model.safetensors").exists()
        && Path::new("assets/config.json").exists()
        && Path::new("assets/tokenizer.json").exists()
}

fn kernel_assets_available() -> bool {
    Path::new("assets/config.json").exists() && Path::new("assets/tokenizer.json").exists()
}

// ── Shared engine (initialized once per bench run) ───────────────────────────

static ENGINE: Lazy<ZeroVecEngine> = Lazy::new(|| {
    let config = std::fs::read_to_string("assets/config.json")
        .expect("assets/config.json missing — run scripts/generate_fixtures.py first");
    ZeroVecEngine::new(
        "assets/model.safetensors",
        &config,
        "assets/tokenizer.json",
        Box::new(MeanPooling),
    )
    .expect("Failed to initialize ZeroVecEngine")
});

#[derive(Debug, Deserialize)]
struct ModelConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
}

#[derive(Debug, Clone)]
struct KernelBenchmarkCase {
    name: String,
    batch_size: usize,
    seq_len: usize,
    total_tokens: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
}

impl KernelBenchmarkCase {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

fn patterned_f32(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i % 251) as f32 - 125.0) / 125.0)
        .collect()
}

fn repeat_copy(src: &[f32], dst: &mut [f32], repeats: usize) {
    for _ in 0..repeats {
        dst.copy_from_slice(src);
        criterion::black_box(&mut *dst);
    }
}

fn repeat_fill(dst: &mut [f32], repeats: usize) {
    for _ in 0..repeats {
        dst.fill(0.0);
        criterion::black_box(&mut *dst);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_sgemm(
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    a: *const f32,
    rsa: isize,
    csa: isize,
    b: *const f32,
    rsb: isize,
    csb: isize,
    c: *mut f32,
    rsc: isize,
    csc: isize,
) {
    unsafe {
        matrixmultiply::sgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, 0.0, c, rsc, csc);
    }
}

#[cfg(all(
    feature = "vendored-blas",
    target_arch = "aarch64",
    target_os = "linux"
))]
fn run_cblas_sgemm(m: usize, k: usize, n: usize, alpha: f32, a: &[f32], b: &[f32], c: &mut [f32]) {
    unsafe {
        cblas::sgemm(
            cblas::Layout::RowMajor,
            cblas::Transpose::None,
            cblas::Transpose::None,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            k as i32,
            b,
            n as i32,
            0.0,
            c,
            n as i32,
        );
    }
}

fn add_bias_rows(out: &mut [f32], batch: usize, bias: &[f32]) {
    let output_size = bias.len();
    for row in 0..batch {
        let offset = row * output_size;
        for (col, value) in bias.iter().enumerate() {
            out[offset + col] += value;
        }
    }
}

fn run_matrixmultiply_linear_with_bias(
    x_rows: &[f32],
    batch: usize,
    input_size: usize,
    weight_t: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    run_sgemm(
        batch,
        input_size,
        bias.len(),
        1.0,
        x_rows.as_ptr(),
        input_size as isize,
        1,
        weight_t.as_ptr(),
        bias.len() as isize,
        1,
        out.as_mut_ptr(),
        bias.len() as isize,
        1,
    );
    add_bias_rows(out, batch, bias);
}

#[cfg(all(
    feature = "vendored-blas",
    target_arch = "aarch64",
    target_os = "linux"
))]
fn run_active_linear_with_bias(
    x_rows: &[f32],
    batch: usize,
    input_size: usize,
    weight_t: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    run_cblas_sgemm(batch, input_size, bias.len(), 1.0, x_rows, weight_t, out);
    add_bias_rows(out, batch, bias);
}

fn kernel_benchmark_cases() -> Result<Vec<KernelBenchmarkCase>, String> {
    let config = std::fs::read_to_string("assets/config.json")
        .map_err(|err| format!("failed to read config.json: {err}"))?;
    let config: ModelConfig = serde_json::from_str(&config)
        .map_err(|err| format!("failed to parse config.json: {err}"))?;
    let tokenizer = HFTokenizer::from_file("assets/tokenizer.json")
        .map_err(|err| format!("failed to load tokenizer: {err}"))?;

    gemm_microbenchmark_scenarios()
        .into_iter()
        .map(|scenario| {
            let token_lengths = scenario_token_lengths(&tokenizer, scenario, BENCHMARK_MAX_LENGTH)
                .map_err(|err| format!("failed to tokenize {}: {err}", scenario.name))?;
            let seq_len = padded_seq_len(&token_lengths);

            Ok(KernelBenchmarkCase {
                name: format!(
                    "{}-seq{}-tokens{}",
                    scenario.name.replace('/', "_"),
                    seq_len,
                    scenario.batch_size * seq_len
                ),
                batch_size: scenario.batch_size,
                seq_len,
                total_tokens: scenario.batch_size * seq_len,
                hidden_size: config.hidden_size,
                intermediate_size: config.intermediate_size,
                num_attention_heads: config.num_attention_heads,
            })
        })
        .collect()
}

// ── Benchmark groups ──────────────────────────────────────────────────────────

fn bench_ltembed_single(c: &mut Criterion) {
    if !assets_available() {
        eprintln!("bench_ltembed: skipping — assets/model.safetensors not found");
        return;
    }
    let engine = &*ENGINE;
    let long = long_input();
    let inputs = [
        ("short", SHORT),
        ("medium", MEDIUM),
        ("long", long.as_str()),
    ];

    let mut group = c.benchmark_group("bench_ltembed");
    for (label, text) in &inputs {
        group.bench_with_input(BenchmarkId::new("single", label), text, |b, t| {
            b.iter(|| engine.embed(criterion::black_box(t)).unwrap())
        });
    }
    group.finish();
}

fn bench_ltembed_batch(c: &mut Criterion) {
    if !assets_available() {
        eprintln!("bench_ltembed_batch: skipping — assets/model.safetensors not found");
        return;
    }
    let engine = &*ENGINE;

    let mut group = c.benchmark_group("bench_ltembed_batch");
    for &batch_size in &[1usize, 4, 8, 16] {
        let texts: Vec<&str> = std::iter::repeat_n(MEDIUM, batch_size).collect();
        group.bench_with_input(
            BenchmarkId::new("medium", batch_size),
            &batch_size,
            |b, _| b.iter(|| engine.embed_batch(criterion::black_box(&texts)).unwrap()),
        );
    }
    group.finish();
}

fn bench_ltembed_batch_parallel(c: &mut Criterion) {
    if !assets_available() {
        eprintln!("bench_ltembed_batch_parallel: skipping — assets/model.safetensors not found");
        return;
    }
    let engine = &*ENGINE;

    let mut group = c.benchmark_group("bench_ltembed_batch_parallel");
    // Total batch size fixed at 16; vary chunk_size to show scaling vs thread count.
    // chunk_size=16 → 1 rayon task (baseline, same as embed_batch)
    // chunk_size=8  → 2 tasks
    // chunk_size=4  → 4 tasks
    // chunk_size=2  → 8 tasks
    let texts: Vec<&str> = std::iter::repeat_n(MEDIUM, 16).collect();
    for &chunk_size in &[16usize, 8, 4, 2] {
        group.bench_with_input(
            BenchmarkId::new("chunk", chunk_size),
            &chunk_size,
            |b, &cs| {
                b.iter(|| {
                    engine
                        .embed_batch_rayon(criterion::black_box(&texts), cs)
                        .unwrap()
                })
            },
        );
    }
    group.finish();
}

fn bench_ltembed_concurrent(c: &mut Criterion) {
    if !assets_available() {
        eprintln!("bench_ltembed_concurrent: skipping — assets/model.safetensors not found");
        return;
    }
    let engine = &*ENGINE;

    let mut group = c.benchmark_group("bench_ltembed_concurrent");
    for &workers in &[2usize, 4] {
        group.bench_with_input(BenchmarkId::new("medium", workers), &workers, |b, &n| {
            b.iter(|| {
                thread::scope(|scope| {
                    for _ in 0..n {
                        scope.spawn(|| engine.embed(criterion::black_box(MEDIUM)).unwrap());
                    }
                });
            })
        });
    }
    group.finish();
}

fn bench_ltembed_kernel_projection(c: &mut Criterion) {
    if !kernel_assets_available() {
        eprintln!(
            "bench_ltembed_kernel_projection: skipping — assets/config.json or assets/tokenizer.json not found"
        );
        return;
    }
    let cases = match kernel_benchmark_cases() {
        Ok(cases) => cases,
        Err(err) => {
            eprintln!("bench_ltembed_kernel_projection: skipping — {err}");
            return;
        }
    };

    let mut group = c.benchmark_group("bench_ltembed_kernel_projection");
    for case in &cases {
        let total_tokens = case.total_tokens;
        let hidden = case.hidden_size;
        let intermediate = case.intermediate_size;

        let x = patterned_f32(total_tokens * hidden);
        let hidden_weight_t = patterned_f32(hidden * hidden);
        let inter_weight_t = patterned_f32(hidden * intermediate);
        let out_weight_t = patterned_f32(intermediate * hidden);

        let mut hidden_out = vec![0.0f32; total_tokens * hidden];
        let mut inter_out = vec![0.0f32; total_tokens * intermediate];

        group.bench_with_input(BenchmarkId::new("qkv", &case.name), case, |b, _case| {
            b.iter(|| {
                run_sgemm(
                    total_tokens,
                    hidden,
                    hidden,
                    1.0,
                    x.as_ptr(),
                    hidden as isize,
                    1,
                    hidden_weight_t.as_ptr(),
                    hidden as isize,
                    1,
                    hidden_out.as_mut_ptr(),
                    hidden as isize,
                    1,
                );
                run_sgemm(
                    total_tokens,
                    hidden,
                    hidden,
                    1.0,
                    x.as_ptr(),
                    hidden as isize,
                    1,
                    hidden_weight_t.as_ptr(),
                    hidden as isize,
                    1,
                    hidden_out.as_mut_ptr(),
                    hidden as isize,
                    1,
                );
                run_sgemm(
                    total_tokens,
                    hidden,
                    hidden,
                    1.0,
                    x.as_ptr(),
                    hidden as isize,
                    1,
                    hidden_weight_t.as_ptr(),
                    hidden as isize,
                    1,
                    hidden_out.as_mut_ptr(),
                    hidden as isize,
                    1,
                );
                criterion::black_box(&hidden_out);
            })
        });

        group.bench_with_input(
            BenchmarkId::new("attn_out", &case.name),
            case,
            |b, _case| {
                b.iter(|| {
                    run_sgemm(
                        total_tokens,
                        hidden,
                        hidden,
                        1.0,
                        x.as_ptr(),
                        hidden as isize,
                        1,
                        hidden_weight_t.as_ptr(),
                        hidden as isize,
                        1,
                        hidden_out.as_mut_ptr(),
                        hidden as isize,
                        1,
                    );
                    criterion::black_box(&hidden_out);
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("ffn_in", &case.name), case, |b, _case| {
            b.iter(|| {
                run_sgemm(
                    total_tokens,
                    hidden,
                    intermediate,
                    1.0,
                    x.as_ptr(),
                    hidden as isize,
                    1,
                    inter_weight_t.as_ptr(),
                    intermediate as isize,
                    1,
                    inter_out.as_mut_ptr(),
                    intermediate as isize,
                    1,
                );
                criterion::black_box(&inter_out);
            })
        });

        group.bench_with_input(BenchmarkId::new("ffn_out", &case.name), case, |b, _case| {
            b.iter(|| {
                run_sgemm(
                    total_tokens,
                    intermediate,
                    hidden,
                    1.0,
                    inter_out.as_ptr(),
                    intermediate as isize,
                    1,
                    out_weight_t.as_ptr(),
                    hidden as isize,
                    1,
                    hidden_out.as_mut_ptr(),
                    hidden as isize,
                    1,
                );
                criterion::black_box(&hidden_out);
            })
        });
    }
    group.finish();
}

fn bench_ltembed_kernel_projection_packing(c: &mut Criterion) {
    if !kernel_assets_available() {
        eprintln!(
            "bench_ltembed_kernel_projection_packing: skipping — assets/config.json or assets/tokenizer.json not found"
        );
        return;
    }
    let cases = match kernel_benchmark_cases() {
        Ok(cases) => cases,
        Err(err) => {
            eprintln!("bench_ltembed_kernel_projection_packing: skipping — {err}");
            return;
        }
    };

    let mut group = c.benchmark_group("bench_ltembed_kernel_projection_packing");
    for case in &cases {
        let shapes =
            projection_kernel_shapes(case.total_tokens, case.hidden_size, case.intermediate_size);

        for shape in shapes {
            let lhs = patterned_f32(shape.rows * shape.depth);
            let rhs = patterned_f32(shape.depth * shape.cols);
            let output = patterned_f32(shape.rows * shape.cols);
            let mut lhs_pack = vec![0.0f32; lhs.len()];
            let mut rhs_pack = vec![0.0f32; rhs.len()];
            let mut output_clear = output;

            group.bench_with_input(
                BenchmarkId::new(format!("{}_lhs_pack", shape.label), &case.name),
                case,
                |b, _case| {
                    b.iter(|| {
                        repeat_copy(&lhs, &mut lhs_pack, shape.repeats);
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("{}_rhs_pack", shape.label), &case.name),
                case,
                |b, _case| {
                    b.iter(|| {
                        repeat_copy(&rhs, &mut rhs_pack, shape.repeats);
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("{}_output_clear", shape.label), &case.name),
                case,
                |b, _case| {
                    b.iter(|| {
                        repeat_fill(&mut output_clear, shape.repeats);
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("{}_lhs_rhs_pack", shape.label), &case.name),
                case,
                |b, _case| {
                    b.iter(|| {
                        repeat_copy(&lhs, &mut lhs_pack, shape.repeats);
                        repeat_copy(&rhs, &mut rhs_pack, shape.repeats);
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("{}_total_setup", shape.label), &case.name),
                case,
                |b, _case| {
                    b.iter(|| {
                        repeat_copy(&lhs, &mut lhs_pack, shape.repeats);
                        repeat_copy(&rhs, &mut rhs_pack, shape.repeats);
                        repeat_fill(&mut output_clear, shape.repeats);
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_ltembed_kernel_projection_backends(c: &mut Criterion) {
    if !kernel_assets_available() {
        eprintln!(
            "bench_ltembed_kernel_projection_backends: skipping — assets/config.json or assets/tokenizer.json not found"
        );
        return;
    }
    let cases = match kernel_benchmark_cases() {
        Ok(cases) => cases,
        Err(err) => {
            eprintln!("bench_ltembed_kernel_projection_backends: skipping — {err}");
            return;
        }
    };

    let mut group = c.benchmark_group("bench_ltembed_kernel_projection_backends");
    for case in &cases {
        let shapes =
            projection_kernel_shapes(case.total_tokens, case.hidden_size, case.intermediate_size);

        for shape in shapes {
            let lhs = patterned_f32(shape.rows * shape.depth);
            let rhs = patterned_f32(shape.depth * shape.cols);
            let bias = patterned_f32(shape.cols);
            let mut out = vec![0.0f32; shape.rows * shape.cols];

            group.bench_with_input(
                BenchmarkId::new(format!("matrixmultiply_{}", shape.label), &case.name),
                case,
                |b, _case| {
                    b.iter(|| {
                        run_matrixmultiply_linear_with_bias(
                            &lhs,
                            shape.rows,
                            shape.depth,
                            &rhs,
                            &bias,
                            &mut out,
                        );
                        criterion::black_box(&out);
                    })
                },
            );

            #[cfg(all(
                feature = "vendored-blas",
                target_arch = "aarch64",
                target_os = "linux"
            ))]
            group.bench_with_input(
                BenchmarkId::new(
                    format!("{}_{}", dense_backend_name(), shape.label),
                    &case.name,
                ),
                case,
                |b, _case| {
                    b.iter(|| {
                        run_active_linear_with_bias(
                            &lhs,
                            shape.rows,
                            shape.depth,
                            &rhs,
                            &bias,
                            &mut out,
                        );
                        criterion::black_box(&out);
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_ltembed_kernel_attention_qk(c: &mut Criterion) {
    if !kernel_assets_available() {
        eprintln!(
            "bench_ltembed_kernel_attention_qk: skipping — assets/config.json or assets/tokenizer.json not found"
        );
        return;
    }
    let cases = match kernel_benchmark_cases() {
        Ok(cases) => cases,
        Err(err) => {
            eprintln!("bench_ltembed_kernel_attention_qk: skipping — {err}");
            return;
        }
    };

    let mut group = c.benchmark_group("bench_ltembed_kernel_attention_qk");
    for case in &cases {
        let hidden = case.hidden_size;
        let seq_len = case.seq_len;
        let head_dim = case.head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = patterned_f32(case.total_tokens * hidden);
        let k = patterned_f32(case.total_tokens * hidden);
        let mut scores = vec![0.0f32; seq_len * seq_len];

        group.bench_with_input(BenchmarkId::new("qk", &case.name), case, |b, case| {
            b.iter(|| {
                for batch_idx in 0..case.batch_size {
                    let hidden_offset = batch_idx * seq_len * hidden;
                    for head_idx in 0..case.num_attention_heads {
                        run_sgemm(
                            seq_len,
                            head_dim,
                            seq_len,
                            scale,
                            unsafe { q.as_ptr().add(hidden_offset + head_idx * head_dim) },
                            hidden as isize,
                            1,
                            unsafe { k.as_ptr().add(hidden_offset + head_idx * head_dim) },
                            1,
                            hidden as isize,
                            scores.as_mut_ptr(),
                            seq_len as isize,
                            1,
                        );
                    }
                }
                criterion::black_box(&scores);
            })
        });
    }
    group.finish();
}

fn bench_ltembed_kernel_attention_sv(c: &mut Criterion) {
    if !kernel_assets_available() {
        eprintln!(
            "bench_ltembed_kernel_attention_sv: skipping — assets/config.json or assets/tokenizer.json not found"
        );
        return;
    }
    let cases = match kernel_benchmark_cases() {
        Ok(cases) => cases,
        Err(err) => {
            eprintln!("bench_ltembed_kernel_attention_sv: skipping — {err}");
            return;
        }
    };

    let mut group = c.benchmark_group("bench_ltembed_kernel_attention_sv");
    for case in &cases {
        let hidden = case.hidden_size;
        let seq_len = case.seq_len;
        let head_dim = case.head_dim();
        let scores = patterned_f32(seq_len * seq_len);
        let v = patterned_f32(case.total_tokens * hidden);
        let mut attn_out = vec![0.0f32; case.total_tokens * hidden];

        group.bench_with_input(BenchmarkId::new("sv", &case.name), case, |b, case| {
            b.iter(|| {
                for batch_idx in 0..case.batch_size {
                    let hidden_offset = batch_idx * seq_len * hidden;
                    for head_idx in 0..case.num_attention_heads {
                        run_sgemm(
                            seq_len,
                            seq_len,
                            head_dim,
                            1.0,
                            scores.as_ptr(),
                            seq_len as isize,
                            1,
                            unsafe { v.as_ptr().add(hidden_offset + head_idx * head_dim) },
                            hidden as isize,
                            1,
                            unsafe {
                                attn_out
                                    .as_mut_ptr()
                                    .add(hidden_offset + head_idx * head_dim)
                            },
                            hidden as isize,
                            1,
                        );
                    }
                }
                criterion::black_box(&attn_out);
            })
        });
    }
    group.finish();
}

// ── Elementwise op microbenchmarks ────────────────────────────────────────────
//
// Measures LayerNorm, GELU, and Softmax in isolation at T=8/25/128/512.
// These T values match the matrixmultiply microbenchmarks for direct comparison.
//
// Shapes match the actual LTEmbed forward pass (e5-small-v2):
//   hidden=384, intermediate=1536
//
// Reporting: Throughput::Elements so Criterion shows elem/s alongside time.
// Each sub-benchmark measures ONE call (per-layer cost is shown in comments).

fn bench_ltembed_kernel_elementwise(c: &mut Criterion) {
    const HIDDEN: usize = 384;
    const INTERMEDIATE: usize = 1536;
    const T_VALUES: &[usize] = &[8, 25, 128, 512];

    // ── LayerNorm ─────────────────────────────────────────────────────────────
    // Shape: [T × HIDDEN]. Called 2× per layer (after attention + after FFN).
    {
        let mut group = c.benchmark_group("elementwise/layernorm");
        for &t in T_VALUES {
            let n = t * HIDDEN;
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new("T", t), &t, |b, &t| {
                let mut x = patterned_f32(t * HIDDEN);
                let weight = patterned_f32(HIDDEN);
                let bias = patterned_f32(HIDDEN);
                b.iter(|| {
                    layer_norm_rows(&mut x, t, HIDDEN, &weight, &bias);
                    criterion::black_box(&x);
                });
            });
        }
        group.finish();
    }

    // ── GELU ──────────────────────────────────────────────────────────────────
    // Shape: [T × INTERMEDIATE]. Called 1× per layer (FFN activation).
    {
        let mut group = c.benchmark_group("elementwise/gelu");
        for &t in T_VALUES {
            let n = t * INTERMEDIATE;
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new("T", t), &t, |b, &t| {
                let mut x = patterned_f32(t * INTERMEDIATE);
                b.iter(|| {
                    gelu(&mut x);
                    criterion::black_box(&x);
                });
            });
        }
        group.finish();
    }

    // ── Softmax: masked path (all-active mask, scalar) ────────────────────────
    // This exercises masked_softmax() with a dense all-ones mask.
    // NOT the hot path in production (see softmax_unmasked below).
    {
        let mut group = c.benchmark_group("elementwise/softmax_masked");
        for &t in T_VALUES {
            let n = t * t;
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new("T", t), &t, |b, &t| {
                let mut scores = patterned_f32(t * t);
                let mask = vec![1u32; t * t];
                b.iter(|| {
                    masked_softmax(&mut scores, &mask);
                    criterion::black_box(&scores);
                });
            });
        }
        group.finish();
    }

    // ── Softmax: unmasked path (SIMD, production hot path) ────────────────────
    // softmax_unmasked is called when all tokens are active (no padding).
    // This is the dominant path for single-sequence embedding inference.
    // Each call covers one attention head row: T elements.
    {
        let mut group = c.benchmark_group("elementwise/softmax_unmasked");
        for &t in T_VALUES {
            group.throughput(Throughput::Elements(t as u64));
            group.bench_with_input(BenchmarkId::new("T", t), &t, |b, &t| {
                let mut scores = patterned_f32(t);
                b.iter(|| {
                    softmax_unmasked(&mut scores);
                    criterion::black_box(&scores);
                });
            });
        }
        group.finish();
    }
}

criterion_group!(
    benches,
    bench_ltembed_single,
    bench_ltembed_batch,
    bench_ltembed_batch_parallel,
    bench_ltembed_concurrent,
    bench_ltembed_kernel_projection,
    bench_ltembed_kernel_projection_packing,
    bench_ltembed_kernel_projection_backends,
    bench_ltembed_kernel_attention_qk,
    bench_ltembed_kernel_attention_sv,
    bench_ltembed_kernel_elementwise
);
criterion_main!(benches);
