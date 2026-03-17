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

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ltembed::{engine::ZeroVecEngine, traits::pooling::MeanPooling};
use once_cell::sync::Lazy;
use std::path::Path;

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

criterion_group!(benches, bench_ltembed_single, bench_ltembed_batch);
criterion_main!(benches);
