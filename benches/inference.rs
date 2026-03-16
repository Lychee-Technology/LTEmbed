// benches/inference.rs
//
// Warm-invocation benchmarks comparing:
//   1. LTEmbed  — ZeroVecEngine::embed() public API
//   2. Candle raw — same pipeline calling candle/tokenizers APIs directly
//
// Run with:
//   cargo bench
//   RUSTFLAGS="-C target-cpu=native" cargo bench   # enable NEON on Apple Silicon
//
// Requires: assets/model.safetensors (download separately)
// Skips gracefully if the model file is absent.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ltembed::{
    engine::ZeroVecEngine,
    traits::pooling::MeanPooling,
    utils::l2_normalize,
};
use once_cell::sync::Lazy;
use std::path::Path;

// ── Test inputs ──────────────────────────────────────────────────────────────

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

// Shared candle resources for the raw benchmark
static CANDLE_MODEL: Lazy<(BertModel, tokenizers::Tokenizer, Device)> = Lazy::new(|| {
    let device = Device::Cpu;
    let config_str = std::fs::read_to_string("assets/config.json").unwrap();
    let config: BertConfig = serde_json::from_str(&config_str).unwrap();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&["assets/model.safetensors"], DType::F32, &device)
            .unwrap()
    };
    let model = BertModel::load(vb, &config).unwrap();
    let tokenizer = tokenizers::Tokenizer::from_file("assets/tokenizer.json").unwrap();
    (model, tokenizer, device)
});

// ── Candle raw: inline mean pool + l2 normalize ───────────────────────────────

fn candle_raw_embed(text: &str) -> Vec<f32> {
    let (model, tokenizer, device) = &*CANDLE_MODEL;

    let encoding = tokenizer.encode(text, true).unwrap();
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    let mask: Vec<u32> = encoding.get_attention_mask().to_vec();
    let type_ids: Vec<u32> = encoding.get_type_ids().to_vec();
    let seq_len = ids.len();

    let ids_t = Tensor::from_vec(ids, (1, seq_len), device).unwrap();
    let type_ids_t = Tensor::from_vec(type_ids, (1, seq_len), device).unwrap();
    let mask_t = Tensor::from_vec(mask.clone(), (1, seq_len), device).unwrap();

    let output = model
        .forward(&ids_t, &type_ids_t, Some(&mask_t))
        .unwrap();
    let hidden: Vec<Vec<f32>> = output.squeeze(0).unwrap().to_vec2::<f32>().unwrap();

    // Mean pool (non-padding tokens only)
    let hidden_size = hidden[0].len();
    let mut sum = vec![0.0f32; hidden_size];
    let mut count = 0u32;
    for (row, &m) in hidden.iter().zip(mask.iter()) {
        if m == 1 {
            for (s, v) in sum.iter_mut().zip(row.iter()) {
                *s += v;
            }
            count += 1;
        }
    }
    let pooled: Vec<f32> = sum.iter().map(|x| x / count as f32).collect();
    l2_normalize(&pooled)
}

// ── Benchmark groups ─────────────────────────────────────────────────────────

fn bench_ltembed_single(c: &mut Criterion) {
    if !assets_available() {
        eprintln!("bench_ltembed: skipping — assets/model.safetensors not found");
        return;
    }
    let engine = &*ENGINE;
    let long = long_input();
    let inputs = [("short", SHORT), ("medium", MEDIUM), ("long", long.as_str())];

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
        let texts: Vec<&str> = std::iter::repeat(MEDIUM).take(batch_size).collect();
        group.bench_with_input(
            BenchmarkId::new("medium", batch_size),
            &batch_size,
            |b, _| b.iter(|| engine.embed_batch(criterion::black_box(&texts)).unwrap()),
        );
    }
    group.finish();
}

fn bench_candle_raw(c: &mut Criterion) {
    if !assets_available() {
        eprintln!("bench_candle_raw: skipping — assets/model.safetensors not found");
        return;
    }
    // Force Lazy initialization outside the timed loop
    let _ = &*CANDLE_MODEL;

    let long = long_input();
    let inputs = [("short", SHORT), ("medium", MEDIUM), ("long", long.as_str())];

    let mut group = c.benchmark_group("bench_candle_raw");
    for (label, text) in &inputs {
        group.bench_with_input(BenchmarkId::new("single", label), text, |b, t| {
            b.iter(|| candle_raw_embed(criterion::black_box(t)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ltembed_single, bench_ltembed_batch, bench_candle_raw);
criterion_main!(benches);
