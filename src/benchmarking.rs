use crate::engine::{EmbeddingInputKind, DOCUMENT_PREFIX, QUERY_PREFIX};
use crate::error::LTEmbedError;
use crate::gemm;
use crate::traits::tokenizer::Tokenizer;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkScenario {
    pub name: &'static str,
    pub batch_size: usize,
    pub text_profile: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencyStats {
    pub mean_ms: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionKernelShape {
    pub label: &'static str,
    pub rows: usize,
    pub depth: usize,
    pub cols: usize,
    pub repeats: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkInput {
    pub text: String,
    pub kind: EmbeddingInputKind,
}

pub const SHORT_TEXT: &str = "Hello, world!";
pub const MEDIUM_TEXT: &str =
    "What is the impact of large language models on software engineering productivity?";
pub const BENCHMARK_MAX_LENGTH: usize = 8192;
const GEMM_MICROBENCHMARK_SCENARIO_NAMES: [&str; 3] =
    ["single/long", "batch/medium/8", "batch/medium/16"];

const BENCHMARK_SCENARIOS: [BenchmarkScenario; 8] = [
    BenchmarkScenario {
        name: "single/short",
        batch_size: 1,
        text_profile: "short",
    },
    BenchmarkScenario {
        name: "single/medium",
        batch_size: 1,
        text_profile: "medium",
    },
    BenchmarkScenario {
        name: "single/long",
        batch_size: 1,
        text_profile: "long",
    },
    BenchmarkScenario {
        name: "batch/medium/1",
        batch_size: 1,
        text_profile: "medium",
    },
    BenchmarkScenario {
        name: "batch/medium/4",
        batch_size: 4,
        text_profile: "medium",
    },
    BenchmarkScenario {
        name: "batch/medium/8",
        batch_size: 8,
        text_profile: "medium",
    },
    BenchmarkScenario {
        name: "batch/mixed/8",
        batch_size: 8,
        text_profile: "mixed",
    },
    BenchmarkScenario {
        name: "batch/medium/16",
        batch_size: 16,
        text_profile: "medium",
    },
];

pub fn benchmark_scenarios() -> &'static [BenchmarkScenario] {
    &BENCHMARK_SCENARIOS
}

pub fn gemm_microbenchmark_scenarios() -> Vec<&'static BenchmarkScenario> {
    GEMM_MICROBENCHMARK_SCENARIO_NAMES
        .iter()
        .filter_map(|name| scenario_by_name(name))
        .collect()
}

pub fn scenario_by_name(name: &str) -> Option<&'static BenchmarkScenario> {
    benchmark_scenarios()
        .iter()
        .find(|scenario| scenario.name == name)
}

pub fn selected_scenarios(name: Option<&str>) -> Result<Vec<&'static BenchmarkScenario>, String> {
    match name {
        Some(name) => scenario_by_name(name)
            .map(|scenario| vec![scenario])
            .ok_or_else(|| format!("unknown scenario: {name}")),
        None => Ok(benchmark_scenarios().iter().collect()),
    }
}

pub fn scenario_inputs(scenario: &BenchmarkScenario) -> Vec<BenchmarkInput> {
    match scenario.name {
        "single/short" => vec![query_input(SHORT_TEXT)],
        "single/medium" => vec![query_input(MEDIUM_TEXT)],
        "single/long" => vec![document_input(&long_text())],
        "batch/medium/1" | "batch/medium/4" | "batch/medium/8" | "batch/medium/16" => {
            std::iter::repeat_with(|| query_input(MEDIUM_TEXT))
                .take(scenario.batch_size)
                .collect()
        }
        "batch/mixed/8" => vec![
            query_input(SHORT_TEXT),
            query_input(MEDIUM_TEXT),
            document_input(&long_text()),
            query_input(SHORT_TEXT),
            query_input(MEDIUM_TEXT),
            document_input(&long_text()),
            query_input(SHORT_TEXT),
            query_input(MEDIUM_TEXT),
        ],
        _ => Vec::new(),
    }
}

pub fn scenario_token_lengths<T: Tokenizer>(
    tokenizer: &T,
    scenario: &BenchmarkScenario,
    max_length: usize,
) -> Result<Vec<usize>, LTEmbedError> {
    scenario_inputs(scenario)
        .into_iter()
        .map(|input| {
            tokenizer
                .encode(&prefixed_text(&input), max_length)
                .map(|encoded| encoded.input_ids.len())
        })
        .collect()
}

pub fn padded_seq_len(token_lengths: &[usize]) -> usize {
    token_lengths.iter().copied().max().unwrap_or(0)
}

pub fn projection_kernel_shapes(
    total_tokens: usize,
    hidden: usize,
    intermediate: usize,
) -> [ProjectionKernelShape; 4] {
    [
        ProjectionKernelShape {
            label: "qkv_triplet",
            rows: total_tokens,
            depth: hidden,
            cols: hidden,
            repeats: 3,
        },
        ProjectionKernelShape {
            label: "attn_out",
            rows: total_tokens,
            depth: hidden,
            cols: hidden,
            repeats: 1,
        },
        ProjectionKernelShape {
            label: "ffn_in",
            rows: total_tokens,
            depth: hidden,
            cols: intermediate,
            repeats: 1,
        },
        ProjectionKernelShape {
            label: "ffn_out",
            rows: total_tokens,
            depth: intermediate,
            cols: hidden,
            repeats: 1,
        },
    ]
}

pub fn dense_backend_name() -> &'static str {
    gemm::dense_backend_name()
}

impl ProjectionKernelShape {
    pub fn lhs_pack_bytes(&self) -> usize {
        self.repeats * self.rows * self.depth * std::mem::size_of::<f32>()
    }

    pub fn rhs_pack_bytes(&self) -> usize {
        self.repeats * self.depth * self.cols * std::mem::size_of::<f32>()
    }

    pub fn output_bytes(&self) -> usize {
        self.repeats * self.rows * self.cols * std::mem::size_of::<f32>()
    }

    pub fn setup_bytes(&self) -> usize {
        self.lhs_pack_bytes() + self.rhs_pack_bytes() + self.output_bytes()
    }
}

pub fn long_text() -> String {
    "The quick brown fox jumps over the lazy dog. ".repeat(30)
}

fn query_input(text: &str) -> BenchmarkInput {
    BenchmarkInput {
        text: text.to_string(),
        kind: EmbeddingInputKind::Query,
    }
}

fn document_input(text: &str) -> BenchmarkInput {
    BenchmarkInput {
        text: text.to_string(),
        kind: EmbeddingInputKind::Document,
    }
}

fn prefixed_text(input: &BenchmarkInput) -> String {
    match input.kind {
        EmbeddingInputKind::Query => format!("{QUERY_PREFIX}{}", input.text),
        EmbeddingInputKind::Document => format!("{DOCUMENT_PREFIX}{}", input.text),
    }
}

impl LatencyStats {
    pub fn from_samples_ms(samples_ms: &[f64]) -> Result<Self, String> {
        if samples_ms.is_empty() {
            return Err("empty latency sample set".to_string());
        }

        let mean_ms = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
        let mut sorted = samples_ms.to_vec();
        sorted.sort_by(f64::total_cmp);

        Ok(Self {
            mean_ms,
            median_ms: percentile_linear(&sorted, 50.0),
            p95_ms: percentile_linear(&sorted, 95.0),
            p99_ms: percentile_linear(&sorted, 99.0),
            min_ms: *sorted.first().unwrap_or(&0.0),
            max_ms: *sorted.last().unwrap_or(&0.0),
        })
    }
}

fn percentile_linear(sorted_samples: &[f64], percentile: f64) -> f64 {
    if sorted_samples.len() == 1 {
        return sorted_samples[0];
    }

    let rank = percentile.clamp(0.0, 100.0) / 100.0 * (sorted_samples.len() - 1) as f64;
    let lower_index = rank.floor() as usize;
    let upper_index = rank.ceil() as usize;
    if lower_index == upper_index {
        return sorted_samples[lower_index];
    }

    let weight = rank - lower_index as f64;
    sorted_samples[lower_index] * (1.0 - weight) + sorted_samples[upper_index] * weight
}
