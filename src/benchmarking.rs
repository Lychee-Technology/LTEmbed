use crate::engine::{EmbeddingInputKind, DOCUMENT_PREFIX, QUERY_PREFIX};
use crate::error::LTEmbedError;
use crate::traits::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkInput {
    pub text: String,
    pub kind: EmbeddingInputKind,
}

pub const ZH_TEXT: &str = "他感冒了";
pub const EN_TEXT: &str = "He caught a cold.";
pub const BENCHMARK_MAX_LENGTH: usize = 8192;

const BENCHMARK_SCENARIOS: [BenchmarkScenario; 5] = [
    BenchmarkScenario {
        name: "single/zh",
        batch_size: 1,
        text_profile: "zh",
    },
    BenchmarkScenario {
        name: "single/en",
        batch_size: 1,
        text_profile: "en",
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
        name: "batch/medium/8",
        batch_size: 8,
        text_profile: "medium",
    },
];

/// Deterministic texts for the medium/long/batch scenarios, shared byte-for-byte with the
/// Python side (`scripts/build_cn_en_retrieval_cases.py`) so fixture-driven CI runs and
/// built-in fallbacks embed identical inputs.
#[derive(Debug, Deserialize)]
struct BenchmarkCorpus {
    medium: CorpusEntry,
    long: CorpusEntry,
    batch_medium_8: Vec<CorpusEntry>,
}

#[derive(Debug, Deserialize)]
struct CorpusEntry {
    kind: CorpusKind,
    text: String,
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum CorpusKind {
    Query,
    Document,
}

static BENCHMARK_CORPUS: LazyLock<BenchmarkCorpus> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../scripts/benchmark_corpus.json"))
        .expect("scripts/benchmark_corpus.json must parse as the benchmark corpus")
});

pub fn benchmark_scenarios() -> &'static [BenchmarkScenario] {
    &BENCHMARK_SCENARIOS
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
        "single/zh" => vec![query_input(ZH_TEXT)],
        "single/en" => vec![query_input(EN_TEXT)],
        "single/medium" => vec![corpus_input(&BENCHMARK_CORPUS.medium)],
        "single/long" => vec![corpus_input(&BENCHMARK_CORPUS.long)],
        "batch/medium/8" => BENCHMARK_CORPUS
            .batch_medium_8
            .iter()
            .map(corpus_input)
            .collect(),
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

fn query_input(text: &str) -> BenchmarkInput {
    BenchmarkInput {
        text: text.to_string(),
        kind: EmbeddingInputKind::Query,
    }
}

fn corpus_input(entry: &CorpusEntry) -> BenchmarkInput {
    BenchmarkInput {
        text: entry.text.clone(),
        kind: match entry.kind {
            CorpusKind::Query => EmbeddingInputKind::Query,
            CorpusKind::Document => EmbeddingInputKind::Document,
        },
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
