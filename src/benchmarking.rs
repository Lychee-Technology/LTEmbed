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

pub const SHORT_TEXT: &str = "query: Hello, world!";
pub const MEDIUM_TEXT: &str =
    "query: What is the impact of large language models on software engineering productivity?";

const BENCHMARK_SCENARIOS: [BenchmarkScenario; 7] = [
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
        name: "batch/medium/16",
        batch_size: 16,
        text_profile: "medium",
    },
];

pub fn benchmark_scenarios() -> &'static [BenchmarkScenario] {
    &BENCHMARK_SCENARIOS
}

pub fn scenario_by_name(name: &str) -> Option<&'static BenchmarkScenario> {
    benchmark_scenarios()
        .iter()
        .find(|scenario| scenario.name == name)
}

pub fn selected_scenarios(
    name: Option<&str>,
) -> Result<Vec<&'static BenchmarkScenario>, String> {
    match name {
        Some(name) => scenario_by_name(name)
            .map(|scenario| vec![scenario])
            .ok_or_else(|| format!("unknown scenario: {name}")),
        None => Ok(benchmark_scenarios().iter().collect()),
    }
}

pub fn scenario_texts(scenario: &BenchmarkScenario) -> Vec<String> {
    match scenario.name {
        "single/short" => vec![SHORT_TEXT.to_string()],
        "single/medium" => vec![MEDIUM_TEXT.to_string()],
        "single/long" => vec![long_text()],
        "batch/medium/1" | "batch/medium/4" | "batch/medium/8" | "batch/medium/16" => {
            std::iter::repeat_n(MEDIUM_TEXT.to_string(), scenario.batch_size).collect()
        }
        _ => Vec::new(),
    }
}

pub fn long_text() -> String {
    "passage: ".to_string() + &"The quick brown fox jumps over the lazy dog. ".repeat(30)
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
