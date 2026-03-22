use ltembed::benchmarking::{
    dense_backend_name, scenario_by_name, scenario_texts, selected_scenarios, LatencyStats,
};
use ltembed::engine::ZeroVecEngine;
use ltembed::error::LTEmbedError;
use ltembed::traits::pooling::MeanPooling;
use serde::Serialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug)]
struct Args {
    mode: String,
    scenario: Option<String>,
    model_dir: PathBuf,
    model_size: String,
    warmup: usize,
    iters: usize,
    threads: usize,
}

#[derive(Serialize)]
struct StatsEntry {
    scenario: String,
    stats: LatencyStats,
}

#[derive(Serialize)]
struct EmbeddingsEntry {
    scenario: String,
    embeddings: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct WarmPayload {
    implementation: &'static str,
    implementation_version: String,
    backend: &'static str,
    results: Vec<StatsEntry>,
}

#[derive(Serialize)]
struct CorrectnessPayload {
    implementation: &'static str,
    implementation_version: String,
    backend: &'static str,
    results: Vec<EmbeddingsEntry>,
}

#[derive(Serialize)]
struct ColdPayload {
    implementation: &'static str,
    implementation_version: String,
    backend: &'static str,
    scenario: String,
    stats: LatencyStats,
}

fn parse_args_from<I, S>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut mode = None;
    let mut scenario = None;
    let mut model_dir = None;
    let mut model_size = "fp32".to_string();
    let mut warmup = 10usize;
    let mut iters = 100usize;
    let mut threads = 1usize;

    let mut iter = args.into_iter().map(Into::into);
    let _program_name = iter.next();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--mode" => mode = iter.next(),
            "--scenario" => scenario = iter.next(),
            "--model-dir" => model_dir = iter.next().map(PathBuf::from),
            "--model-size" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "missing value for --model-size".to_string())?;
                if v != "fp16" && v != "fp32" {
                    return Err(format!(
                        "invalid value for --model-size: '{v}' (expected fp16 or fp32)"
                    ));
                }
                model_size = v;
            }
            "--warmup" => {
                warmup = iter
                    .next()
                    .ok_or_else(|| "missing value for --warmup".to_string())?
                    .parse()
                    .map_err(|_| "invalid value for --warmup".to_string())?
            }
            "--iters" => {
                iters = iter
                    .next()
                    .ok_or_else(|| "missing value for --iters".to_string())?
                    .parse()
                    .map_err(|_| "invalid value for --iters".to_string())?
            }
            "--threads" => {
                threads = iter
                    .next()
                    .ok_or_else(|| "missing value for --threads".to_string())?
                    .parse()
                    .map_err(|_| "invalid value for --threads".to_string())?
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        mode: mode.ok_or_else(|| "missing required --mode".to_string())?,
        scenario,
        model_dir: model_dir.ok_or_else(|| "missing required --model-dir".to_string())?,
        model_size,
        warmup,
        iters,
        threads,
    })
}

fn parse_args() -> Result<Args, String> {
    parse_args_from(env::args())
}

fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
            } else {
                None
            }
        })
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string())
}

fn engine_from_model_dir(
    model_dir: &Path,
    model_size: &str,
) -> Result<ZeroVecEngine, LTEmbedError> {
    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let safetensors_filename = if model_size == "fp16" {
        "model_fp16.safetensors"
    } else {
        "model.safetensors"
    };
    let safetensors_path = model_dir.join(safetensors_filename);
    let config = fs::read_to_string(config_path)?;
    ZeroVecEngine::new(
        &safetensors_path.to_string_lossy(),
        &config,
        &tokenizer_path.to_string_lossy(),
        Box::new(MeanPooling),
    )
}

fn run_scenario(
    engine: &ZeroVecEngine,
    scenario_name: &str,
) -> Result<Vec<Vec<f32>>, LTEmbedError> {
    let scenario = scenario_by_name(scenario_name)
        .ok_or_else(|| LTEmbedError::Inference("unknown scenario".into()))?;
    let texts = scenario_texts(scenario);
    let refs = texts.iter().map(String::as_str).collect::<Vec<_>>();
    engine.embed_batch(&refs)
}

fn measure_warm_stats(
    engine: &ZeroVecEngine,
    scenario_name: &str,
    warmup: usize,
    iters: usize,
) -> Result<LatencyStats, LTEmbedError> {
    for _ in 0..warmup {
        let _ = run_scenario(engine, scenario_name)?;
    }

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let _ = run_scenario(engine, scenario_name)?;
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }

    LatencyStats::from_samples_ms(&samples).map_err(LTEmbedError::Inference)
}

fn measure_cold_stats(
    model_dir: &Path,
    model_size: &str,
    scenario_name: &str,
) -> Result<LatencyStats, LTEmbedError> {
    let start = Instant::now();
    let engine = engine_from_model_dir(model_dir, model_size)?;
    let _ = run_scenario(&engine, scenario_name)?;
    LatencyStats::from_samples_ms(&[start.elapsed().as_secs_f64() * 1_000.0])
        .map_err(LTEmbedError::Inference)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args =
        parse_args().map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    let _threads = args.threads;
    let implementation_version = git_sha();
    let backend = dense_backend_name();

    match args.mode.as_str() {
        "warm" => {
            let engine = engine_from_model_dir(&args.model_dir, &args.model_size)?;
            let mut results = Vec::new();
            let scenarios = selected_scenarios(args.scenario.as_deref())
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
            for scenario in scenarios {
                let stats = measure_warm_stats(&engine, scenario.name, args.warmup, args.iters)?;
                results.push(StatsEntry {
                    scenario: scenario.name.to_string(),
                    stats,
                });
            }
            serde_json::to_writer(
                std::io::stdout(),
                &WarmPayload {
                    implementation: "ltembed",
                    implementation_version,
                    backend,
                    results,
                },
            )?;
        }
        "cold" => {
            let scenario_name = args.scenario.as_deref().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing --scenario")
            })?;
            let stats = measure_cold_stats(&args.model_dir, &args.model_size, scenario_name)?;
            serde_json::to_writer(
                std::io::stdout(),
                &ColdPayload {
                    implementation: "ltembed",
                    implementation_version,
                    backend,
                    scenario: scenario_name.to_string(),
                    stats,
                },
            )?;
        }
        "correctness" => {
            let engine = engine_from_model_dir(&args.model_dir, &args.model_size)?;
            let mut results = Vec::new();
            let scenarios = selected_scenarios(args.scenario.as_deref())
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
            for scenario in scenarios {
                let embeddings = run_scenario(&engine, scenario.name)?;
                results.push(EmbeddingsEntry {
                    scenario: scenario.name.to_string(),
                    embeddings,
                });
            }
            serde_json::to_writer(
                std::io::stdout(),
                &CorrectnessPayload {
                    implementation: "ltembed",
                    implementation_version,
                    backend,
                    results,
                },
            )?;
        }
        other => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unknown mode: {other}"),
            )
            .into())
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_accepts_optional_scenario_for_warm_mode() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "warm",
            "--scenario",
            "single/medium",
            "--model-dir",
            "assets",
        ])
        .unwrap();

        assert_eq!(args.mode, "warm");
        assert_eq!(args.scenario.as_deref(), Some("single/medium"));
        assert_eq!(args.model_dir, PathBuf::from("assets"));
    }
}
