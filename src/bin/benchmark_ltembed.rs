use ltembed::benchmarking::{scenario_by_name, scenario_inputs, selected_scenarios, LatencyStats};
use ltembed::engine::{EmbeddingInput, EmbeddingInputKind, OnnxEngine, OnnxEngineConfig};
use ltembed::error::LTEmbedError;
use serde::Serialize;
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug)]
struct Args {
    mode: String,
    scenario: Option<String>,
    ort_bundle_dir: PathBuf,
    output_dimension: usize,
    l2_normalize: bool,
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
    results: Vec<StatsEntry>,
}

#[derive(Serialize)]
struct CorrectnessPayload {
    implementation: &'static str,
    implementation_version: String,
    results: Vec<EmbeddingsEntry>,
}

#[derive(Serialize)]
struct ColdPayload {
    implementation: &'static str,
    implementation_version: String,
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
    let mut ort_bundle_dir = None;
    let mut output_dimension = None;
    let mut l2_normalize = None;
    let mut warmup = 10usize;
    let mut iters = 100usize;
    let mut threads = 1usize;

    let mut iter = args.into_iter().map(Into::into);
    let _program_name = iter.next();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--mode" => mode = iter.next(),
            "--scenario" => scenario = iter.next(),
            "--ort-bundle-dir" => ort_bundle_dir = iter.next().map(PathBuf::from),
            "--output-dimension" => {
                output_dimension = Some(
                    iter.next()
                        .ok_or_else(|| "missing value for --output-dimension".to_string())?
                        .parse()
                        .map_err(|_| "invalid value for --output-dimension".to_string())?,
                )
            }
            "--l2-normalize" => {
                l2_normalize = Some(
                    match iter
                        .next()
                        .ok_or_else(|| "missing value for --l2-normalize".to_string())?
                        .as_str()
                    {
                        "true" => true,
                        "false" => false,
                        _ => return Err("invalid value for --l2-normalize".to_string()),
                    },
                )
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
        ort_bundle_dir: ort_bundle_dir
            .ok_or_else(|| "missing required --ort-bundle-dir".to_string())?,
        output_dimension: output_dimension
            .ok_or_else(|| "missing required --output-dimension".to_string())?,
        l2_normalize: l2_normalize.ok_or_else(|| "missing required --l2-normalize".to_string())?,
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

fn engine_from_bundle_dir(args: &Args) -> Result<OnnxEngine, LTEmbedError> {
    OnnxEngine::from_bundle_dir(
        Path::new(&args.ort_bundle_dir),
        OnnxEngineConfig {
            output_dimension: args.output_dimension,
            l2_normalize: args.l2_normalize,
        },
    )
}

fn run_scenario(engine: &OnnxEngine, scenario_name: &str) -> Result<Vec<Vec<f32>>, LTEmbedError> {
    let scenario = scenario_by_name(scenario_name)
        .ok_or_else(|| LTEmbedError::Inference("unknown scenario".into()))?;
    let benchmark_inputs = scenario_inputs(scenario);
    let inputs = benchmark_inputs
        .iter()
        .map(|input| match input.kind {
            EmbeddingInputKind::Query => EmbeddingInput::query(input.text.as_str()),
            EmbeddingInputKind::Document => EmbeddingInput::document(input.text.as_str()),
        })
        .collect::<Vec<_>>();
    engine.embed_batch(&inputs)
}

fn measure_warm_stats(
    engine: &OnnxEngine,
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

fn measure_cold_stats(args: &Args, scenario_name: &str) -> Result<LatencyStats, LTEmbedError> {
    let start = Instant::now();
    let engine = engine_from_bundle_dir(args)?;
    let _ = run_scenario(&engine, scenario_name)?;
    LatencyStats::from_samples_ms(&[start.elapsed().as_secs_f64() * 1_000.0])
        .map_err(LTEmbedError::Inference)
}

fn progress_label(mode: &str, scenario_name: &str, state: &str) -> String {
    format!("{mode} {scenario_name} {state}")
}

fn emit_progress(mode: &str, scenario_name: &str, state: &str) {
    eprintln!("{}", progress_label(mode, scenario_name, state));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args =
        parse_args().map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    let _threads = args.threads;
    let implementation_version = git_sha();

    match args.mode.as_str() {
        "warm" => {
            let engine = engine_from_bundle_dir(&args)?;
            let mut results = Vec::new();
            let scenarios = selected_scenarios(args.scenario.as_deref())
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
            for scenario in scenarios {
                emit_progress("warm", scenario.name, "start");
                let stats = measure_warm_stats(&engine, scenario.name, args.warmup, args.iters)?;
                emit_progress("warm", scenario.name, "done");
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
                    results,
                },
            )?;
        }
        "cold" => {
            let scenario_name = args.scenario.as_deref().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing --scenario")
            })?;
            emit_progress("cold", scenario_name, "start");
            let stats = measure_cold_stats(&args, scenario_name)?;
            emit_progress("cold", scenario_name, "done");
            serde_json::to_writer(
                std::io::stdout(),
                &ColdPayload {
                    implementation: "ltembed",
                    implementation_version,
                    scenario: scenario_name.to_string(),
                    stats,
                },
            )?;
        }
        "correctness" => {
            let engine = engine_from_bundle_dir(&args)?;
            let mut results = Vec::new();
            let scenarios = selected_scenarios(args.scenario.as_deref())
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
            for scenario in scenarios {
                emit_progress("correctness", scenario.name, "start");
                let embeddings = run_scenario(&engine, scenario.name)?;
                emit_progress("correctness", scenario.name, "done");
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
            "--ort-bundle-dir",
            "ort_bundle",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
        ])
        .unwrap();

        assert_eq!(args.mode, "warm");
        assert_eq!(args.scenario.as_deref(), Some("single/medium"));
        assert_eq!(args.ort_bundle_dir, PathBuf::from("ort_bundle"));
        assert_eq!(args.output_dimension, 512);
        assert!(args.l2_normalize);
    }

    #[test]
    fn test_progress_label_includes_mode_scenario_and_state() {
        assert_eq!(
            progress_label("correctness", "batch/mixed/8", "start"),
            "correctness batch/mixed/8 start"
        );
    }
}
