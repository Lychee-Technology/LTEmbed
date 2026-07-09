use ltembed::benchmarking::{
    scenario_by_name, scenario_inputs, selected_scenarios, BenchmarkInput, LatencyStats,
};
use ltembed::engine::{
    EmbedBatchProfile, EmbeddingEngine, EmbeddingInput, EmbeddingInputKind, EngineConfig,
};
use ltembed::error::{InferenceError, LTEmbedError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, PartialEq)]
enum Mode {
    Warm,
    Cold,
    Retrieval,
    Other(String),
}

impl From<&str> for Mode {
    fn from(value: &str) -> Self {
        match value {
            "warm" => Mode::Warm,
            "cold" => Mode::Cold,
            "retrieval" => Mode::Retrieval,
            other => Mode::Other(other.to_string()),
        }
    }
}

#[derive(Debug)]
struct Args {
    mode: Mode,
    scenario: Option<String>,
    bundle_dir: PathBuf,
    retrieval_eval_path: Option<PathBuf>,
    fixture_path: Option<PathBuf>,
    output_dimension: usize,
    l2_normalize: bool,
    warmup: usize,
    iters: usize,
    threads: usize,
}

/// Resolved benchmark fixture: per-scenario texts selected upstream (e.g. from the
/// jane-austen corpus) by the orchestrator. When present it overrides the built-in
/// synthetic scenario texts so both this binary and the PyTorch reference embed
/// byte-identical inputs, which is what makes the cosine comparison meaningful.
#[derive(Debug, Deserialize)]
struct ResolvedFixture {
    scenarios: HashMap<String, Vec<FixtureInput>>,
}

#[derive(Debug, Deserialize)]
struct FixtureInput {
    kind: String,
    text: String,
}

fn load_resolved_fixture(path: &Path) -> io::Result<ResolvedFixture> {
    let contents = fs::read_to_string(path)?;
    serde_json::from_str(&contents).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}

/// Build the inputs for a scenario, preferring the resolved fixture (keyed by scenario
/// name) and falling back to the built-in `scenario_inputs` when no fixture is supplied.
fn scenario_inputs_resolved(
    scenario_name: &str,
    fixture: Option<&ResolvedFixture>,
) -> Result<Vec<BenchmarkInput>, LTEmbedError> {
    if let Some(fixture) = fixture {
        let items = fixture.scenarios.get(scenario_name).ok_or_else(|| {
            LTEmbedError::Inference(InferenceError::Internal(format!(
                "fixture is missing scenario: {scenario_name}"
            )))
        })?;
        return items
            .iter()
            .map(|item| {
                let kind = match item.kind.as_str() {
                    "query" => EmbeddingInputKind::Query,
                    "document" => EmbeddingInputKind::Document,
                    other => {
                        return Err(LTEmbedError::Inference(InferenceError::Internal(format!(
                            "fixture input has unknown kind: {other}"
                        ))))
                    }
                };
                Ok(BenchmarkInput {
                    text: item.text.clone(),
                    kind,
                })
            })
            .collect();
    }

    let scenario = scenario_by_name(scenario_name).ok_or_else(|| {
        LTEmbedError::Inference(InferenceError::Internal("unknown scenario".to_string()))
    })?;
    Ok(scenario_inputs(scenario))
}

#[derive(Serialize)]
struct StatsEntry {
    scenario: String,
    stats: LatencyStats,
}

#[derive(Serialize)]
struct WarmPayload {
    implementation: &'static str,
    implementation_version: String,
    results: Vec<StatsEntry>,
}

#[derive(Serialize)]
struct ColdPayload {
    implementation: &'static str,
    implementation_version: String,
    scenario: String,
    stats: LatencyStats,
}

#[derive(Serialize)]
struct RetrievalEmbedding {
    id: String,
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct RetrievalResult {
    dataset_name: String,
    queries: Vec<RetrievalEmbedding>,
    documents: Vec<RetrievalEmbedding>,
}

#[derive(Serialize)]
struct RetrievalPayload {
    implementation: &'static str,
    implementation_version: String,
    results: Vec<RetrievalResult>,
}

#[derive(Debug, Deserialize)]
struct RetrievalEvalDocument {
    id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct RetrievalEvalQuery {
    id: String,
    text: String,
    #[serde(rename = "relevant_document_ids")]
    #[allow(dead_code)]
    relevant_document_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RetrievalEvalCase {
    name: String,
    documents: Vec<RetrievalEvalDocument>,
    queries: Vec<RetrievalEvalQuery>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RetrievalEvalSpec {
    Suite { cases: Vec<RetrievalEvalCase> },
}

#[derive(Debug, Default)]
struct ProfileAccumulator {
    samples: usize,
    batch_size_sum: usize,
    seq_len_sum: usize,
    prefix_ms_sum: f64,
    tokenize_ms_sum: f64,
    tensorize_ms_sum: f64,
    run_ms_sum: f64,
    extract_ms_sum: f64,
    postprocess_ms_sum: f64,
    total_ms_sum: f64,
}

fn parse_args_from<I, S>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut mode = None;
    let mut scenario = None;
    let mut bundle_dir = None;
    let mut retrieval_eval_path = None;
    let mut fixture_path = None;
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
            "--bundle-dir" => bundle_dir = iter.next().map(PathBuf::from),
            "--retrieval-eval-path" => retrieval_eval_path = iter.next().map(PathBuf::from),
            "--fixture-path" => fixture_path = iter.next().map(PathBuf::from),
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
                    .map_err(|_| "invalid value for --threads".to_string())?;
                if threads == 0 {
                    return Err("--threads must be greater than zero".to_string());
                }
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        mode: Mode::from(
            mode.ok_or_else(|| "missing required --mode".to_string())?
                .as_str(),
        ),
        scenario,
        bundle_dir: bundle_dir.ok_or_else(|| "missing required --bundle-dir".to_string())?,
        retrieval_eval_path,
        fixture_path,
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

fn engine_from_bundle_dir(args: &Args) -> Result<EmbeddingEngine, LTEmbedError> {
    EmbeddingEngine::from_gguf_bundle_dir_with_threads(
        Path::new(&args.bundle_dir),
        EngineConfig {
            output_dimension: args.output_dimension,
            l2_normalize: args.l2_normalize,
        },
        args.threads,
    )
}

fn resolve_scenarios(
    args: &Args,
) -> io::Result<Vec<&'static ltembed::benchmarking::BenchmarkScenario>> {
    selected_scenarios(args.scenario.as_deref())
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))
}

fn required_scenario(args: &Args) -> io::Result<&str> {
    args.scenario
        .as_deref()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing --scenario"))
}

fn required_retrieval_eval_path(args: &Args) -> io::Result<&Path> {
    args.retrieval_eval_path
        .as_deref()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing --retrieval-eval-path"))
}

fn load_retrieval_eval_cases(path: &Path) -> io::Result<Vec<RetrievalEvalCase>> {
    let contents = fs::read_to_string(path)?;
    let spec: RetrievalEvalSpec = serde_json::from_str(&contents)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    match spec {
        RetrievalEvalSpec::Suite { cases } => Ok(cases),
    }
}

fn embed_retrieval_inputs(
    engine: &EmbeddingEngine,
    inputs: Vec<EmbeddingInput<'_>>,
) -> Result<Vec<Vec<f32>>, LTEmbedError> {
    engine.embed_batch(&inputs)
}

fn run_scenario(
    engine: &EmbeddingEngine,
    scenario_name: &str,
    fixture: Option<&ResolvedFixture>,
) -> Result<Vec<Vec<f32>>, LTEmbedError> {
    let (embeddings, _) = run_scenario_maybe_profiled(engine, scenario_name, fixture, false)?;
    Ok(embeddings)
}

fn run_scenario_profiled(
    engine: &EmbeddingEngine,
    scenario_name: &str,
    fixture: Option<&ResolvedFixture>,
) -> Result<(Vec<Vec<f32>>, EmbedBatchProfile), LTEmbedError> {
    let (embeddings, profile) = run_scenario_maybe_profiled(engine, scenario_name, fixture, true)?;
    let profile = profile.ok_or_else(|| {
        LTEmbedError::Inference(InferenceError::Internal(
            "profiling requested but no profile was collected".to_string(),
        ))
    })?;
    Ok((embeddings, profile))
}

fn run_scenario_maybe_profiled(
    engine: &EmbeddingEngine,
    scenario_name: &str,
    fixture: Option<&ResolvedFixture>,
    collect_profile: bool,
) -> Result<(Vec<Vec<f32>>, Option<EmbedBatchProfile>), LTEmbedError> {
    let benchmark_inputs = scenario_inputs_resolved(scenario_name, fixture)?;
    let inputs = benchmark_inputs
        .iter()
        .map(|input| match input.kind {
            EmbeddingInputKind::Query => EmbeddingInput::query(input.text.as_str()),
            EmbeddingInputKind::Document => EmbeddingInput::document(input.text.as_str()),
        })
        .collect::<Vec<_>>();
    if collect_profile {
        let (embeddings, profile) = engine.embed_batch_profiled(&inputs)?;
        Ok((embeddings, Some(profile)))
    } else {
        Ok((engine.embed_batch(&inputs)?, None))
    }
}

fn profiling_enabled_from_value(value: Option<&str>) -> bool {
    matches!(value, Some("1" | "true" | "TRUE" | "yes" | "YES"))
}

fn profiling_enabled_from_env() -> bool {
    let value = env::var("LTEMBED_PROFILE").ok();
    profiling_enabled_from_value(value.as_deref())
}

fn profile_summary_line(scenario_name: &str, samples: usize, profile: EmbedBatchProfile) -> String {
    format!(
        "profile {scenario_name} samples={samples} batch_size={} seq_len={} prefix_ms={:.3} tokenize_ms={:.3} tensorize_ms={:.3} run_ms={:.3} extract_ms={:.3} postprocess_ms={:.3} total_ms={:.3}",
        profile.batch_size,
        profile.sequence_length,
        profile.prefix_ms,
        profile.tokenize_ms,
        profile.tensorize_ms,
        profile.run_ms,
        profile.extract_ms,
        profile.postprocess_ms,
        profile.total_ms,
    )
}

impl ProfileAccumulator {
    fn record(&mut self, profile: EmbedBatchProfile) {
        self.samples += 1;
        self.batch_size_sum += profile.batch_size;
        self.seq_len_sum += profile.sequence_length;
        self.prefix_ms_sum += profile.prefix_ms;
        self.tokenize_ms_sum += profile.tokenize_ms;
        self.tensorize_ms_sum += profile.tensorize_ms;
        self.run_ms_sum += profile.run_ms;
        self.extract_ms_sum += profile.extract_ms;
        self.postprocess_ms_sum += profile.postprocess_ms;
        self.total_ms_sum += profile.total_ms;
    }

    fn summary_line(&self, scenario_name: &str) -> Option<String> {
        if self.samples == 0 {
            return None;
        }

        let samples = self.samples as f64;
        Some(profile_summary_line(
            scenario_name,
            self.samples,
            EmbedBatchProfile {
                batch_size: self.batch_size_sum / self.samples,
                sequence_length: self.seq_len_sum / self.samples,
                prefix_ms: self.prefix_ms_sum / samples,
                tokenize_ms: self.tokenize_ms_sum / samples,
                tensorize_ms: self.tensorize_ms_sum / samples,
                run_ms: self.run_ms_sum / samples,
                extract_ms: self.extract_ms_sum / samples,
                postprocess_ms: self.postprocess_ms_sum / samples,
                total_ms: self.total_ms_sum / samples,
            },
        ))
    }
}

fn measure_warm_stats(
    engine: &EmbeddingEngine,
    scenario_name: &str,
    fixture: Option<&ResolvedFixture>,
    warmup: usize,
    iters: usize,
) -> Result<(LatencyStats, Option<String>), LTEmbedError> {
    let profiling_enabled = profiling_enabled_from_env();
    for _ in 0..warmup {
        if profiling_enabled {
            let _ = run_scenario_profiled(engine, scenario_name, fixture)?;
        } else {
            let _ = run_scenario(engine, scenario_name, fixture)?;
        }
    }

    let mut samples = Vec::with_capacity(iters);
    let mut accumulator = profiling_enabled.then(ProfileAccumulator::default);
    for _ in 0..iters {
        let start = Instant::now();
        if let Some(accumulator) = &mut accumulator {
            let (_, profile) = run_scenario_profiled(engine, scenario_name, fixture)?;
            accumulator.record(profile);
        } else {
            let _ = run_scenario(engine, scenario_name, fixture)?;
        }
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }

    let stats = LatencyStats::from_samples_ms(&samples)
        .map_err(|err| LTEmbedError::Inference(InferenceError::Internal(err)))?;
    let profile_line = accumulator.and_then(|accumulator| accumulator.summary_line(scenario_name));
    Ok((stats, profile_line))
}

fn measure_cold_stats(
    args: &Args,
    scenario_name: &str,
    fixture: Option<&ResolvedFixture>,
) -> Result<LatencyStats, LTEmbedError> {
    let start = Instant::now();
    let engine = engine_from_bundle_dir(args)?;
    let _ = run_scenario(&engine, scenario_name, fixture)?;
    LatencyStats::from_samples_ms(&[start.elapsed().as_secs_f64() * 1_000.0])
        .map_err(|err| LTEmbedError::Inference(InferenceError::Internal(err)))
}

fn progress_label(mode: &str, scenario_name: &str, state: &str) -> String {
    format!("{mode} {scenario_name} {state}")
}

fn emit_progress(mode: &str, scenario_name: &str, state: &str) {
    eprintln!("{}", progress_label(mode, scenario_name, state));
}

fn emit_profile_summary(line: &str) {
    eprintln!("{line}");
}

fn run_warm_mode(
    args: &Args,
    fixture: Option<&ResolvedFixture>,
    implementation_version: &str,
) -> Result<WarmPayload, Box<dyn std::error::Error>> {
    let engine = engine_from_bundle_dir(args)?;
    let mut results = Vec::new();
    let scenarios = resolve_scenarios(args)?;
    for scenario in scenarios {
        emit_progress("warm", scenario.name, "start");
        let (stats, profile_line) =
            measure_warm_stats(&engine, scenario.name, fixture, args.warmup, args.iters)?;
        emit_progress("warm", scenario.name, "done");
        if let Some(profile_line) = profile_line {
            emit_profile_summary(&profile_line);
        }
        results.push(StatsEntry {
            scenario: scenario.name.to_string(),
            stats,
        });
    }
    Ok(WarmPayload {
        implementation: "ltembed",
        implementation_version: implementation_version.to_string(),
        results,
    })
}

fn run_cold_mode(
    args: &Args,
    fixture: Option<&ResolvedFixture>,
    implementation_version: &str,
) -> Result<ColdPayload, Box<dyn std::error::Error>> {
    let scenario_name = required_scenario(args)?;
    emit_progress("cold", scenario_name, "start");
    let stats = measure_cold_stats(args, scenario_name, fixture)?;
    emit_progress("cold", scenario_name, "done");
    Ok(ColdPayload {
        implementation: "ltembed",
        implementation_version: implementation_version.to_string(),
        scenario: scenario_name.to_string(),
        stats,
    })
}

fn run_retrieval_mode(
    args: &Args,
    implementation_version: &str,
) -> Result<RetrievalPayload, Box<dyn std::error::Error>> {
    let engine = engine_from_bundle_dir(args)?;
    let retrieval_eval_path = required_retrieval_eval_path(args)?;
    let cases = load_retrieval_eval_cases(retrieval_eval_path)?;

    Ok(RetrievalPayload {
        implementation: "ltembed",
        implementation_version: implementation_version.to_string(),
        results: cases
            .into_iter()
            .map(|case| {
                let query_embeddings = embed_retrieval_inputs(
                    &engine,
                    case.queries
                        .iter()
                        .map(|query| EmbeddingInput::query(query.text.as_str()))
                        .collect(),
                )?;
                let document_embeddings = embed_retrieval_inputs(
                    &engine,
                    case.documents
                        .iter()
                        .map(|document| EmbeddingInput::document(document.text.as_str()))
                        .collect(),
                )?;
                Ok(RetrievalResult {
                    dataset_name: case.name,
                    queries: case
                        .queries
                        .into_iter()
                        .zip(query_embeddings)
                        .map(|(query, embedding)| RetrievalEmbedding {
                            id: query.id,
                            embedding,
                        })
                        .collect(),
                    documents: case
                        .documents
                        .into_iter()
                        .zip(document_embeddings)
                        .map(|(document, embedding)| RetrievalEmbedding {
                            id: document.id,
                            embedding,
                        })
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>, LTEmbedError>>()?,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
    let implementation_version = git_sha();
    let fixture = args
        .fixture_path
        .as_deref()
        .map(load_resolved_fixture)
        .transpose()?;
    let fixture = fixture.as_ref();

    match &args.mode {
        Mode::Warm => serde_json::to_writer(
            io::stdout(),
            &run_warm_mode(&args, fixture, &implementation_version)?,
        )?,
        Mode::Cold => serde_json::to_writer(
            io::stdout(),
            &run_cold_mode(&args, fixture, &implementation_version)?,
        )?,
        Mode::Retrieval => serde_json::to_writer(
            io::stdout(),
            &run_retrieval_mode(&args, &implementation_version)?,
        )?,
        Mode::Other(other) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
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
    use serde_json::json;

    #[test]
    fn test_parse_args_maps_warm_mode_to_typed_variant() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "warm",
            "--bundle-dir",
            "gguf_bundle",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
        ])
        .unwrap();

        assert_eq!(args.mode, Mode::Warm);
    }

    #[test]
    fn test_parse_args_maps_cold_mode_to_typed_variant() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "cold",
            "--scenario",
            "single/zh",
            "--bundle-dir",
            "gguf_bundle",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
        ])
        .unwrap();

        assert_eq!(args.mode, Mode::Cold);
    }

    #[test]
    fn test_parse_args_preserves_unknown_mode_for_main_validation() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "unknown",
            "--bundle-dir",
            "gguf_bundle",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
        ])
        .unwrap();

        assert_eq!(args.mode, Mode::Other("unknown".to_string()));
    }

    #[test]
    fn test_parse_args_accepts_optional_scenario_for_warm_mode() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "warm",
            "--scenario",
            "single/zh",
            "--bundle-dir",
            "gguf_bundle",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
        ])
        .unwrap();

        assert_eq!(args.mode, Mode::Warm);
        assert_eq!(args.scenario.as_deref(), Some("single/zh"));
        assert_eq!(args.bundle_dir, PathBuf::from("gguf_bundle"));
        assert_eq!(args.output_dimension, 512);
        assert!(args.l2_normalize);
    }

    #[test]
    fn test_required_scenario_rejects_missing_cold_scenario() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "cold",
            "--bundle-dir",
            "gguf_bundle",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
        ])
        .unwrap();

        let err = required_scenario(&args).unwrap_err();

        assert_eq!(err.to_string(), "missing --scenario");
    }

    #[test]
    fn test_run_cold_mode_preserves_invalid_input_error_message() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "cold",
            "--bundle-dir",
            "gguf_bundle",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
        ])
        .unwrap();

        match run_cold_mode(&args, None, "test-sha") {
            Ok(_) => panic!("expected missing --scenario error"),
            Err(err) => {
                assert_eq!(err.to_string(), "missing --scenario");
            }
        }
    }

    #[test]
    fn test_resolve_scenarios_rejects_unknown_scenario_without_io_prefix() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "warm",
            "--scenario",
            "missing/scenario",
            "--bundle-dir",
            "gguf_bundle",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
        ])
        .unwrap();

        let err = resolve_scenarios(&args).unwrap_err();

        assert_eq!(err.to_string(), "unknown scenario: missing/scenario");
    }

    #[test]
    fn test_progress_label_includes_mode_scenario_and_state() {
        assert_eq!(
            progress_label("warm", "single/zh", "start"),
            "warm single/zh start"
        );
    }

    #[test]
    fn test_profiling_enabled_from_value_defaults_to_false() {
        assert!(!profiling_enabled_from_value(None));
        assert!(!profiling_enabled_from_value(Some("")));
        assert!(!profiling_enabled_from_value(Some("0")));
        assert!(!profiling_enabled_from_value(Some("false")));
    }

    #[test]
    fn test_profiling_enabled_from_value_accepts_enabled_values() {
        assert!(profiling_enabled_from_value(Some("1")));
        assert!(profiling_enabled_from_value(Some("true")));
        assert!(profiling_enabled_from_value(Some("TRUE")));
        assert!(profiling_enabled_from_value(Some("yes")));
        assert!(profiling_enabled_from_value(Some("YES")));
    }

    #[test]
    fn test_profile_summary_line_renders_parseable_key_values() {
        let line = profile_summary_line(
            "single/zh",
            3,
            EmbedBatchProfile {
                batch_size: 1,
                sequence_length: 7,
                prefix_ms: 1.25,
                tokenize_ms: 2.5,
                tensorize_ms: 3.75,
                run_ms: 5.0,
                extract_ms: 6.25,
                postprocess_ms: 7.5,
                total_ms: 26.25,
            },
        );

        assert_eq!(
            line,
            "profile single/zh samples=3 batch_size=1 seq_len=7 prefix_ms=1.250 tokenize_ms=2.500 tensorize_ms=3.750 run_ms=5.000 extract_ms=6.250 postprocess_ms=7.500 total_ms=26.250"
        );
    }

    #[test]
    fn test_warm_payload_serializes_current_json_shape() {
        let payload = WarmPayload {
            implementation: "ltembed",
            implementation_version: "test-sha".to_string(),
            results: vec![StatsEntry {
                scenario: "single/zh".to_string(),
                stats: LatencyStats::from_samples_ms(&[12.0, 18.0]).unwrap(),
            }],
        };

        let value = serde_json::to_value(payload).unwrap();

        assert_eq!(value["implementation"], json!("ltembed"));
        assert_eq!(value["implementation_version"], json!("test-sha"));
        assert_eq!(value["results"][0]["scenario"], json!("single/zh"));
        assert!(value["results"][0]["stats"]["mean_ms"].is_number());
        assert!(value["results"][0]["stats"]["median_ms"].is_number());
        assert!(value["results"][0]["stats"]["p95_ms"].is_number());
        assert!(value["results"][0]["stats"]["p99_ms"].is_number());
        assert!(value["results"][0]["stats"]["min_ms"].is_number());
        assert!(value["results"][0]["stats"]["max_ms"].is_number());
    }

    #[test]
    fn test_cold_payload_serializes_current_json_shape() {
        let payload = ColdPayload {
            implementation: "ltembed",
            implementation_version: "test-sha".to_string(),
            scenario: "single/zh".to_string(),
            stats: LatencyStats::from_samples_ms(&[25.0]).unwrap(),
        };

        let value = serde_json::to_value(payload).unwrap();

        assert_eq!(value["implementation"], json!("ltembed"));
        assert_eq!(value["implementation_version"], json!("test-sha"));
        assert_eq!(value["scenario"], json!("single/zh"));
        assert!(value["stats"]["mean_ms"].is_number());
        assert!(value["stats"]["median_ms"].is_number());
        assert!(value["stats"]["p95_ms"].is_number());
        assert!(value["stats"]["p99_ms"].is_number());
        assert!(value["stats"]["min_ms"].is_number());
        assert!(value["stats"]["max_ms"].is_number());
    }

    #[test]
    fn test_parse_args_maps_retrieval_mode_to_typed_variant() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "retrieval",
            "--bundle-dir",
            "gguf_bundle",
            "--retrieval-eval-path",
            "scripts/retrieval_eval_cases.json",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
        ])
        .unwrap();

        assert_eq!(args.mode, Mode::Retrieval);
    }

    #[test]
    fn test_parse_args_accepts_retrieval_eval_path_for_retrieval_mode() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "retrieval",
            "--bundle-dir",
            "gguf_bundle",
            "--retrieval-eval-path",
            "scripts/retrieval_eval_cases.json",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
        ])
        .unwrap();

        assert_eq!(args.mode, Mode::Retrieval);
        assert_eq!(
            args.retrieval_eval_path.as_deref(),
            Some(Path::new("scripts/retrieval_eval_cases.json"))
        );
    }

    #[test]
    fn test_load_retrieval_eval_cases_reads_queries_and_documents() {
        let path = std::env::temp_dir().join("ltembed-retrieval-eval-cases.json");
        fs::write(
            &path,
            r#"{
                "cases": [
                    {
                        "name": "mini-retrieval-v1",
                        "documents": [{"id": "d1", "text": "Rust ownership protects memory safety."}],
                        "queries": [{"id": "q1", "text": "How does Rust avoid a garbage collector?", "relevant_document_ids": ["d1"]}]
                    },
                    {
                        "name": "mini-retrieval-hard-v1",
                        "documents": [{"id": "d2", "text": "ANN indexes power vector retrieval."}],
                        "queries": [{"id": "q2", "text": "What supports nearest-neighbor search over embeddings?", "relevant_document_ids": ["d2"]}]
                    }
                ]
            }"#,
        )
        .unwrap();

        let cases = load_retrieval_eval_cases(&path).unwrap();

        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].name, "mini-retrieval-v1");
        assert_eq!(cases[0].documents.len(), 1);
        assert_eq!(cases[0].documents[0].id, "d1");
        assert_eq!(cases[1].name, "mini-retrieval-hard-v1");
        assert_eq!(cases[1].queries[0].relevant_document_ids, vec!["d2"]);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_required_retrieval_eval_path_rejects_missing_path() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "retrieval",
            "--bundle-dir",
            "gguf_bundle",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
        ])
        .unwrap();

        let err = required_retrieval_eval_path(&args).unwrap_err();

        assert_eq!(err.to_string(), "missing --retrieval-eval-path");
    }

    #[test]
    fn test_retrieval_payload_serializes_current_json_shape() {
        let payload = RetrievalPayload {
            implementation: "ltembed",
            implementation_version: "test-sha".to_string(),
            results: vec![RetrievalResult {
                dataset_name: "mini-retrieval-v1".to_string(),
                queries: vec![RetrievalEmbedding {
                    id: "q1".to_string(),
                    embedding: vec![0.1, 0.2],
                }],
                documents: vec![RetrievalEmbedding {
                    id: "d1".to_string(),
                    embedding: vec![0.3, 0.4],
                }],
            }],
        };

        let value = serde_json::to_value(payload).unwrap();

        assert_eq!(
            value,
            json!({
                "implementation": "ltembed",
                "implementation_version": "test-sha",
                "results": [
                    {
                        "dataset_name": "mini-retrieval-v1",
                        "queries": [{"id": "q1", "embedding": [0.1_f32, 0.2_f32]}],
                        "documents": [{"id": "d1", "embedding": [0.3_f32, 0.4_f32]}]
                    }
                ]
            })
        );
    }

    #[test]
    fn test_parse_args_accepts_threads() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "warm",
            "--bundle-dir",
            "gguf_bundle",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
            "--threads",
            "4",
        ])
        .unwrap();
        assert_eq!(args.threads, 4);
    }

    #[test]
    fn test_parse_args_accepts_fixture_path() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "warm",
            "--bundle-dir",
            "gguf_bundle",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
            "--fixture-path",
            "artifacts/resolved_fixture.json",
        ])
        .unwrap();
        assert_eq!(
            args.fixture_path.as_deref(),
            Some(Path::new("artifacts/resolved_fixture.json"))
        );
    }

    #[test]
    fn test_scenario_inputs_resolved_uses_fixture_when_present() {
        let fixture: ResolvedFixture = serde_json::from_str(
            r#"{ "scenarios": {
                "single/zh": [{"kind": "query", "text": "他感冒了"}],
                "single/en": [{"kind": "document", "text": "He caught a cold."}]
            } }"#,
        )
        .unwrap();
        let zh = scenario_inputs_resolved("single/zh", Some(&fixture)).unwrap();
        assert_eq!(zh.len(), 1);
        assert_eq!(zh[0].kind, EmbeddingInputKind::Query);
        let en = scenario_inputs_resolved("single/en", Some(&fixture)).unwrap();
        assert_eq!(en.len(), 1);
        assert_eq!(en[0].kind, EmbeddingInputKind::Document);
    }

    #[test]
    fn test_scenario_inputs_resolved_falls_back_to_builtin_without_fixture() {
        let inputs = scenario_inputs_resolved("single/zh", None).unwrap();
        assert_eq!(
            inputs,
            scenario_inputs(scenario_by_name("single/zh").unwrap())
        );
    }

    #[test]
    fn test_scenario_inputs_resolved_errors_on_missing_scenario_in_fixture() {
        let fixture: ResolvedFixture =
            serde_json::from_str(r#"{"scenarios": {"single/zh": []}}"#).unwrap();
        let err = scenario_inputs_resolved("single/en", Some(&fixture)).unwrap_err();
        assert!(err
            .to_string()
            .contains("fixture is missing scenario: single/en"));
    }

    #[test]
    fn test_scenario_inputs_resolved_errors_on_unknown_kind() {
        let fixture: ResolvedFixture = serde_json::from_str(
            r#"{"scenarios": {"single/zh": [{"kind": "passage", "text": "x"}]}}"#,
        )
        .unwrap();
        let err = scenario_inputs_resolved("single/zh", Some(&fixture)).unwrap_err();
        assert!(err.to_string().contains("unknown kind: passage"));
    }

    #[test]
    fn test_parse_args_rejects_zero_threads() {
        let err = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "warm",
            "--bundle-dir",
            "gguf_bundle",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
            "--threads",
            "0",
        ])
        .unwrap_err();
        assert!(err.contains("--threads"));
    }
}
