use ltembed::benchmarking::{
    scenario_by_name, scenario_inputs, selected_scenarios, BenchmarkScenario, LatencyStats,
};
use ltembed::engine::{EmbeddingInput, EmbeddingInputKind, OnnxEngine, OnnxEngineConfig};
use ltembed::error::LTEmbedError;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;
use thiserror::Error;

#[derive(Debug)]
struct Args {
    mode: Mode,
    scenario: Option<String>,
    ort_bundle_dir: PathBuf,
    retrieval_eval_path: Option<PathBuf>,
    output_dimension: usize,
    l2_normalize: bool,
    warmup: usize,
    iters: usize,
    threads: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Mode {
    Warm,
    Cold,
    Correctness,
    Retrieval,
    Other(String),
}

#[derive(Debug, Error)]
enum ModeRunError {
    #[error(transparent)]
    InvalidInput(#[from] io::Error),
    #[error(transparent)]
    LTEmbed(#[from] LTEmbedError),
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

#[derive(Debug, Deserialize)]
struct RetrievalEvalCase {
    name: String,
    documents: Vec<RetrievalDocument>,
    queries: Vec<RetrievalQuery>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RetrievalEvalSpec {
    Suite { cases: Vec<RetrievalEvalCase> },
    Single(RetrievalEvalCase),
}

#[derive(Debug, Deserialize)]
struct RetrievalDocument {
    id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct RetrievalQuery {
    id: String,
    text: String,
    relevant_document_ids: Vec<String>,
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

fn parse_args_from<I, S>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut mode = None;
    let mut scenario = None;
    let mut ort_bundle_dir = None;
    let mut retrieval_eval_path = None;
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
            "--retrieval-eval-path" => retrieval_eval_path = iter.next().map(PathBuf::from),
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
        mode: mode
            .map(Mode::from)
            .ok_or_else(|| "missing required --mode".to_string())?,
        scenario,
        ort_bundle_dir: ort_bundle_dir
            .ok_or_else(|| "missing required --ort-bundle-dir".to_string())?,
        retrieval_eval_path,
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

impl From<String> for Mode {
    fn from(value: String) -> Self {
        match value.as_str() {
            "warm" => Self::Warm,
            "cold" => Self::Cold,
            "correctness" => Self::Correctness,
            "retrieval" => Self::Retrieval,
            _ => Self::Other(value),
        }
    }
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

fn load_retrieval_eval_cases(path: &Path) -> io::Result<Vec<RetrievalEvalCase>> {
    let contents = fs::read_to_string(path)?;
    let spec: RetrievalEvalSpec = serde_json::from_str(&contents)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    let cases = match spec {
        RetrievalEvalSpec::Suite { cases } => cases,
        RetrievalEvalSpec::Single(case) => vec![case],
    };
    for case in &cases {
        let document_ids = case
            .documents
            .iter()
            .map(|document| document.id.as_str())
            .collect::<std::collections::HashSet<_>>();
        for query in &case.queries {
            if query.relevant_document_ids.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "query {} must declare at least one relevant document",
                        query.id
                    ),
                ));
            }
            for document_id in &query.relevant_document_ids {
                if !document_ids.contains(document_id.as_str()) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "query {} references unknown document {}",
                            query.id, document_id
                        ),
                    ));
                }
            }
        }
    }
    Ok(cases)
}

fn embed_retrieval_inputs(
    engine: &OnnxEngine,
    inputs: Vec<EmbeddingInput<'_>>,
) -> Result<Vec<Vec<f32>>, LTEmbedError> {
    engine.embed_batch(&inputs)
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

fn resolve_scenarios(args: &Args) -> io::Result<Vec<&'static BenchmarkScenario>> {
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

fn run_warm_mode(args: &Args, implementation_version: &str) -> Result<WarmPayload, ModeRunError> {
    let engine = engine_from_bundle_dir(args)?;
    let mut results = Vec::new();
    let scenarios = resolve_scenarios(args)?;
    for scenario in scenarios {
        let stats = measure_warm_stats(&engine, scenario.name, args.warmup, args.iters)?;
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

fn run_cold_mode(args: &Args, implementation_version: &str) -> Result<ColdPayload, ModeRunError> {
    let scenario_name = required_scenario(args)?;
    let stats = measure_cold_stats(args, scenario_name)?;

    Ok(ColdPayload {
        implementation: "ltembed",
        implementation_version: implementation_version.to_string(),
        scenario: scenario_name.to_string(),
        stats,
    })
}

fn run_correctness_mode(
    args: &Args,
    implementation_version: &str,
) -> Result<CorrectnessPayload, ModeRunError> {
    let engine = engine_from_bundle_dir(args)?;
    let mut results = Vec::new();
    let scenarios = resolve_scenarios(args)?;
    for scenario in scenarios {
        let embeddings = run_scenario(&engine, scenario.name)?;
        results.push(EmbeddingsEntry {
            scenario: scenario.name.to_string(),
            embeddings,
        });
    }

    Ok(CorrectnessPayload {
        implementation: "ltembed",
        implementation_version: implementation_version.to_string(),
        results,
    })
}

fn run_retrieval_mode(
    args: &Args,
    implementation_version: &str,
) -> Result<RetrievalPayload, ModeRunError> {
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
    let args =
        parse_args().map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    let _threads = args.threads;
    let implementation_version = git_sha();

    match &args.mode {
        Mode::Warm => serde_json::to_writer(
            std::io::stdout(),
            &run_warm_mode(&args, &implementation_version)?,
        )?,
        Mode::Cold => serde_json::to_writer(
            std::io::stdout(),
            &run_cold_mode(&args, &implementation_version)?,
        )?,
        Mode::Correctness => serde_json::to_writer(
            std::io::stdout(),
            &run_correctness_mode(&args, &implementation_version)?,
        )?,
        Mode::Retrieval => serde_json::to_writer(
            std::io::stdout(),
            &run_retrieval_mode(&args, &implementation_version)?,
        )?,
        Mode::Other(other) => {
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
    use serde_json::json;
    use std::fs;

    #[test]
    fn test_parse_args_maps_warm_mode_to_typed_variant() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "warm",
            "--ort-bundle-dir",
            "ort_bundle",
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
            "single/medium",
            "--ort-bundle-dir",
            "ort_bundle",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
        ])
        .unwrap();

        assert_eq!(args.mode, Mode::Cold);
    }

    #[test]
    fn test_parse_args_maps_correctness_mode_to_typed_variant() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "correctness",
            "--ort-bundle-dir",
            "ort_bundle",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
        ])
        .unwrap();

        assert_eq!(args.mode, Mode::Correctness);
    }

    #[test]
    fn test_parse_args_maps_retrieval_mode_to_typed_variant() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "retrieval",
            "--ort-bundle-dir",
            "ort_bundle",
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
    fn test_parse_args_preserves_unknown_mode_for_main_validation() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "unknown",
            "--ort-bundle-dir",
            "ort_bundle",
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
            "single/medium",
            "--ort-bundle-dir",
            "ort_bundle",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
        ])
        .unwrap();

        assert_eq!(args.mode, Mode::Warm);
        assert_eq!(args.scenario.as_deref(), Some("single/medium"));
        assert_eq!(args.ort_bundle_dir, PathBuf::from("ort_bundle"));
        assert_eq!(args.output_dimension, 512);
        assert!(args.l2_normalize);
    }

    #[test]
    fn test_parse_args_accepts_retrieval_eval_path_for_retrieval_mode() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "retrieval",
            "--ort-bundle-dir",
            "ort_bundle",
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
    fn test_required_scenario_rejects_missing_cold_scenario() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "cold",
            "--ort-bundle-dir",
            "ort_bundle",
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
    fn test_required_retrieval_eval_path_rejects_missing_path() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "retrieval",
            "--ort-bundle-dir",
            "ort_bundle",
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
    fn test_run_cold_mode_preserves_invalid_input_error_message() {
        let args = parse_args_from([
            "benchmark_ltembed",
            "--mode",
            "cold",
            "--ort-bundle-dir",
            "ort_bundle",
            "--output-dimension",
            "512",
            "--l2-normalize",
            "true",
        ])
        .unwrap();

        match run_cold_mode(&args, "test-sha") {
            Ok(_) => panic!("expected missing --scenario error"),
            Err(err) => {
                assert_eq!(err.to_string(), "missing --scenario");
                assert!(matches!(err, ModeRunError::InvalidInput(_)));
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
            "--ort-bundle-dir",
            "ort_bundle",
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
    fn test_warm_payload_serializes_current_json_shape() {
        let payload = WarmPayload {
            implementation: "ltembed",
            implementation_version: "test-sha".to_string(),
            results: vec![StatsEntry {
                scenario: "single/medium".to_string(),
                stats: LatencyStats::from_samples_ms(&[12.0, 18.0]).unwrap(),
            }],
        };

        let value = serde_json::to_value(payload).unwrap();

        assert_eq!(value["implementation"], json!("ltembed"));
        assert_eq!(value["implementation_version"], json!("test-sha"));
        assert_eq!(value["results"][0]["scenario"], json!("single/medium"));
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
            scenario: "single/medium".to_string(),
            stats: LatencyStats::from_samples_ms(&[25.0]).unwrap(),
        };

        let value = serde_json::to_value(payload).unwrap();

        assert_eq!(value["implementation"], json!("ltembed"));
        assert_eq!(value["implementation_version"], json!("test-sha"));
        assert_eq!(value["scenario"], json!("single/medium"));
        assert!(value["stats"]["mean_ms"].is_number());
        assert!(value["stats"]["median_ms"].is_number());
        assert!(value["stats"]["p95_ms"].is_number());
        assert!(value["stats"]["p99_ms"].is_number());
        assert!(value["stats"]["min_ms"].is_number());
        assert!(value["stats"]["max_ms"].is_number());
    }

    #[test]
    fn test_correctness_payload_serializes_current_json_shape() {
        let payload = CorrectnessPayload {
            implementation: "ltembed",
            implementation_version: "test-sha".to_string(),
            results: vec![EmbeddingsEntry {
                scenario: "single/medium".to_string(),
                embeddings: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
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
                        "scenario": "single/medium",
                        "embeddings": [[0.1_f32, 0.2_f32], [0.3_f32, 0.4_f32]]
                    }
                ]
            })
        );
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
}
