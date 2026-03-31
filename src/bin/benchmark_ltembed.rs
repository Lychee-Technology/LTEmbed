use ltembed::benchmarking::{scenario_by_name, scenario_inputs, selected_scenarios, LatencyStats};
use ltembed::engine::{EmbeddingInput, EmbeddingInputKind, OnnxEngine, OnnxEngineConfig};
use ltembed::error::LTEmbedError;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug)]
struct Args {
    mode: String,
    scenario: Option<String>,
    ort_bundle_dir: PathBuf,
    retrieval_eval_path: Option<PathBuf>,
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

#[derive(Debug, Deserialize)]
struct RetrievalEvalCases {
    name: String,
    documents: Vec<RetrievalDocument>,
    queries: Vec<RetrievalQuery>,
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
struct RetrievalPayload {
    implementation: &'static str,
    implementation_version: String,
    dataset_name: String,
    queries: Vec<RetrievalEmbedding>,
    documents: Vec<RetrievalEmbedding>,
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
        mode: mode.ok_or_else(|| "missing required --mode".to_string())?,
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

fn engine_from_bundle_dir(args: &Args) -> Result<OnnxEngine, LTEmbedError> {
    OnnxEngine::from_bundle_dir(
        Path::new(&args.ort_bundle_dir),
        OnnxEngineConfig {
            output_dimension: args.output_dimension,
            l2_normalize: args.l2_normalize,
        },
    )
}

fn load_retrieval_eval_cases(path: &Path) -> io::Result<RetrievalEvalCases> {
    let contents = fs::read_to_string(path)?;
    let cases: RetrievalEvalCases = serde_json::from_str(&contents)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    let document_ids = cases
        .documents
        .iter()
        .map(|document| document.id.as_str())
        .collect::<std::collections::HashSet<_>>();
    for query in &cases.queries {
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
                    results,
                },
            )?;
        }
        "cold" => {
            let scenario_name = args.scenario.as_deref().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing --scenario")
            })?;
            let stats = measure_cold_stats(&args, scenario_name)?;
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
                    results,
                },
            )?;
        }
        "retrieval" => {
            let engine = engine_from_bundle_dir(&args)?;
            let retrieval_eval_path = args.retrieval_eval_path.as_deref().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "missing --retrieval-eval-path")
            })?;
            let cases = load_retrieval_eval_cases(retrieval_eval_path)?;
            let query_embeddings = embed_retrieval_inputs(
                &engine,
                cases
                    .queries
                    .iter()
                    .map(|query| EmbeddingInput::query(query.text.as_str()))
                    .collect(),
            )?;
            let document_embeddings = embed_retrieval_inputs(
                &engine,
                cases
                    .documents
                    .iter()
                    .map(|document| EmbeddingInput::document(document.text.as_str()))
                    .collect(),
            )?;
            serde_json::to_writer(
                std::io::stdout(),
                &RetrievalPayload {
                    implementation: "ltembed",
                    implementation_version,
                    dataset_name: cases.name,
                    queries: cases
                        .queries
                        .into_iter()
                        .zip(query_embeddings)
                        .map(|(query, embedding)| RetrievalEmbedding {
                            id: query.id,
                            embedding,
                        })
                        .collect(),
                    documents: cases
                        .documents
                        .into_iter()
                        .zip(document_embeddings)
                        .map(|(document, embedding)| RetrievalEmbedding {
                            id: document.id,
                            embedding,
                        })
                        .collect(),
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
    use std::fs;

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

        assert_eq!(args.mode, "retrieval");
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
                "name": "mini-retrieval-v1",
                "documents": [{"id": "d1", "text": "Rust ownership protects memory safety."}],
                "queries": [{"id": "q1", "text": "How does Rust avoid a garbage collector?", "relevant_document_ids": ["d1"]}]
            }"#,
        )
        .unwrap();

        let cases = load_retrieval_eval_cases(&path).unwrap();

        assert_eq!(cases.name, "mini-retrieval-v1");
        assert_eq!(cases.documents.len(), 1);
        assert_eq!(cases.documents[0].id, "d1");
        assert_eq!(cases.queries.len(), 1);
        assert_eq!(cases.queries[0].relevant_document_ids, vec!["d1"]);

        let _ = fs::remove_file(path);
    }
}
