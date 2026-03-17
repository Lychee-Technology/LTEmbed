use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use ltembed::benchmarking::{benchmark_scenarios, scenario_by_name, scenario_texts, LatencyStats};
use ltembed::error::LTEmbedError;
use ltembed::traits::pooling::{MeanPooling, Pooling};
use ltembed::traits::tokenizer::{HFTokenizer, Tokenizer};
use ltembed::utils::l2_normalize_inplace;
use serde::Serialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MAX_LENGTH: usize = 512;

#[derive(Debug)]
struct Args {
    mode: String,
    scenario: Option<String>,
    model_dir: PathBuf,
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

struct CandleEmbedder {
    model: BertModel,
    tokenizer: HFTokenizer,
    pooling: MeanPooling,
    device: Device,
    hidden_size: usize,
    pad_token_id: u32,
}

fn parse_args() -> Result<Args, String> {
    let mut mode = None;
    let mut scenario = None;
    let mut model_dir = None;
    let mut warmup = 10usize;
    let mut iters = 100usize;
    let mut threads = 1usize;

    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--mode" => mode = iter.next(),
            "--scenario" => scenario = iter.next(),
            "--model-dir" => model_dir = iter.next().map(PathBuf::from),
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
        warmup,
        iters,
        threads,
    })
}

fn implementation_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

fn candle_error(err: impl ToString) -> LTEmbedError {
    LTEmbedError::Inference(err.to_string())
}

impl CandleEmbedder {
    fn from_model_dir(model_dir: &Path) -> Result<Self, LTEmbedError> {
        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let safetensors_path = model_dir.join("model.safetensors");
        let config: BertConfig = serde_json::from_str(&fs::read_to_string(config_path)?)?;
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[safetensors_path], DType::F32, &device)
                .map_err(candle_error)?
        };
        let model = BertModel::load(vb, &config).map_err(candle_error)?;
        let tokenizer = HFTokenizer::from_file(&tokenizer_path.to_string_lossy())?;
        Ok(Self {
            model,
            tokenizer,
            pooling: MeanPooling,
            device,
            hidden_size: config.hidden_size,
            pad_token_id: config.pad_token_id as u32,
        })
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LTEmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encoded = texts
            .iter()
            .map(|text| self.tokenizer.encode(text, MAX_LENGTH))
            .collect::<Result<Vec<_>, _>>()?;
        let batch_size = encoded.len();
        let seq_len = encoded
            .iter()
            .map(|item| item.input_ids.len())
            .max()
            .unwrap_or(0);
        let total_tokens = batch_size * seq_len;

        let mut input_ids = vec![self.pad_token_id; total_tokens];
        let mut token_type_ids = vec![0u32; total_tokens];
        let mut attention_mask = vec![0u32; total_tokens];

        for (batch_idx, item) in encoded.iter().enumerate() {
            let row_start = batch_idx * seq_len;
            let row_end = row_start + item.input_ids.len();
            input_ids[row_start..row_end].copy_from_slice(&item.input_ids);
            token_type_ids[row_start..row_end].copy_from_slice(&item.token_type_ids);
            attention_mask[row_start..row_end].copy_from_slice(&item.attention_mask);
        }

        let input_ids = Tensor::new(input_ids.as_slice(), &self.device)
            .map_err(candle_error)?
            .reshape((batch_size, seq_len))
            .map_err(candle_error)?;
        let token_type_ids = Tensor::new(token_type_ids.as_slice(), &self.device)
            .map_err(candle_error)?
            .reshape((batch_size, seq_len))
            .map_err(candle_error)?;
        let attention_mask = Tensor::new(attention_mask.as_slice(), &self.device)
            .map_err(candle_error)?
            .reshape((batch_size, seq_len))
            .map_err(candle_error)?;

        let hidden_state = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(candle_error)?;
        let flat_hidden_state = hidden_state
            .flatten_all()
            .map_err(candle_error)?
            .to_vec1::<f32>()
            .map_err(candle_error)?;
        let attention_mask_vec = attention_mask
            .flatten_all()
            .map_err(candle_error)?
            .to_vec1::<u32>()
            .map_err(candle_error)?;

        let mut embeddings = Vec::with_capacity(batch_size);
        for batch_idx in 0..batch_size {
            let state_start = batch_idx * seq_len * self.hidden_size;
            let state_end = state_start + seq_len * self.hidden_size;
            let mask_start = batch_idx * seq_len;
            let mask_end = mask_start + seq_len;
            let mut pooled = self.pooling.pool(
                &flat_hidden_state[state_start..state_end],
                seq_len,
                self.hidden_size,
                &attention_mask_vec[mask_start..mask_end],
            )?;
            l2_normalize_inplace(&mut pooled);
            embeddings.push(pooled);
        }
        Ok(embeddings)
    }
}

fn run_scenario(
    embedder: &CandleEmbedder,
    scenario_name: &str,
) -> Result<Vec<Vec<f32>>, LTEmbedError> {
    let scenario = scenario_by_name(scenario_name)
        .ok_or_else(|| LTEmbedError::Inference("unknown scenario".into()))?;
    embedder.embed_batch(&scenario_texts(scenario))
}

fn measure_warm_stats(
    embedder: &CandleEmbedder,
    scenario_name: &str,
    warmup: usize,
    iters: usize,
) -> Result<LatencyStats, LTEmbedError> {
    for _ in 0..warmup {
        let _ = run_scenario(embedder, scenario_name)?;
    }

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let _ = run_scenario(embedder, scenario_name)?;
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }

    LatencyStats::from_samples_ms(&samples).map_err(LTEmbedError::Inference)
}

fn measure_cold_stats(model_dir: &Path, scenario_name: &str) -> Result<LatencyStats, LTEmbedError> {
    let start = Instant::now();
    let embedder = CandleEmbedder::from_model_dir(model_dir)?;
    let _ = run_scenario(&embedder, scenario_name)?;
    LatencyStats::from_samples_ms(&[start.elapsed().as_secs_f64() * 1_000.0])
        .map_err(LTEmbedError::Inference)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args =
        parse_args().map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    env::set_var("RAYON_NUM_THREADS", args.threads.to_string());
    let implementation_version = implementation_version();

    match args.mode.as_str() {
        "warm" => {
            let embedder = CandleEmbedder::from_model_dir(&args.model_dir)?;
            let mut results = Vec::new();
            for scenario in benchmark_scenarios() {
                let stats = measure_warm_stats(&embedder, scenario.name, args.warmup, args.iters)?;
                results.push(StatsEntry {
                    scenario: scenario.name.to_string(),
                    stats,
                });
            }
            serde_json::to_writer(
                std::io::stdout(),
                &WarmPayload {
                    implementation: "candle",
                    implementation_version,
                    results,
                },
            )?;
        }
        "cold" => {
            let scenario_name = args.scenario.as_deref().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing --scenario")
            })?;
            let stats = measure_cold_stats(&args.model_dir, scenario_name)?;
            serde_json::to_writer(
                std::io::stdout(),
                &ColdPayload {
                    implementation: "candle",
                    implementation_version,
                    scenario: scenario_name.to_string(),
                    stats,
                },
            )?;
        }
        "correctness" => {
            let embedder = CandleEmbedder::from_model_dir(&args.model_dir)?;
            let mut results = Vec::new();
            for scenario in benchmark_scenarios() {
                let embeddings = run_scenario(&embedder, scenario.name)?;
                results.push(EmbeddingsEntry {
                    scenario: scenario.name.to_string(),
                    embeddings,
                });
            }
            serde_json::to_writer(
                std::io::stdout(),
                &CorrectnessPayload {
                    implementation: "candle",
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
