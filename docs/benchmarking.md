# Benchmarking

How LTEmbed is measured, how to run the measurements, and the results recorded so far.
Benchmarks are not part of CI; they run by hand or through the manually dispatched
`benchmark-arm64` workflow.

## Methodology

### Scenarios

Defined in `src/benchmarking.rs` (the Python side mirrors them in
`scripts/run_embedding_benchmarks.py`):

| Scenario | Batch | Input |
|---|---|---|
| `single/zh` | 1 | Query `他感冒了` |
| `single/en` | 1 | Query `He caught a cold.` |
| `single/medium` | 1 | `medium` entry of `scripts/benchmark_corpus.json` |
| `single/long` | 1 | `long` entry of `scripts/benchmark_corpus.json` |
| `batch/medium/8` | 8 | `batch_medium_8` entries of `scripts/benchmark_corpus.json` |

`--fixture-path` replaces these texts with a pre-resolved fixture, such as the CN/EN fixture
that `scripts/build_cn_en_retrieval_cases.py --fixture-output` writes.

Batch scenarios measure `embed_batch`, which encodes inputs one at a time (see
[architecture.md](./architecture.md#inference-path)). They do not measure multi-sequence
batching.

### Modes

| Mode | What one run measures |
|---|---|
| `warm` | Load the engine once, run `--warmup` untimed iterations, then time `--iters` iterations of each scenario |
| `cold` | Load the engine and embed one scenario once, in a fresh process. One sample per process; the orchestrator runs `--cold-iters` processes and aggregates them. |
| `retrieval` | Embed the queries and documents of each case in a retrieval-eval JSON and print the vectors. Quality metrics are computed by the orchestrator. |

### Metrics

- **Latency** (`warm`, `cold`): mean, median, p95, p99, min and max in milliseconds.
- **Stage profile**: with `LTEMBED_PROFILE=1` (also `true` or `yes`), warm mode prints one
  line per scenario to stderr with mean `prefix_ms`, `tokenize_ms`, `tensorize_ms`,
  `run_ms`, `extract_ms`, `postprocess_ms` and `total_ms`.
- **Correctness vs. PyTorch**: cosine between LTEmbed and the PyTorch FP32 model on the
  retrieval-eval texts. The default pass threshold is `0.98`
  (`--correctness-threshold`).
- **Golden parity**: cosine against the immutable fixture `tests/fixtures/test_fixtures.json`
  (see [testing.md](./testing.md#golden-fixture-testsfixturestest_fixturesjson)). Threshold
  `0.98` by default (`--golden-parity-threshold`). It is skipped when `--output-dimension`
  differs from the fixture's `dim` (512).
- **Retrieval quality**: `recall_at_1`, `recall_at_3`, `both_at_3` and `mrr_at_3` per
  dataset, for LTEmbed and for PyTorch.

## Running `benchmark_ltembed` directly

The binary needs the same build environment as the library (aarch64 or x86-64-v3 Linux
and `STATIC_LLAMA_DIR`; see [development.md](./development.md)) and a bundle (see
[bundle-format.md](./bundle-format.md)).

```bash
cargo build --release --bin benchmark_ltembed

# Warm latency, every scenario, 1 thread, with the stage profile
LTEMBED_PROFILE=1 ./target/release/benchmark_ltembed --mode warm \
  --bundle-dir gguf_bundle --output-dimension 512 --l2-normalize true \
  --warmup 10 --iters 100 --threads 1

# One cold start
./target/release/benchmark_ltembed --mode cold --scenario single/en \
  --bundle-dir gguf_bundle --output-dimension 512 --l2-normalize true

# Retrieval embeddings for the small built-in eval set
./target/release/benchmark_ltembed --mode retrieval \
  --bundle-dir gguf_bundle --retrieval-eval-path scripts/retrieval_eval_cases.json \
  --output-dimension 512 --l2-normalize true
```

| Flag | Required | Default | Notes |
|---|---|---|---|
| `--mode warm\|cold\|retrieval` | yes | — | |
| `--bundle-dir <dir>` | yes | — | GGUF bundle directory |
| `--output-dimension <n>` | yes | — | Must be ≤ the bundle's `raw_embedding_dimension` |
| `--l2-normalize true\|false` | yes | — | |
| `--scenario <name>` | `cold`: yes | all scenarios | Restricts `warm` to one scenario |
| `--retrieval-eval-path <json>` | `retrieval`: yes | — | |
| `--fixture-path <json>` | no | built-in texts | Overrides scenario texts |
| `--warmup <n>` | no | `10` | |
| `--iters <n>` | no | `100` | |
| `--threads <n>` | no | `1` | llama.cpp `n_threads`; must be > 0 |

Results are printed as JSON on stdout (`implementation: "ltembed"`, then a `results` list
for `warm` and `retrieval`, or a single `stats` object for `cold`). Progress lines go to
stderr.

## Running the orchestrator: `scripts/run_embedding_benchmarks.py`

The orchestrator runs the warm, cold, retrieval and golden-parity passes, compares LTEmbed
with PyTorch, and writes a CSV and a text summary. It launches
`target/release/benchmark_ltembed` if it exists and otherwise falls back to
`cargo run --release --bin benchmark_ltembed`, so build the release binary first to keep
compilation out of the timings.

Without `--reference-path`, the orchestrator runs the PyTorch reference itself
(`scripts/bench_pytorch.py`). That needs `torch`, `transformers` and `numpy`, plus a local
Hugging Face snapshot of the model passed as `--model-dir`. The default `--model-dir` is
`assets/`, which does not contain the model, so pass it explicitly:

```bash
cargo build --release --bin benchmark_ltembed
python3 scripts/run_embedding_benchmarks.py \
  --model-dir /path/to/jina-embeddings-v5-text-nano-retrieval-snapshot \
  --bundle-dir gguf_bundle \
  --output-dimension 512 --threads 1 \
  --output-csv artifacts/benchmark-report.csv \
  --output-summary artifacts/benchmark-summary.txt
```

Useful options: `--scenario`, `--warmup`, `--iters`, `--cold-iters` (default 10),
`--no-include-cold-start`, `--no-include-retrieval-eval`, `--no-golden-parity`,
`--no-l2-normalize`, `--retrieval-eval-path` (default `scripts/retrieval_eval_cases.json`),
`--emit-reference` and `--reference-path`. All options are described in
[scripts/README.md](../scripts/README.md).

## The `benchmark-arm64` workflow

`.github/workflows/benchmark-arm64.yml` runs only on `workflow_dispatch`, on GitHub-hosted
`ubuntu-24.04-arm` runners. Those runners are **not** Graviton instances.

| Input | Default |
|---|---|
| `quants` | `IQ4_NL,Q5_K_M,Q8_0` |
| `warmup` / `iters` / `cold_iters` | `10` / `100` / `10` |
| `threads` | `1` |
| `output_dimension` / `l2_normalize` | `512` / `true` |
| `scenario` | empty (all) |
| `include_cold_start` / `include_retrieval_eval` | `true` / `true` |
| `retrieval_pairs` | `500` |

Jobs:

1. **`prepare`** turns `quants` into a matrix.
2. **`reference`** downloads the FP32 model snapshot, builds a CN/EN cross-lingual retrieval
   set from `tests/CN_EN_Data.csv` (`scripts/build_cn_en_retrieval_cases.py`), and runs
   PyTorch once with `--emit-reference`. Artifact: `benchmark-reference`.
3. **`benchmark`** (one job per quant) fetches the static llama.cpp release, assembles a
   bundle from `resolve/main` on Hugging Face (not pinned; see
   [bundle-format.md](./bundle-format.md#where-each-workflow-gets-its-bundle)), builds the
   release binary, runs the orchestrator with `--reference-path`, `--fixture-path`,
   `--golden-parity` and `LTEMBED_PROFILE=1`, and records `metadata.json` with
   `scripts/write_benchmark_metadata.py`. Artifact: `benchmark-arm64-<QUANT>`.
4. **`report`** runs `scripts/render_benchmark_report.py` over all quant artifacts.
   Artifact: `benchmark-report` (`report.md` and `results.json`). The report is also
   written to the run summary.

All artifacts are kept for 3 days. Copy any result you want to keep.

### How the report recommends a quant

`render_benchmark_report.py` recommends the **smallest bundle** that passes both checks:

- **Quality gate**: mean cosine ≥ `0.99` against the golden fixture, or against the
  workflow's dynamic PyTorch reference when golden parity is missing. This is stricter
  than the orchestrator's `0.98` pass thresholds.
- **Size budget**: `model.gguf` + `tokenizer.json` + `build-info.json` ≤ 230 MiB, which is
  the 250 MiB unzipped AWS Lambda package limit minus a 20 MiB allowance for a binary.

If any requested quant has no results, the report is marked **INCOMPLETE** and makes no
recommendation. If no quant passes the quality gate, it makes none either. The size
budget is a packaging target; this repository does not deploy to Lambda (see
[releases.md](./releases.md#aws-lambda)).

## Recorded results

No benchmark output is committed to the repository except the table below. In
particular, there are **no committed Graviton results** and no committed output from the
`benchmark-arm64` workflow.

### llama.cpp migration spike, July 2026

Source: [history/migrations/llama-cpp-spike-results.md](./history/migrations/llama-cpp-spike-results.md),
committed with #149 on 2026-07-09.

**Golden parity** (cosine after truncation to 512 and L2 normalization, over the 4
golden fixtures):

| Quant | GGUF size | min cosine | mean cosine |
|---|---|---|---|
| Q8_0 | 233 MB | 0.99970 | 0.99980 |
| Q5_K_M | 169 MB | 0.99663 | 0.99720 |
| Q4_K_M | 157 MB | 0.98937 | 0.99169 |

**Warm latency, in ms.**

- Environment: an **Apple Silicon aarch64 development machine, not Graviton**. The exact
  chip, OS and llama.cpp build flags were not recorded.
- Settings: `--mode warm`, `threads=1`, 50 iterations.

Use these numbers only to compare quants with each other. They do not represent
Graviton or Lambda performance.

| Scenario | Q8_0 p50 | Q8_0 p95 | Q5_K_M p50 | Q5_K_M p95 |
|---|---|---|---|---|
| `single/short` | 10.5 | 11.0 | 21.4 | 33.6 |
| `single/medium` | 18.4 | 19.5 | 32.6 | 33.1 |
| `single/long` | 466 | 480 | 741 | 748 |
| `batch/medium/8` | 146 | 147 | 261 | 283 |

On that machine Q8_0 was about 1.7–1.8× faster than Q5_K_M. The scenario set has changed
since: `single/short` no longer exists, and the medium, long and batch texts have come from
`scripts/benchmark_corpus.json` since #153 (2026-07-19), so new runs are not directly
comparable with this table.
