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
- **Correctness vs. PyTorch**: cosine between LTEmbed and the PyTorch F32 model (run by
  SentenceTransformers, `scripts/bench_pytorch.py`) on the retrieval-eval texts. The default pass threshold is `0.98`
  (`--correctness-threshold`).
- **Golden parity**: cosine against the immutable fixture `tests/fixtures/test_fixtures.json`
  (see [testing.md](./testing.md#golden-fixture-testsfixturestest_fixturesjson)), truncated
  to `--output-dimension`. Threshold `0.98` per input by default
  (`--golden-parity-threshold`), looser than the test suite's gates (0.985 per input, 0.993
  mean). It is skipped unless `--output-dimension` is one of the fixture's widths (768, 512,
  256, 128).
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
(`scripts/bench_pytorch.py`). That needs `sentence-transformers`, `transformers`, `torch`,
`torchvision`, `pillow` and `numpy` (the versions are in the script's docstring), plus a
local snapshot of `unsloth/embeddinggemma-2` at the revision
`scripts/assemble_gguf_bundle.py` pins, passed as `--model-dir`. The default `--model-dir`
is `assets/`, which the repository does not contain, so pass it explicitly:

```bash
cargo build --release --bin benchmark_ltembed
python3 scripts/run_embedding_benchmarks.py \
  --model-dir /path/to/embeddinggemma-2-snapshot \
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
| `quants` | `UD-Q4_K_XL,UD-Q5_K_XL,Q8_0` (any quant pinned in `scripts/assemble_gguf_bundle.py`) |
| `warmup` / `iters` / `cold_iters` | `10` / `100` / `10` |
| `threads` | `1` |
| `output_dimension` / `l2_normalize` | `512` / `true` |
| `scenario` | empty (all) |
| `include_cold_start` / `include_retrieval_eval` | `true` / `true` |
| `retrieval_pairs` | `500` |

Jobs:

1. **`prepare`** turns `quants` into a matrix.
2. **`reference`** downloads the safetensors model at the pinned tokenizer revision,
   installs pinned CPU versions of `torch`, `torchvision`, `transformers` and
   `sentence-transformers`, builds a CN/EN cross-lingual retrieval set from
   `tests/CN_EN_Data.csv` (`scripts/build_cn_en_retrieval_cases.py`), and runs PyTorch once
   with `--emit-reference`. Artifact: `benchmark-reference`.
3. **`benchmark`** (one job per quant) fetches the static llama.cpp release, assembles the
   quant's bundle with `scripts/assemble_gguf_bundle.py` (pinned revisions and hashes; see
   [bundle-format.md](./bundle-format.md#assembling-a-bundle)), builds the release binary, runs the orchestrator with `--reference-path`, `--fixture-path`,
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
  than the orchestrator's `0.98` pass thresholds, and looser than the test suite's 0.993
  mean gate.
- **Size budget**: every file in the bundle directory (`model.gguf`, `build-info.json`,
  `SHA256SUMS`, `provenance.json`: `metadata.json`'s `bundle_size_bytes`) ≤ 230 MiB, which
  is the 250 MiB unzipped AWS Lambda package limit minus a 20 MiB allowance for a binary.

If any requested quant has no results, the report is marked **INCOMPLETE** and makes no
recommendation. If no quant passes the quality gate, it makes none either. The size
budget is a packaging target; this repository does not deploy to Lambda (see
[releases.md](./releases.md#aws-lambda)).

## Recorded results

No benchmark output is committed to the repository except the tables below. In
particular, there are **no committed Graviton or other aarch64 timings** and no committed
output from the `benchmark-arm64` workflow. The previous model's results, from the
llama.cpp migration spike of July 2026, are in
[history/models/jina-embeddings-v5-text-nano-retrieval.md](./history/models/jina-embeddings-v5-text-nano-retrieval.md#recorded-results-llamacpp-migration-spike-july-2026).

### EmbeddingGemma 2 on x86-64-v3, October 2026

Measured during the EmbeddingGemma 2 migration, on one machine:

- AMD Ryzen 5 5600G (Zen 3, 6 cores / 12 threads, x86-64-v3), 27 GiB RAM, Fedora Linux
  (kernel 7.2), nothing else running;
- static llama.cpp `v0.1.159-2` (`b11577`), `x86_64-v3` variant; flash attention on;
- `benchmark_ltembed --mode warm`, `--output-dimension 512 --l2-normalize true`, 3 warm-up
  and 20 timed iterations (`single/long`: 1 and 3). Values are medians in ms.

These are single-machine numbers. They show how the quants, thread counts and input lengths
compare; they do not predict Graviton or Lambda latency.

**Warm latency by quant**, 6 threads. Token counts include the prompt, `<bos>` and `<eos>`.

| Quant | `single/zh` (13) | `single/en` (14) | `single/medium` (90) | `batch/medium/8` (8 × 90) | `single/long` (4075) |
|---|---|---|---|---|---|
| `UD-Q4_K_XL` | 18.4 | 19.4 | 68.7 | 555 | 5374 |
| `UD-Q5_K_XL` | 24.6 | 25.1 | 85.2 | 701 | 5288 |
| `UD-Q6_K_XL` | 26.7 | 27.0 | 89.8 | 735 | 5390 |
| `Q8_0` | 23.0 | 23.7 | 98.8 | 786 | 6293 |
| `BF16` | 24.4 | 25.3 | 101.6 | 805 | 6553 |

**Warm latency of `UD-Q4_K_XL` by thread count**:

| Threads | `single/zh` | `single/en` | `single/medium` | `batch/medium/8` | `single/long` |
|---|---|---|---|---|---|
| 1 | 51.3 | 54.5 | 265.7 | 1873 | 23627 |
| 4 | 21.0 | 22.2 | 85.1 | 666 | 6887 |
| 6 | 18.4 | 19.4 | 68.7 | 555 | 5374 |

- `batch/medium/8` costs eight single calls, as expected from one `llama_encode` per input.
- Output width does not matter: at 768 instead of 512 dimensions, `single/medium` takes
  69.5 ms and `batch/medium/8` 562 ms. Truncation and normalization take about 2 µs.
- The bundle's `max_length` does not matter for inputs that fit: `single/zh` takes 51.4,
  51.5, 51.7 and 51.9 ms at 1 thread with `max_length` 8192, 2048, 512 and 128.
- Loading the engine takes 1.3–1.4 s. A cold start of `single/zh` (load and one embedding
  in a fresh process, 6 threads) takes about 1.2 s. Both were measured while the engine
  still parsed `tokenizer.json` (0.65 s); without it, loading takes about 0.42 s (6
  threads, warm page cache).
- Memory and first-call latency for inputs of up to 8192 tokens, and the effect of flash
  attention, are in [architecture.md](./architecture.md#long-inputs-and-memory).

**Quality by quant.** Cosine against the PyTorch F32 model at 512 dimensions, over the 25
golden inputs (10 to 8192 tokens) and over the 2000 texts of the CN/EN set that
`scripts/build_cn_en_retrieval_cases.py --num-pairs 500` builds from `tests/CN_EN_Data.csv`
(500 Chinese–English sentence pairs, each sentence embedded as a query and as a document);
and that set's retrieval metrics, where each query must find both sentences of its pair:

| Quant | Golden min / mean | CN/EN min / 1st percentile / mean | CN/EN texts below 0.985 | recall@3 | both@3 |
|---|---|---|---|---|---|
| `UD-Q4_K_XL` | 0.98972 / 0.99486 | 0.9614 / 0.9854 / 0.9954 | 20 of 2000 | 0.9725 | 0.945 |
| `UD-Q5_K_XL` | 0.99702 / 0.99852 | 0.9911 / 0.9962 / 0.9987 | 0 | 0.973 | 0.946 |
| `UD-Q6_K_XL` | 0.99839 / 0.99926 | 0.9949 / 0.9983 / 0.9993 | 0 | — | — |
| `Q8_0` | 0.99962 / 0.99985 | 0.9986 / 0.9997 / 0.9999 | 0 | 0.969 | 0.938 |
| `BF16` | 0.99999 / 1.00000 | 0.9999 / 1.0000 / 1.0000 | 0 | — | — |
| PyTorch F32 | 1 | 1 | — | 0.969 | 0.938 |

- `BF16` matches the F32 model on every text, so the drift of the smaller quants is
  quantization error, not an LTEmbed error. It shrinks with each larger quant.
- `UD-Q4_K_XL` keeps retrieval quality on this set, but 1% of its short sentences drift to
  cosine 0.96–0.985, below the per-input gate the test suite applies to its 25 golden
  inputs. The worst is the English sentence "The worker is pushing the distaff." as a
  query (0.9614). Deployments that compare scores across models or depend on absolute
  thresholds should measure on their own data, or use `UD-Q5_K_XL` or larger where the
  size budget allows.
- Recall@1 is at most 0.5 here, because each query has two relevant documents and recall
  counts the fraction found.
- The orchestrator and `render_benchmark_report.py` reproduce these runs: with the
  `UD-Q4_K_XL`, `UD-Q5_K_XL` and `Q8_0` results, the report recommends `UD-Q4_K_XL` as the
  only quant within the 230 MiB budget. Those bundles still held the 32 MB `tokenizer.json`;
  without it `UD-Q5_K_XL` (200.3 MiB) fits too, and the report's smallest-bundle rule
  still picks `UD-Q4_K_XL` ([releases.md](./releases.md#why-ud-q4_k_xl)).
