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
  line per scenario to stderr with the mean of each `EmbedBatchProfile` field: `batch_size`,
  `seq_len`, `prefix_ms`, `tokenize_ms`, `run_ms`, `postprocess_ms` and `total_ms`.
  `run_ms` covers building each input's `llama_batch`, `llama_encode` and the copy of its
  pooled vector; `total_ms` is the whole call, so it also covers the input size checks.
- **Correctness vs. PyTorch**: cosine between LTEmbed and the PyTorch F32 model (run by
  SentenceTransformers, `scripts/bench_pytorch.py`) on the retrieval-eval texts. The default pass threshold is `0.98`
  (`--correctness-threshold`).
- **Golden parity**: cosine against the immutable fixture `tests/fixtures/test_fixtures.json`
  (see [testing.md](./testing.md#golden-fixture-testsfixturestest_fixturesjson)), truncated
  to `--output-dimension`. Threshold `0.98` per input by default
  (`--golden-parity-threshold`), looser than the test suite's gate (0.995 per input). It is
  skipped unless `--output-dimension` is one of the fixture's widths (768, 512,
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
  than the orchestrator's `0.98` pass thresholds, and looser than the test suite's gate,
  which requires 0.995 for every input.
- **Size budget**: every file in the bundle directory (`model.gguf`, `build-info.json`,
  `SHA256SUMS`, `provenance.json`: `metadata.json`'s `bundle_size_bytes`) ≤ 230 MiB, which
  is the 250 MiB unzipped AWS Lambda package limit minus a 20 MiB allowance for a binary.

If any requested quant has no results, the report is marked **INCOMPLETE** and makes no
recommendation. If no quant passes the quality gate, it makes none either. The size
budget is a packaging target; this repository does not deploy to Lambda (see
[releases.md](./releases.md#aws-lambda)).

## Recorded results

No benchmark output is committed to the repository except the tables below. In
particular, there are **no committed Graviton timings** and no committed output from the
`benchmark-arm64` workflow. The aarch64 timings are from a Raspberry Pi 5, below, and the
first-call times of a GitHub ARM runner in
[architecture.md](./architecture.md#long-inputs-and-memory). The previous model's results, from the
llama.cpp migration spike of July 2026, are in
[history/models/jina-embeddings-v5-text-nano-retrieval.md](./history/models/jina-embeddings-v5-text-nano-retrieval.md#recorded-results-llamacpp-migration-spike-july-2026).

### EmbeddingGemma 2 on x86-64-v3, October 2026

Measured during the EmbeddingGemma 2 migration, on one machine:

- AMD Ryzen 5 5600G (Zen 3, 6 cores / 12 threads, x86-64-v3), 27 GiB RAM, Fedora Linux
  (kernel 7.2), nothing else running;
- static llama.cpp `v0.1.159-2` (`b11577`), `x86_64-v3` variant; flash attention on;
- the engine tokenizing with the GGUF's vocabulary, bundles without `tokenizer.json`;
- `benchmark_ltembed --mode warm`, `--output-dimension 512 --l2-normalize true`, 3 warm-up
  and 20 timed iterations (`single/long`: 1 and 2 or 3). Values are medians in ms.

These are single-machine numbers. They show how the quants, thread counts and input lengths
compare; they do not predict Graviton or Lambda latency.

**Long inputs vary between processes.** The same bundle, opened through paths of
different lengths, gives `single/long` times that differ by up to 13% between processes
(`UD-Q5_K_XL`: 5330 to 5914 ms), while repeated calls in one process differ by under 1%.
Another copy of the same GGUF, evicting it from the page cache, and padding the environment
by up to 112 bytes change nothing; the likely cause, not confirmed, is the process's memory
layout, which the path's length shifts. `BF16` varies by 0.5%, the quantized GGUFs by
9–13%, short inputs by 2–4%. A single process's `single/long` time is therefore one sample
of that spread. The `single/long` column below is the median over eight such paths, with
their range; the ranges of the 4-, 5- and 6-bit quants overlap.

**Warm latency by quant**, 6 threads. Token counts include the prompt, `<bos>` and `<eos>`.

| Quant | `single/zh` (13) | `single/en` (14) | `single/medium` (90) | `batch/medium/8` (8 × 90) | `single/long` (4075), median (range) |
|---|---|---|---|---|---|
| `UD-Q4_K_XL` | 18.7 | 19.6 | 69.6 | 557 | 5521 (5000–5652) |
| `UD-Q5_K_XL` | 24.3 | 25.0 | 84.1 | 689 | 5816 (5330–5914) |
| `UD-Q6_K_XL` | 26.3 | 27.0 | 87.2 | 715 | 5747 (5446–6004) |
| `Q8_0` | 22.6 | 23.7 | 95.7 | 759 | 6765 (6274–6821) |
| `BF16` | 24.5 | 25.6 | 98.6 | 780 | 6625 (6618–6648) |

**Warm latency of `UD-Q5_K_XL` by thread count**. `single/long` at 1 and 4 threads is
one process each:

| Threads | `single/zh` | `single/en` | `single/medium` | `batch/medium/8` | `single/long` |
|---|---|---|---|---|---|
| 1 | 67.8 | 68.8 | 279.4 | 2190 | 25161 |
| 4 | 27.1 | 28.0 | 104.9 | 810 | 7753 |
| 6 | 24.3 | 25.0 | 84.1 | 689 | 5816 |

- `batch/medium/8` costs eight single calls, as expected from one `llama_encode` per input.
- Output width does not matter: at 768 instead of 512 dimensions, `single/medium` takes
  84.1 ms and `batch/medium/8` 691 ms. Truncation and normalization take about 2 µs.
- The bundle's `max_length` does not matter for inputs that fit: `single/zh` takes 67.8,
  67.9, 67.6 and 67.9 ms at 1 thread with `max_length` 8192, 2048, 512 and 128.
- Loading the engine takes 0.48–0.53 s (warm page cache). A cold start of `single/zh`
  (load and one embedding in a fresh process, 6 threads) takes 0.50–0.51 s.
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
- `UD-Q5_K_XL` is closer to the F32 model than `UD-Q4_K_XL` on every one of the 2000 texts
  (by 0.0033 on average) and on every golden input at every trained width; its mean
  error, 1 − cosine, is 3.6 times smaller on the CN/EN set and 3.5 times on the golden
  inputs at 512 dimensions. The two quants also differ from each other more
  than either differs from the F32 model: the cosine between their vectors of the same text
  at 512 dimensions is at least 0.98805 (mean 0.99383) over the golden inputs, and at
  least 0.9458 (mean 0.9945; 886 texts below 0.995) over the CN/EN set. Their errors point
  in different directions, so vectors from different quants do not belong in one index.
- `UD-Q4_K_XL` keeps retrieval quality on this set, but 1% of its short sentences drift to
  cosine 0.96–0.985. The worst is the English sentence "The worker is pushing the
  distaff." as a query (0.9614). This tail is why the release ships `UD-Q5_K_XL`, whose
  worst is 0.9911; 10 of its 2000 texts fall below 0.995, the per-input gate the test suite
  applies to its 25 golden inputs. Deployments that compare scores across models or depend
  on absolute thresholds should still measure on their own data.
- Recall@1 is at most 0.5 here, because each query has two relevant documents and recall
  counts the fraction found.
- The orchestrator and `render_benchmark_report.py` reproduce these runs: with the
  `UD-Q4_K_XL`, `UD-Q5_K_XL` and `Q8_0` results, the report recommends `UD-Q4_K_XL` as the
  only quant within the 230 MiB budget. Those bundles still held the 32 MB `tokenizer.json`;
  without it `UD-Q5_K_XL` (200.3 MiB) fits too. The report's smallest-bundle rule still
  picks `UD-Q4_K_XL`, because its mean-cosine gate does not see the tail above; the release
  ships `UD-Q5_K_XL` anyway ([releases.md](./releases.md#why-ud-q5_k_xl)).

### EmbeddingGemma 2 on a Raspberry Pi 5 (aarch64), October 2026

Measured natively with the same probe, inputs and `benchmark_ltembed` flags as the x86
tables above, at 4, 2 and 1 threads:

- Raspberry Pi 5 Model B (4 × Cortex-A76, 2.4 GHz), 16 GB RAM, Ubuntu 26.04 (kernel 7.0,
  4 KiB pages); every run had 0 major page faults, so swap played no part;
- static llama.cpp `v0.1.159-2`, `aarch64-graviton2` variant: the same Graviton2-targeted
  kernels CI's ARM job runs, since the release has no runtime ISA dispatch; flash attention
  on;
- the release bundle (`UD-Q5_K_XL`), `--output-dimension 512 --l2-normalize true`, 3
  warm-up and 20 timed iterations (`single/long`: 1 and 3). Values are medians in ms.
  `single/long` is one process per thread count; its spread between processes was not
  measured here;
- temperature, sampled every 2 s: the firmware's soft temperature limit was active for
  about 10 s, at 81 °C, at the end of the 8192-token first call at 4 threads. Otherwise the
  ARM clock read 2400 MHz and no throttling or capping flag was set.

These are one Raspberry Pi's numbers. They do not predict Graviton or Lambda latency.

**Warm latency of `UD-Q5_K_XL` by thread count**:

| Threads | `single/zh` | `single/en` | `single/medium` | `batch/medium/8` | `single/long` |
|---|---|---|---|---|---|
| 1 | 145.4 | 160.9 | 878.5 | 6749 | 99985 |
| 2 | 79.5 | 86.5 | 464.9 | 3640 | 53025 |
| 4 | 60.8 | 63.3 | 357.6 | 2687 | 38030 |

- At 4 threads the Pi takes 2.5 times as long as the x86 host at 6 threads on `single/zh`,
  4.3 times on `single/medium` and 6.5 times on `single/long`.
- At 768 dimensions, `single/medium` took 386.7 ms and `batch/medium/8` 2765 ms, 3–8% more
  than at 512. They ran last, after the 1-thread runs, in a warm phase that reached
  80.7 °C; on x86 the width makes no difference.
- The bundle's `max_length` does not matter for inputs that fit: `single/zh` takes 145.2 to
  145.8 ms at 1 thread and 59.5 to 60.4 ms at 4 threads with `max_length` 128, 512, 2048 and
  8192.
- Loading the engine takes 1.32–1.39 s (1.82 s for the session's first load). A cold start
  of `single/zh` (load and one embedding in a fresh process, 4 threads) takes 1.28–1.29 s.

**Memory and first call**: peak resident memory (`/usr/bin/time -v`) of one process that
loads the release bundle (`max_length` 8192) and embeds one input, and the time of that
first call:

| Input tokens | Peak RSS | First call, 4 threads | First call, 1 thread |
|---|---|---|---|
| 14 | 0.41 GiB | 0.08 s | 0.17 s |
| 1024 | 1.59 GiB | 5.6 s | 15.2 s |
| 2039 | 2.60 GiB | 15.0 s | — |
| 4074 | 4.84 GiB | 38.8 s | 99.5 s |
| 8192 | 9.48 GiB (10.2 GB) | 120 s | 302 s |

Peak RSS is 0.09–0.12 GiB above the x86 host's at every length, and the thread count does
not change it. With `max_length` 2048, the 2039-token input peaks at the same 2.60 GiB and
the 1024-token one at 1.51 GiB.

These peaks include the 1 MiB per input token of logits that `v0.1.159-2` reserved and
`v0.1.159-3` no longer does. Measured again with `v0.1.159-3` in the same way, the Pi peaks
at 0.42, 0.57, 0.92 and 1.48 GiB (1.58 GB) for 14, 1024, 4074 and 8192 tokens, at 4 threads
and, for the first three, at 1 thread. That is 0.01–0.03 GiB above the aarch64 runner in
[architecture.md](./architecture.md#long-inputs-and-memory). The first calls took 0.09–0.20,
5.7, 39.1 and 103 s at 4 threads, and 0.17, 15.3 and 99.8 s at 1 thread; the 8192-token call
reached 84 °C. One engine that embeds 8192, 14, 8192 and 1024 tokens in turn holds 1.48 GiB
from the first input on, and its 8192-token calls took 118.7 and 118.9 s.
