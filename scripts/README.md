# scripts/

Helper scripts for benchmarking, profiling, and data preparation.

---

## run_embedding_benchmarks.py

Unified benchmark orchestrator. Runs LTEmbed, Candle, and PyTorch against the
same scenario suite and writes a normalized CSV report covering cold-start
latency, warm latency, and correctness vs PyTorch.

This is the script invoked by the `benchmark-arm64` CI workflow.

**Prerequisites**

```bash
pip install numpy torch transformers huggingface_hub
cargo build --release --bin benchmark_ltembed
```

**Basic usage**

```bash
python3 scripts/run_embedding_benchmarks.py \
  --model-dir .cache/models/intfloat-e5-small-v2 \
  --output-csv artifacts/report.csv \
  --output-summary artifacts/summary.txt
```

**Key options**

| Option | Default | Description |
|--------|---------|-------------|
| `--model-dir` | — | Local directory with model assets (`config.json`, `tokenizer.json`, `model.safetensors`) |
| `--model-id` | `intfloat/e5-small-v2` | HuggingFace model ID (for metadata) |
| `--model-size` | `fp32` | `fp32` or `fp16` — selects `model.safetensors` or `model_fp16.safetensors` |
| `--warmup` | `10` | Warmup iterations per scenario |
| `--iters` | `100` | Timed iterations per scenario |
| `--threads` | `1` | CPU threads per implementation |
| `--no-include-cold-start` | — | Skip cold-start measurements |
| `--no-include-correctness` | — | Skip correctness checks vs PyTorch |
| `--output-csv` | — | Path for CSV report |
| `--output-summary` | — | Path for human-readable summary |

**Testing a custom matrixmultiply build**

```bash
# From a local checkout
python3 scripts/run_embedding_benchmarks.py \
  --model-dir .cache/models/intfloat-e5-small-v2 \
  --ltembed-matrixmultiply-source path \
  --ltembed-matrixmultiply-path ../matrixmultiply \
  --output-csv artifacts/report.csv

# From a git branch
python3 scripts/run_embedding_benchmarks.py \
  --model-dir .cache/models/intfloat-e5-small-v2 \
  --ltembed-matrixmultiply-source git \
  --ltembed-matrixmultiply-git https://github.com/bluss/matrixmultiply \
  --ltembed-matrixmultiply-rev main \
  --output-csv artifacts/report.csv
```

---

## compare_benchmarks.py

Compare two or more benchmark CSV reports produced by `run_embedding_benchmarks.py`.
Rows are matched by key columns (`model_id`, `implementation`, `scenario`, `mode`,
`batch_size`, `text_profile`, `threads`, `warmup_iters`, `timed_iters`).
Improvements ✅ / regressions ❌ are flagged at a ±2% threshold.

**Usage**

```bash
# Two CSVs — first is baseline
python3 scripts/compare_benchmarks.py main.csv candidate.csv

# Custom labels
python3 scripts/compare_benchmarks.py main:main.csv neon:neon.csv fuse:fuse.csv

# Filter to a single implementation
python3 scripts/compare_benchmarks.py main.csv candidate.csv --impl ltembed

# Different metric
python3 scripts/compare_benchmarks.py main.csv candidate.csv --metric median_ms

# Include all modes (cold-start, correctness, warm)
python3 scripts/compare_benchmarks.py main.csv candidate.csv --mode ""
```

**Options**

| Option | Default | Description |
|--------|---------|-------------|
| `--impl` | — | Filter by `implementation` (e.g. `ltembed`, `pytorch`, `candle`) |
| `--mode` | `warm_latency` | Filter by `mode` column; pass `""` for all rows |
| `--metric` | `mean_ms` | Metric to compare: `mean_ms`, `median_ms`, `p95_ms`, `p99_ms`, `min_ms`, `max_ms` |

---

## bench_pytorch.py

Low-level PyTorch benchmark runner. Called internally by
`run_embedding_benchmarks.py`; not intended for direct use. Outputs
machine-readable JSON for warm latency, cold start, or correctness checks.

---

## convert_to_fp16.py

Convert a float32 safetensors model to float16 (~50% size reduction).
The resulting file can be used with `--model-size fp16`.

**Usage**

```bash
pip install torch safetensors

python3 scripts/convert_to_fp16.py \
  --input  assets/model.safetensors \
  --output assets/model_fp16.safetensors
```

---

## generate_fixtures.py

Generate golden test fixtures (`tests/fixtures/test_fixtures.json`) from
the e5-small-v2 model via HuggingFace Transformers. Run this once after
changing the model or adding new test sentences.

**Usage**

```bash
pip install transformers torch numpy huggingface_hub

python3 scripts/generate_fixtures.py
```

Output: `tests/fixtures/test_fixtures.json`

---

## profile_projection_gemm_perf.py

Profile projection-heavy GEMM paths using Linux `perf`. Builds
`benchmark_ltembed`, records a warm run, and annotates the hottest
`matrixmultiply::*` symbols. **Linux ARM64 only** (requires `perf`).

**Usage**

```bash
python3 scripts/profile_projection_gemm_perf.py \
  --model-dir .cache/models/intfloat-e5-small-v2 \
  --scenario single/long \
  --output-dir perf-reports/
```

**Key options**

| Option | Default | Description |
|--------|---------|-------------|
| `--scenario` | `single/long` | Scenario name to profile |
| `--model-dir` | — | Path to model directory |
| `--iters` | `100` | Iterations to record |
| `--perf-event` | `cycles` | `perf record` event (e.g. `cache-misses`) |
| `--call-graph` | `fp` | Call graph mode (`fp`, `dwarf`, `lbr`) |
| `--output-dir` | — | Directory for perf report files |
| `--skip-build` | — | Skip `cargo build` and use existing binary |

---

## install-git-hooks.sh

One-time setup: installs the repo's Git hooks (`pre-commit`, `pre-push`)
from `.githooks/` into the local Git config.

```bash
bash scripts/install-git-hooks.sh
```
