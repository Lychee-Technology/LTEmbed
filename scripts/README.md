# Scripts

Python tooling for fixtures and benchmarks, plus the Git hook installer. How the pieces fit
together is described in [docs/benchmarking.md](../docs/benchmarking.md) and
[docs/testing.md](../docs/testing.md). The script tests in `tests/test_*.py` cover most of
these files.

## Fixtures

### `generate_fixtures.py`

Regenerates `tests/fixtures/test_fixtures.json` for `jinaai/jina-embeddings-v5-text-nano-retrieval`.
This is the immutable PyTorch FP32 golden; never regenerate it from GGUF or LTEmbed output.
Requires `pip install transformers torch numpy`.

- input schema: `kind + text`
- pooling: last token
- post-processing: truncate `768 -> 512`, then normalize

### `generate_token_ids.py`

Regenerates `tests/fixtures/token_ids.json`, the token-id parity fixture for `HFTokenizer`
(`pip install tokenizers==<version>` with the Rust `tokenizers` version from `Cargo.lock`,
which the script checks; no model weights). It downloads `tokenizer.json` at the pinned
`REVISION`, which must equal `HF_REVISION` in `.github/workflows/ci.yml`, and records
single and batch encodings plus the file's sha256. Rerun it after changing its inputs,
bumping the revision, or bumping `tokenizers` in `Cargo.lock`.

## Benchmarks

### `run_embedding_benchmarks.py`

Top-level orchestrator for the LTEmbed and PyTorch runners. It launches
`target/release/benchmark_ltembed` if that file exists, and otherwise
`cargo run --release --bin benchmark_ltembed`.

Important assumptions:

- `--bundle-dir` points at a GGUF bundle directory containing `model.gguf`, `tokenizer.json`, and `build-info.json`
- `--model-dir` points at a local Hugging Face snapshot of the model for the PyTorch runner. The default, `assets/`, is not such a snapshot.
- `--cold-iters` runs the cold-start pass N times per scenario (fresh process each time) and aggregates the latency distribution Python-side
- `--golden-parity` re-embeds the texts from the immutable `tests/fixtures/test_fixtures.json` golden and reports per-item cosine similarity (`mode=golden_parity` CSV rows); the golden file is never written
- `--emit-reference <path>` runs only the PyTorch retrieval pass, writes its embeddings to `<path>`, and exits. `--reference-path <path>` loads that file instead of running PyTorch, so only LTEmbed runs. The `benchmark-arm64` workflow uses this pair to run PyTorch once for all quants.
- `--output-dimension` and `--l2-normalize` describe LTEmbed post-processing explicitly
- correctness and golden-parity thresholds (`--correctness-threshold`, `--golden-parity-threshold`, both `0.98` by default) should allow for the drift of a quantized GGUF model from the PyTorch FP32 reference
- `--threads` is passed to PyTorch as `torch.set_num_threads(...)` and to `benchmark_ltembed --threads`, which sets llama.cpp `n_threads` and `n_threads_batch`; the CSV `threads` column records this requested runner thread count
- Historical note (ONNX Runtime era): before LTEmbed passed `--threads` to ONNX Runtime's `with_intra_threads(...)`, LTEmbed rows recorded the requested `--threads` value in the CSV while the engine actually ran with 1 intra-op thread; LTEmbed rows with `threads > 1` produced before that fix are mislabeled and must not be compared against later runs (relevant when using `compare_benchmarks.py` across runs)
- `--ltembed-cargo-features` is only used by the `cargo run` fallback. The crate defines no Cargo features, so leave it empty.

### `bench_pytorch.py`

PyTorch retrieval-eval reference runner (retrieval-only; other `--mode` values exit with an
error). Requires `torch`, `transformers` and `numpy`.

### `benchmark_corpus.json`

Deterministic texts for the `single/medium`, `single/long`, and `batch/medium/8` benchmark
scenarios. Embedded into the Rust binary via `include_str!` and read by
`build_cn_en_retrieval_cases.py`, so both sides embed byte-identical inputs.

### `retrieval_eval_cases.json`

A small built-in retrieval-eval case (`mini-retrieval-v1`). It is the default
`--retrieval-eval-path` of `run_embedding_benchmarks.py`.

### `build_cn_en_retrieval_cases.py`

Generates the CN/EN cross-lingual retrieval-eval case from `tests/CN_EN_Data.csv` and, with
`--fixture-output`, the resolved latency fixture covering every benchmark scenario.

### `write_benchmark_metadata.py`

Writes a quant matrix job's `metadata.json` (model/bundle sizes and SHA, static llama
tag/SHA/contract version, runner + CPU flags, run parameters, scenario list) with proper
JSON number types. Consumed by `render_benchmark_report.py`.

### `render_benchmark_report.py`

Aggregates every quant's `metadata.json` + `benchmark-report.csv` into the cross-quant
`results.json` + `report.md` (normalized per quant × scenario × warm/cold records, parity
and retrieval summaries, and the recommended quant under the Lambda bundle-size budget).
The recommendation rules are in
[docs/benchmarking.md](../docs/benchmarking.md#how-the-report-recommends-a-quant).

### `compare_benchmarks.py`

Compares two or more `benchmark-report.csv` files. The first file is the baseline.

```bash
python3 scripts/compare_benchmarks.py baseline.csv candidate.csv
python3 scripts/compare_benchmarks.py --label before:a.csv after:b.csv --metric p95_ms
```

By default it compares `mean_ms` of `warm_latency` rows; see `--mode`, `--metric` and
`--impl`.

### `compare_embedding_outputs.py`

Compares one scenario's embeddings between an LTEmbed and a PyTorch JSON payload and
reports cosine and norms. It expects payloads with `results[].scenario` and
`results[].embeddings`, the output of the old `correctness` mode. Neither current runner
produces that shape: `benchmark_ltembed` lost its `correctness` mode in #149, and
`bench_pytorch.py` runs only retrieval.

### `compare_q4f16_onnx_vs_pytorch.py`

ONNX Runtime era. Compares a q4f16 ONNX model, run through `onnxruntime`, with the PyTorch
reference. It does not apply to GGUF bundles.

## Git hooks

### `install-git-hooks.sh`

Sets `core.hooksPath=.githooks` for this clone. See
[docs/development.md](../docs/development.md#git-hooks).
