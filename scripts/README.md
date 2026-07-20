# Scripts

## `generate_fixtures.py`

Regenerates `tests/fixtures/test_fixtures.json` for `jinaai/jina-embeddings-v5-text-nano-retrieval`.

- input schema: `kind + text`
- pooling: last token
- post-processing: truncate `768 -> 512`, then normalize

## `bench_pytorch.py`

PyTorch retrieval-eval reference runner (retrieval-only).

## `run_embedding_benchmarks.py`

Top-level orchestrator for LTEmbed and PyTorch runners.

Important assumptions:

- `--bundle-dir` points at a GGUF bundle directory containing `model.gguf`, `tokenizer.json`, and `build-info.json`
- `--cold-iters` runs the cold-start pass N times per scenario (fresh process each time) and aggregates the latency distribution Python-side
- `--golden-parity` re-embeds the texts from the immutable `tests/fixtures/test_fixtures.json` golden and reports per-item cosine similarity (`mode=golden_parity` CSV rows); the golden file is never written
- `--output-dimension` and `--l2-normalize` describe LTEmbed post-processing explicitly
- correctness thresholds should account for quantized ONNX output vs. Python reference
- `--threads` is passed to PyTorch as `torch.set_num_threads(...)` and to LTEmbed as ONNX Runtime `with_intra_threads(...)`; the CSV `threads` column records this requested runner thread count
- Historical note: before this fix, LTEmbed rows recorded the requested `--threads` value in the CSV while the engine actually ran with 1 intra-op thread; LTEmbed rows with `threads > 1` produced before the fix are mislabeled and must not be compared against post-fix runs (relevant when using `compare_benchmarks.py` across runs)

## `benchmark_corpus.json`

Deterministic texts for the `single/medium`, `single/long`, and `batch/medium/8` benchmark
scenarios. Embedded into the Rust binary via `include_str!` and read by
`build_cn_en_retrieval_cases.py`, so both sides embed byte-identical inputs.

## `build_cn_en_retrieval_cases.py`

Generates the CN/EN cross-lingual retrieval-eval case from `tests/CN_EN_Data.csv` and, with
`--fixture-output`, the resolved latency fixture covering every benchmark scenario.

## `write_benchmark_metadata.py`

Writes a quant matrix job's `metadata.json` (model/bundle sizes and SHA, static llama
tag/SHA/contract version, runner + CPU flags, run parameters, scenario list) with proper
JSON number types. Consumed by `render_benchmark_report.py`.

## `render_benchmark_report.py`

Aggregates every quant's `metadata.json` + `benchmark-report.csv` into the cross-quant
`results.json` + `report.md` (normalized per quant × scenario × warm/cold records, parity
and retrieval summaries, and the recommended quant under the Lambda bundle-size budget).
