# Scripts

## `generate_fixtures.py`

Regenerates `tests/fixtures/test_fixtures.json` for `jinaai/jina-embeddings-v5-text-nano-retrieval`.

- input schema: `kind + text`
- pooling: last token
- post-processing: truncate `768 -> 512`, then normalize

## `bench_pytorch.py`

Python reference runner for warm, cold, correctness, and mini retrieval eval modes against the same Jina retrieval contract.

## `retrieval_eval_cases.json`

Small retrieval ranking fixture used by the benchmark harness to report `recall@1` and `mrr@3`.

## `run_embedding_benchmarks.py`

Top-level orchestrator for LTEmbed and PyTorch runners.

Important assumptions:

- `--ort-bundle-dir` points at a directory containing `model.ort`, `tokenizer.json`, `libonnxruntime.so`, and `build-info.json`
- `--output-dimension` and `--l2-normalize` describe LTEmbed post-processing explicitly
- correctness thresholds should account for quantized ONNX output vs. Python reference
- retrieval eval uses `scripts/retrieval_eval_cases.json` unless `--retrieval-eval-path` overrides it
