# Scripts

## `generate_fixtures.py`

Regenerates `tests/fixtures/test_fixtures.json` for `jinaai/jina-embeddings-v5-text-nano-retrieval`.

- input schema: `kind + text`
- pooling: last token
- post-processing: truncate `768 -> 512`, then normalize

## `bench_pytorch.py`

Python reference runner for warm, cold, and correctness benchmark modes against the same Jina retrieval contract.

## `run_embedding_benchmarks.py`

Top-level orchestrator for LTEmbed and PyTorch runners.

Important assumptions:

- `--model-dir` points at a directory containing `tokenizer.json` and `onnx/model_q4f16.onnx`
- correctness thresholds should account for quantized ONNX output vs. Python reference
