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

- `--ort-bundle-dir` points at a directory containing `model.ort`, `tokenizer.json`, `libonnxruntime.so`, and `build-info.json`
- `--output-dimension` and `--l2-normalize` describe LTEmbed post-processing explicitly
- correctness thresholds should account for quantized ONNX output vs. Python reference
- `--threads` is passed to PyTorch as `torch.set_num_threads(...)` and to LTEmbed as ONNX Runtime `with_intra_threads(...)`; the CSV `threads` column records this requested runner thread count
- Historical note: before this fix, LTEmbed rows recorded the requested `--threads` value in the CSV while the engine actually ran with 1 intra-op thread; LTEmbed rows with `threads > 1` produced before the fix are mislabeled and must not be compared against post-fix runs (relevant when using `compare_benchmarks.py` across runs)
