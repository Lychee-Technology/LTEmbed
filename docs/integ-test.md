# Integration Testing

## Goal

Verify the `OnnxEngine` path end to end:

- tokenizer loading
- ORT bundle loading
- typed query/document prefix handling
- last-token pooling
- `768 -> 512` truncation
- L2 normalization

## Bundle Expectations

Tier 2 tests expect:

- a valid `ort_bundle/`
  - `model.ort`
  - `tokenizer.json`
  - `libonnxruntime.so`
  - `build-info.json`
- regenerated `tests/fixtures/test_fixtures.json`

Tier 1 tests must remain runnable without local model weights.

## Tier 1

Always safe for CI and local smoke runs:

- missing `model.ort` returns `ModelLoad`
- missing `libonnxruntime.so` returns `ModelLoad`
- missing tokenizer returns `ModelLoad`
- missing or malformed `build-info.json` returns `ModelLoad`
- unsupported metadata returns `ModelLoad`
- tokenizer overlength returns `InputTooLong { max: 8192 }`
- output config validation preserves the `512`-d contract

## Tier 2

Local or manually gated checks:

- Rust outputs match regenerated Python/Jina fixtures to the configured cosine threshold
- output vectors are unit-normalized
- `embed_batch` ordering matches repeated single-input calls

## Fixture Contract

Fixtures must be generated with `scripts/generate_fixtures.py` and use:

- `kind`: `query` or `document`
- `text`: raw caller text without retrieval prefix
- `embedding`: final `512`-d truncated-and-normalized reference vector

If the fixture file still advertises an older dimension, the parity test skips rather than silently comparing against the wrong baseline.
