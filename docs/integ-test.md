# Integration Testing

## Goal

Verify the `EmbeddingEngine` (llama.cpp/GGUF) path end to end:

- tokenizer loading
- GGUF bundle loading
- typed query/document prefix handling
- last-token pooling
- `768 -> 512` truncation
- L2 normalization

## Bundle Expectations

Tier 2 tests (gated on `LTEMBED_TEST_BUNDLE_DIR`) expect a valid GGUF bundle directory:

- `model.gguf`
- `tokenizer.json` (the model's real tokenizer)
- `build-info.json` (`model_format: "gguf"`)
- regenerated `tests/fixtures/test_fixtures.json`

They also require the build/link prerequisites (aarch64-linux + `STATIC_LLAMA_DIR`). Tier 1
tests must remain runnable without local model weights (they still link the static libs but
do not run inference).

## Skipping vs. Failing Without a Bundle

By default a Tier 2 test skips (returns early and passes, printing `Skipping: …` to the
captured output) when `LTEMBED_TEST_BUNDLE_DIR` is unset, or when the directory lacks a file
the test reads: `model.gguf` and `tokenizer.json` for the engine tests, only `tokenizer.json`
for the token-id test. Local runs without a bundle therefore still pass.

`LTEMBED_REQUIRE_TEST_BUNDLE=1` turns each of those skips into a failure that names the
missing variable or file. `0`, empty or unset keeps skipping; any other value fails, so a typo
cannot quietly switch the check off.

The CI `Test` job sets `LTEMBED_REQUIRE_TEST_BUNDLE=1` at job level. The bundle step hands
`LTEMBED_TEST_BUNDLE_DIR` to later steps through `$GITHUB_ENV`. If that hand-off breaks, the
integration step fails instead of passing with no model-backed assertions run.

```bash
# Local run without a bundle: Tier 2 tests skip.
cargo test --test integration_tests
# As in CI: Tier 2 tests must run.
LTEMBED_REQUIRE_TEST_BUNDLE=1 LTEMBED_TEST_BUNDLE_DIR=/path/to/bundle \
  cargo test --test integration_tests
```

## Tier 1

Always safe for CI and local smoke runs:

- missing `model.gguf` returns `ModelLoad`
- missing tokenizer returns `ModelLoad`
- missing or malformed `build-info.json` returns `ModelLoad`
- non-`gguf` `model_format` or otherwise unsupported metadata returns `ModelLoad`
- tokenizer overlength returns `InputTooLong { max: 8192 }`
- output config validation preserves the `512`-d contract

## Tier 2

Run in CI and locally when `LTEMBED_TEST_BUNDLE_DIR` points at a bundle:

- Rust outputs match regenerated Python/Jina fixtures to the configured cosine threshold
- `HFTokenizer` token ids, attention masks and type ids (`encode` and padded `encode_batch`)
  match the Python `tokenizers` fixture exactly; this needs only the bundle's `tokenizer.json`
- output vectors are unit-normalized
- `embed_batch` ordering matches repeated single-input calls

## Fixture Contract

Fixtures must be generated with `scripts/generate_fixtures.py` and use:

- `kind`: `query` or `document`
- `text`: raw caller text without retrieval prefix
- `embedding`: final `512`-d truncated-and-normalized reference vector

If the fixture file still advertises an older dimension, the parity test skips rather than silently comparing against the wrong baseline. `LTEMBED_REQUIRE_TEST_BUNDLE` does not affect this skip.

## Token-ID Fixture Contract

`tests/fixtures/token_ids.json` is generated with `scripts/generate_token_ids.py` from the
`tokenizer.json` at the Hugging Face revision that CI pins (`HF_REVISION` in
`.github/workflows/ci.yml`). It records:

- `tokenizer`: source repo, revision and the file's `sha256`
- `tokenizers_version`: the Python `tokenizers` version that produced the ids; the script
  refuses to run unless it equals the Rust `tokenizers` version in `Cargo.lock`
- `cases`: each input as `text` or as a `repeat` spec (the 1M-space run is never stored as a
  literal), with its `single` encoding and its row of one batch encoding of all cases
- `padding`: right-padding values; batch rows store their trailing padding as a `padding` count

If the bundle's `tokenizer.json` hash differs from the recorded one, the test fails with
instructions to regenerate the fixture instead of reporting a bare id mismatch.
