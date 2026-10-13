# Test Agent Guidance

These instructions apply to tests and test fixtures.

Also follow `/AGENTS.md`.

`docs/testing.md` is the detailed reference for the test suites, the Tier 1/Tier 2 split, skip-versus-fail behavior, and fixture provenance.

## Test Philosophy

Tests define behavioral contracts.

When observable behavior changes, add or update the corresponding tests rather than weakening assertions to accommodate the implementation.

Keep lightweight contract tests separate from model-backed inference tests.

## Bundle-Gated Tests

Real inference tests may require:

```text
LTEMBED_TEST_BUNDLE_DIR
```

and the normal static llama.cpp build prerequisites.

Local runs may skip bundle-gated tests when a bundle is unavailable. CI sets:

```text
LTEMBED_REQUIRE_TEST_BUNDLE=1
```

so a missing or incomplete test bundle fails instead of silently skipping. Preserve this distinction.

A skipped local model-backed test does not prove inference correctness.

Run model-backed tests whenever a change can affect tokenizer behavior, prefix handling, GGUF loading, metadata validation, pooling, batching, embedding dimensions, normalization, or llama.cpp FFI.

## Golden Embedding Fixtures

Embedding golden fixtures represent the independent Python/PyTorch reference behavior.

Never regenerate those embeddings from LTEmbed or GGUF output.

Relevant generator:

```text
scripts/generate_fixtures.py
```

Fixture regeneration must be intentional and reviewed.

The golden gates (per-input and mean cosine) are set just below the measured quantization error of the release GGUF (`docs/testing.md` § "Gates and how they were set"). Do not lower them to make a failing result pass; a result below them is a finding to investigate.

## llama.cpp Reference Fixture

`tests/fixtures/llama_cpp_reference.json` holds what upstream llama.cpp's own embedding tool computes on the release GGUF, per static-release variant. It is reference data from a tool that shares LTEmbed's kernels but none of its code, built by `scripts/build_llama_cpp_reference_tool.sh` and recorded by:

```text
scripts/generate_llama_cpp_reference.py
```

Never fill it from LTEmbed output. Regenerate it, for every variant, when the golden inputs, the release GGUF or the static llama.cpp release change.

## Tokenizer Fixtures

Tokenizer parity fixtures protect against tokenizer drift.

Relevant generator:

```text
scripts/generate_token_ids.py
```

When changing tokenizer revisions or the Rust `tokenizers` dependency, verify that fixture provenance and version metadata remain correct.

Do not update expected token IDs merely to make a failing implementation pass.

## Large Fixture Changes

Treat large generated diffs as review-sensitive.

Before accepting them, explain why regeneration was necessary, which reference implementation produced them, which model/tokenizer revision was used, and what semantic change is expected.
