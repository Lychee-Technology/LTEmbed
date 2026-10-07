# Test Agent Guidance

These instructions apply to tests and test fixtures.

Also follow `/AGENTS.md`.

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
