# Issue 30 Scalar Hotspots Design

**Date:** 2026-03-16
**Issue:** `#30 perf: profile and optimize masked softmax, layer norm, and other scalar hotspots`

## Goal

Add a reproducible profiling path for the Rust BERT inference engine, identify the hottest remaining scalar kernels on macOS ARM64, and land one high-impact optimization with benchmark evidence while preserving model outputs.

## Context

The large dense operations in LTEmbed already rely on `matrixmultiply::sgemm`, and recent work has improved batching and model loading. The remaining likely hotspots are scalar and element-wise loops implemented in Rust inside [`src/models/bert.rs`](../../../src/models/bert.rs), especially:

- attention mask application before softmax
- softmax normalization
- layer norm row processing
- other per-element loops such as GELU and residual additions

Issue #30 requires evidence, not just code changes. The work therefore needs both measurement and at least one shipped optimization.

## Scope

This work includes:

- a repeatable way to collect release-mode profiling data on representative inference scenarios
- instrumentation or workflow documentation that makes kernel-level hotspots identifiable
- one concrete optimization in the attention scalar path
- tests and benchmark verification that protect numerical behavior

This work does not include:

- changing the public embedding API
- replacing `matrixmultiply`
- broad architectural refactors outside the inference hot path
- guaranteeing Linux ARM64 profiling results in this session if that hardware is unavailable locally

## Recommended Approach

### Option 1: Benchmark-only profiling

Use existing end-to-end benchmarks and infer likely hotspots from wall-clock deltas.

Pros:

- low implementation cost
- minimal code churn

Cons:

- weak attribution to specific kernels
- does not satisfy the issue as strongly as direct profiling evidence

### Option 2: Optimize likely hotspots immediately

Rewrite `softmax` and `layer_norm` first, then compare benchmarks.

Pros:

- fastest path to a code change

Cons:

- easy to optimize the wrong thing
- does not establish a profiling workflow for future issues

### Option 3: Add reproducible profiling first, then optimize the top scalar hotspot

Use release-mode profiling on representative scenarios to rank hotspots, then optimize the highest-value scalar kernel and verify with targeted benchmarks.

Pros:

- matches the issue acceptance criteria
- creates a reusable workflow for future performance work
- reduces the chance of speculative optimization

Cons:

- slightly more setup before the first code optimization

### Recommendation

Choose option 3.

The main implementation target should be the attention scalar path inside [`src/models/bert.rs`](../../../src/models/bert.rs), because the matmul-heavy paths are already delegated to optimized kernels while mask writes and softmax are still hand-written loops. The first optimization should fuse mask application with softmax preparation so the code performs fewer passes over the score rows and improves cache behavior.

## Design

### Profiling Path

Profiling will target at least two representative scenarios:

- `single/medium`
- `batch/medium/8`

These cover both latency-sensitive single inference and a throughput-oriented batched path. The profiling flow should build in release mode and run through the existing benchmark infrastructure or a small dedicated binary/script so that repeated runs use the same model load path and text corpus.

Expected profiling outputs:

- a documented command path for macOS ARM64
- captured hotspot summary identifying top self-time functions
- enough detail to compare pre- and post-optimization runs

### Hotspot Optimization

The first optimization will target the attention score post-processing path:

1. compute attention scores per head
2. apply padding mask
3. run row-wise softmax

The current implementation applies the mask in one pass over columns and then performs a separate softmax pass per row. The optimized path should reduce passes over the same memory by combining mask handling with the per-row softmax preparation. Concretely:

- replace the separate masking loop with a softmax helper that accepts the row and its mask context
- compute the row max while accounting for masked positions
- compute exponentials and normalization in the same row-oriented flow
- keep masked positions at zero probability

This change should improve both the single-item `forward` path and the batched `forward_batch` path, since they currently duplicate the same scalar pattern.

### Numerical Behavior

The optimization must preserve:

- softmax rows summing to approximately `1.0` when at least one token is unmasked
- masked positions contributing zero attention probability
- unchanged embedding outputs within a small floating-point tolerance on existing integration scenarios

### Error Handling

No new public error surface is needed. Profiling helpers may return ordinary process failures or test failures, but inference behavior should remain unchanged. If a fully masked row is observed unexpectedly, the helper should remain numerically stable and avoid NaN propagation.

## Testing Strategy

The implementation will use TDD.

Required tests:

- unit tests for the new masked-softmax behavior in [`src/models/bert.rs`](../../../src/models/bert.rs)
- coverage for masked entries becoming zero probability
- coverage for rows remaining normalized after masking
- existing `bert` and integration tests to catch output regressions

Required verification:

- targeted Rust tests for the changed helper and affected model code
- release benchmark comparison for at least one representative scenario before and after the optimization
- profiling evidence captured on macOS ARM64

## Deliverables

- code changes in [`src/models/bert.rs`](../../../src/models/bert.rs) and any small supporting benchmark/profiling files
- updated documentation describing how to reproduce profiling
- benchmark and profiling evidence suitable for issue #30 notes

## Risks And Mitigations

**Risk:** profiling overhead obscures true hot functions.
**Mitigation:** use release builds and keep the measured scenarios fixed.

**Risk:** scalar rewrites change numerical behavior.
**Mitigation:** add focused unit tests plus integration parity checks.

**Risk:** the first suspected hotspot is not the highest-value one.
**Mitigation:** make profiling the first execution step and only optimize after confirming the ranking.
