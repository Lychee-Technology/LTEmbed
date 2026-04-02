# Main Standards Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve maintainability on the ORT-only `main` line by addressing the three largest standards gaps identified in the coding-standard review without changing LTEmbed behavior.

**Architecture:** Keep the cleanup split into three independent refactor tracks: the Rust benchmark CLI, the ORT engine and error model, and the Python benchmark orchestrator. Each track should preserve current user-visible behavior, current JSON and CSV contracts, and current ORT-only scope while reducing file size, tightening boundaries, and making future changes safer.

**Tech Stack:** Rust 2021, Python 3, `thiserror`, `serde`, `ort`, `gh` CLI, existing Rust and Python test suites

---

## File Map

| File | Role in cleanup |
|---|---|
| `src/bin/benchmark_ltembed.rs` | Split CLI parsing and per-mode dispatch into smaller typed helpers |
| `src/benchmarking.rs` | Shared benchmark scenario helpers consumed by the benchmark binary |
| `src/engine.rs` | Separate bundle validation, ORT init, inference preparation, and postprocessing responsibilities |
| `src/error.rs` | Replace broad stringly buckets with more structured error variants where it improves clarity |
| `src/traits/tokenizer.rs` | Keep tokenizer-facing error integration aligned with the error refactor |
| `scripts/run_embedding_benchmarks.py` | Break orchestration into smaller units while preserving workflow-facing behavior |
| `tests/integration_tests.rs` | Preserve engine behavior during refactors |
| `tests/benchmarking_support_tests.rs` | Protect scenario and benchmark helper behavior |
| `tests/test_benchmark_orchestrator.py` | Preserve Python orchestrator contracts and output expectations |
| `docs/rust-coding-std.md` | Standards baseline for the cleanup work |

---

## Task 1: Refactor `benchmark_ltembed` CLI Structure

**Files:**
- Modify: `src/bin/benchmark_ltembed.rs`
- Modify: `src/benchmarking.rs`
- Test: `src/bin/benchmark_ltembed.rs`
- Test: `tests/benchmarking_support_tests.rs`

- [ ] **Step 1: Lock current CLI behavior with tests**

Add or extend unit tests around `parse_args_from`, scenario selection, and each supported mode so the refactor preserves:
- current flag names
- current required and optional argument rules
- current JSON output shape for `warm`, `cold`, `correctness`, and `retrieval`

- [ ] **Step 2: Run targeted tests before refactoring**

Run: `cargo test --bin benchmark_ltembed`

Expected: PASS, establishing the current behavior baseline before structural changes.

- [ ] **Step 3: Introduce typed mode helpers and smaller dispatch functions**

Refactor `src/bin/benchmark_ltembed.rs` so:
- argument parsing remains testable through `parse_args_from`
- mode-specific execution moves into focused helpers such as `run_warm_mode`, `run_cold_mode`, `run_correctness_mode`, and `run_retrieval_mode`
- shared setup like `engine_from_bundle_dir` and scenario resolution stays centralized

Keep the change minimal: do not redesign flags or output payload formats.

- [ ] **Step 4: Re-run targeted CLI tests**

Run: `cargo test --bin benchmark_ltembed`

Expected: PASS, confirming the refactor did not change benchmark binary behavior.

- [ ] **Step 5: Run broader Rust verification for touched benchmark helpers**

Run: `cargo test --test benchmarking_support_tests`

Expected: PASS, confirming scenario helper behavior still matches the benchmark binary contract.

- [ ] **Step 6: Commit the isolated CLI refactor**

```bash
git add src/bin/benchmark_ltembed.rs src/benchmarking.rs tests/benchmarking_support_tests.rs
git commit -m "refactor: split benchmark cli mode handling"
```

---

## Task 2: Decompose `OnnxEngine` and Structure `LTEmbedError`

**Files:**
- Modify: `src/engine.rs`
- Modify: `src/error.rs`
- Modify: `src/traits/tokenizer.rs`
- Test: `src/engine.rs`
- Test: `tests/integration_tests.rs`

- [ ] **Step 1: Lock current engine and error behavior with tests**

Extend focused tests around:
- bundle file validation failures
- build-info parsing failures
- ORT initialization conflict paths
- tensor shape validation failures
- tokenizer load and input-length errors

The goal is to preserve behavior while changing structure, not to expand the feature set.

- [ ] **Step 2: Run targeted engine and integration tests before refactoring**

Run: `cargo test engine::`

Expected: PASS for unit tests in `src/engine.rs`.

Run: `cargo test --test integration_tests`

Expected: PASS for non-skipped integration coverage in the current environment.

- [ ] **Step 3: Split `OnnxEngine` internals into smaller focused helpers**

Refactor `src/engine.rs` to reduce the size and responsibility count of the main impl block. The split should isolate at least:
- bundle path validation
- build-info to `ModelSpec` loading
- ORT initialization and session IO discovery
- batch tensor preparation
- output extraction and embedding postprocessing

Prefer private helpers or private supporting types over broad public API changes.

- [ ] **Step 4: Replace broad stringly variants where structured errors improve clarity**

Update `src/error.rs` so core failure classes that callers and tests care about can be matched more precisely than `ModelLoad(String)` or `Inference(String)` alone. Keep the public surface minimal and avoid speculative enum expansion.

- [ ] **Step 5: Re-run focused Rust verification**

Run: `cargo test engine::`

Expected: PASS.

Run: `cargo test --test integration_tests`

Expected: PASS for non-skipped tests.

- [ ] **Step 6: Run full Rust test suite after the engine refactor**

Run: `cargo test`

Expected: PASS, confirming the engine and error cleanup preserved overall behavior.

- [ ] **Step 7: Commit the isolated engine and error refactor**

```bash
git add src/engine.rs src/error.rs src/traits/tokenizer.rs tests/integration_tests.rs
git commit -m "refactor: split onnx engine responsibilities"
```

---

## Task 3: Decompose `run_embedding_benchmarks.py`

**Files:**
- Modify: `scripts/run_embedding_benchmarks.py`
- Test: `tests/test_benchmark_orchestrator.py`

- [ ] **Step 1: Lock current orchestrator contracts with tests**

Add or extend tests around:
- LTEmbed command construction
- PyTorch command construction
- warm, cold, correctness, and retrieval row generation
- CSV and summary output contracts

Focus on protecting behavior that the CI workflows depend on.

- [ ] **Step 2: Run targeted Python tests before refactoring**

Run: `python3 -m unittest tests.test_benchmark_orchestrator -v`

Expected: PASS, establishing the current orchestrator behavior baseline.

- [ ] **Step 3: Split orchestration into focused helpers**

Refactor `scripts/run_embedding_benchmarks.py` so responsibilities are separated more clearly across helper functions, for example:
- command builders
- subprocess and JSON execution
- row builders
- retrieval metric computation
- summary generation

Keep the change in one file unless a second file clearly reduces coupling without complicating workflow usage.

- [ ] **Step 4: Re-run targeted Python verification**

Run: `python3 -m unittest tests.test_benchmark_orchestrator -v`

Expected: PASS.

- [ ] **Step 5: Run any broader benchmark-script tests that already exist**

Run: `python3 -m unittest`

Expected: PASS for the current Python test suite, or document any unrelated pre-existing failures before proceeding.

- [ ] **Step 6: Commit the isolated Python cleanup**

```bash
git add scripts/run_embedding_benchmarks.py tests/test_benchmark_orchestrator.py
git commit -m "refactor: split benchmark orchestrator helpers"
```

---

## Issue Breakdown

Create and track this work as three separate GitHub issues:

1. `refactor: split benchmark_ltembed CLI parsing and mode dispatch`
2. `refactor: decompose OnnxEngine responsibilities and structure LTEmbedError`
3. `refactor: break run_embedding_benchmarks.py into focused orchestration units`

Each issue should:
- reference this plan document
- call out the affected files
- state that behavior preservation is required
- include the verification commands relevant to that issue

---

## Execution Order

1. `benchmark_ltembed` CLI refactor
2. `OnnxEngine` and `LTEmbedError` refactor
3. Python benchmark orchestrator refactor

This order keeps the highest-risk engine changes separate from the lowest-risk benchmark tooling cleanup and preserves reviewable boundaries.
