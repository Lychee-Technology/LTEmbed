# Issue 30 Scalar Hotspots Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible profiling outputs for LTEmbed scalar hotspots and land one verified attention-path optimization with benchmark evidence.

**Architecture:** Keep the public embedding API unchanged. Extend the existing benchmark/profiling path so a single representative scenario can be profiled repeatedly, then optimize the duplicated masked-softmax scalar loops in the BERT attention path shared by `forward` and `forward_batch`. Preserve numerical behavior with focused unit tests and existing integration coverage.

**Tech Stack:** Rust 2021, `matrixmultiply`, `criterion`, existing `benchmark_ltembed` binary, Python benchmark orchestrator, `cargo test`, `cargo run --release`, macOS ARM64 profiler output with flame graph or equivalent hotspot chart.

---

## File Map

| File | Role |
|---|---|
| `src/models/bert.rs` | Scalar kernels in the attention path; add masked-softmax helper and reuse it from `forward` and `forward_batch` |
| `src/benchmarking.rs` | Scenario helpers; add reusable scenario selection helpers if needed for profiling entrypoints |
| `src/bin/benchmark_ltembed.rs` | Release benchmark/profiling driver; add single-scenario execution and optional profiling-oriented output |
| `tests/benchmarking_support_tests.rs` | Protect scenario selection and latency/report helper behavior |
| `scripts/run_embedding_benchmarks.py` | Keep benchmark orchestration aligned with the updated binary interface |
| `docs/superpowers/specs/2026-03-16-issue-30-scalar-hotspots-design.md` | Approved design reference for this plan |
| `docs/performance/issue-30-profiling.md` | Reproduction notes for macOS ARM64 profiling and expected artifacts |

---

## Chunk 1: Profiling Baseline

### Task 1: Add a scenario-selectable profiling path to the benchmark binary

**Files:**
- Modify: `src/bin/benchmark_ltembed.rs`
- Modify: `src/benchmarking.rs`
- Test: `tests/benchmarking_support_tests.rs`

- [ ] **Step 1.1: Write the failing test for scenario selection helpers**

Add a helper in `src/benchmarking.rs` that resolves either all scenarios or one named scenario, then add tests like:

```rust
#[test]
fn test_selected_scenarios_returns_one_requested_scenario() {
    let selected = selected_scenarios(Some("batch/medium/8")).unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].name, "batch/medium/8");
}

#[test]
fn test_selected_scenarios_rejects_unknown_name() {
    let err = selected_scenarios(Some("missing/scenario")).unwrap_err();
    assert!(err.contains("unknown scenario"));
}
```

- [ ] **Step 1.2: Run the targeted test to verify RED**

Run: `cargo test --test benchmarking_support_tests selected_scenarios`

Expected: FAIL because `selected_scenarios` does not exist yet.

- [ ] **Step 1.3: Write the minimal helper in `src/benchmarking.rs`**

Add:

```rust
pub fn selected_scenarios(name: Option<&str>) -> Result<Vec<&'static BenchmarkScenario>, String> {
    match name {
        Some(name) => scenario_by_name(name)
            .map(|scenario| vec![scenario])
            .ok_or_else(|| format!("unknown scenario: {name}")),
        None => Ok(benchmark_scenarios().iter().collect()),
    }
}
```

- [ ] **Step 1.4: Run the targeted test to verify GREEN**

Run: `cargo test --test benchmarking_support_tests selected_scenarios`

Expected: PASS.

- [ ] **Step 1.5: Add failing tests for the benchmark binary argument behavior**

Inside `src/bin/benchmark_ltembed.rs`, add unit tests around argument parsing and scenario validation, for example:

```rust
#[test]
fn test_parse_args_accepts_optional_scenario_for_warm_mode() {
    let args = parse_args_from([
        "benchmark_ltembed",
        "--mode", "warm",
        "--scenario", "single/medium",
        "--model-dir", "assets",
    ]).unwrap();
    assert_eq!(args.mode, "warm");
    assert_eq!(args.scenario.as_deref(), Some("single/medium"));
}
```

Refactor `parse_args` into `parse_args_from<I, S>(iter: I)` so it can be tested without shelling out.

- [ ] **Step 1.6: Run the targeted binary tests to verify RED**

Run: `cargo test --bin benchmark_ltembed test_parse_args_accepts_optional_scenario_for_warm_mode`

Expected: FAIL because `parse_args_from` and/or the test does not exist yet.

- [ ] **Step 1.7: Implement minimal scenario filtering in `src/bin/benchmark_ltembed.rs`**

Update the `warm` and `correctness` modes to use `selected_scenarios(args.scenario.as_deref())?` instead of always iterating every scenario. Keep `cold` mode requiring exactly one scenario.

- [ ] **Step 1.8: Run the targeted binary tests to verify GREEN**

Run: `cargo test --bin benchmark_ltembed`

Expected: PASS.

- [ ] **Step 1.9: Commit**

```bash
git add src/benchmarking.rs src/bin/benchmark_ltembed.rs tests/benchmarking_support_tests.rs
git commit -m "feat: add scenario-selectable benchmark profiling path"
```

### Task 2: Document reproducible profiling commands and artifacts

**Files:**
- Create: `docs/performance/issue-30-profiling.md`
- Modify: `scripts/run_embedding_benchmarks.py`

- [ ] **Step 2.1: Write the failing integration test for the benchmark script interface**

Add or extend the existing `unittest` suite in `tests/test_benchmark_orchestrator.py` asserting the LTEmbed warm command can target one scenario:

```python
def test_ltembed_warm_command_includes_optional_scenario(self):
    bench = load_module()
    args = type(
        "Args",
        (),
        {
            "model_dir": Path("assets"),
            "warmup": 5,
            "iters": 10,
            "threads": 1,
            "scenario": "single/medium",
        },
    )
    command = bench.ltembed_warm_command(args)
    self.assertIn("--scenario", command)
    self.assertIn("single/medium", command)
```

- [ ] **Step 2.2: Run the targeted Python test to verify RED**

Run: `python -m unittest tests.test_benchmark_orchestrator.BenchmarkOrchestratorTests.test_ltembed_warm_command_includes_optional_scenario`

Expected: FAIL because the helper does not yet pass `--scenario`.

- [ ] **Step 2.3: Implement the minimal script update**

Update `ltembed_warm_command` to append `--scenario <name>` when `args.scenario` is present. Do not change default multi-scenario behavior.

- [ ] **Step 2.4: Write `docs/performance/issue-30-profiling.md`**

Document:

- the representative scenarios to use
- the release command to warm/profile a single scenario
- the macOS ARM64 profiler command sequence
- required artifacts: flame graph or equivalent hotspot chart, top-functions table, before/after benchmark summary

Use concrete commands such as:

```bash
cargo run --release --bin benchmark_ltembed -- \
  --mode warm \
  --scenario single/medium \
  --model-dir assets \
  --warmup 20 \
  --iters 100
```

If the chosen macOS profiler needs a wrapper command, document that exact command in the file rather than burying it in issue notes.

- [ ] **Step 2.5: Run the targeted Python test to verify GREEN**

Run: `python -m unittest tests.test_benchmark_orchestrator.BenchmarkOrchestratorTests.test_ltembed_warm_command_includes_optional_scenario`

Expected: PASS.

- [ ] **Step 2.6: Commit**

```bash
git add scripts/run_embedding_benchmarks.py tests/test_benchmark_orchestrator.py docs/performance/issue-30-profiling.md
git commit -m "docs: add issue 30 profiling workflow and script support"
```

---

## Chunk 2: Masked Softmax Optimization

### Task 3: Add failing tests for masked softmax behavior

**Files:**
- Modify: `src/models/bert.rs`

- [ ] **Step 3.1: Write the failing tests first**

Add unit tests in `src/models/bert.rs` for the new helper, for example:

```rust
#[test]
fn test_masked_softmax_zeroes_masked_positions() {
    let mut scores = vec![1.0f32, 2.0, 3.0, 4.0];
    let mask = vec![1u32, 0, 1, 0];
    masked_softmax(&mut scores, &mask);
    assert_eq!(scores[1], 0.0);
    assert_eq!(scores[3], 0.0);
}

#[test]
fn test_masked_softmax_normalizes_unmasked_positions() {
    let mut scores = vec![1.0f32, 2.0, 3.0];
    let mask = vec![1u32, 0, 1];
    masked_softmax(&mut scores, &mask);
    let sum: f32 = scores.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
}
```

- [ ] **Step 3.2: Run the targeted tests to verify RED**

Run: `cargo test masked_softmax`

Expected: FAIL because `masked_softmax` does not exist yet.

- [ ] **Step 3.3: Implement the minimal helper**

Add a helper near `softmax`:

```rust
fn masked_softmax(x: &mut [f32], attention_mask: &[u32]) {
    debug_assert_eq!(x.len(), attention_mask.len());

    let mut max = f32::NEG_INFINITY;
    for (value, &mask) in x.iter().zip(attention_mask.iter()) {
        if mask != 0 {
            max = max.max(*value);
        }
    }

    let mut sum = 0.0f32;
    for (value, &mask) in x.iter_mut().zip(attention_mask.iter()) {
        if mask == 0 {
            *value = 0.0;
        } else {
            *value = (*value - max).exp();
            sum += *value;
        }
    }

    if sum != 0.0 {
        for (value, &mask) in x.iter_mut().zip(attention_mask.iter()) {
            if mask != 0 {
                *value /= sum;
            }
        }
    }
}
```

Keep the existing plain `softmax` if other tests still exercise it, but prefer migrating call sites to the new helper.

- [ ] **Step 3.4: Run the targeted tests to verify GREEN**

Run: `cargo test masked_softmax`

Expected: PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/models/bert.rs
git commit -m "test: add masked softmax coverage for attention rows"
```

### Task 4: Replace duplicated attention masking loops with the fused helper

**Files:**
- Modify: `src/models/bert.rs`

- [ ] **Step 4.1: Write the failing regression test around attention masking semantics**

Add a unit test that exercises the helper on values that would strongly prefer masked positions if masking were not applied:

```rust
#[test]
fn test_masked_softmax_ignores_large_masked_scores() {
    let mut scores = vec![1.0f32, 1000.0, 2.0];
    let mask = vec![1u32, 0, 1];
    masked_softmax(&mut scores, &mask);
    assert_eq!(scores[1], 0.0);
    assert!(scores[2] > scores[0]);
}
```

- [ ] **Step 4.2: Run the targeted test to verify RED**

Run: `cargo test ignores_large_masked_scores`

Expected: FAIL until the production call sites use the new helper consistently.

- [ ] **Step 4.3: Replace the old mask-plus-softmax loops**

In both `forward` and `forward_batch`:

- delete the column-wise `for (j, &m) in attention_mask...` masking loop
- replace the row softmax call with `masked_softmax(row, attention_mask)` or `masked_softmax(row, batch_mask)`

Use the row slice directly:

```rust
for i in 0..seq_len {
    let row = &mut sc.scores[i * seq_len..(i + 1) * seq_len];
    masked_softmax(row, attention_mask);
}
```

and the equivalent `scores`/`batch_mask` loop in `forward_batch`.

- [ ] **Step 4.4: Run the focused model tests**

Run: `cargo test masked_softmax`

Expected: PASS.

- [ ] **Step 4.5: Run the broader BERT unit test set**

Run: `cargo test bert::tests`

Expected: PASS, including the existing `softmax` and shape tests.

- [ ] **Step 4.6: Commit**

```bash
git add src/models/bert.rs
git commit -m "perf: fuse attention masking with row softmax"
```

---

## Chunk 3: Verification And Evidence

### Task 5: Verify no behavioral regression

**Files:**
- No code changes expected unless a regression is found

- [ ] **Step 5.1: Run the integration test suite**

Run: `cargo test --test integration_tests`

Expected: PASS, or graceful skips if model assets are unavailable.

- [ ] **Step 5.2: Run the benchmark support and Python benchmark tests**

Run: `cargo test --test benchmarking_support_tests`

Run: `python -m unittest tests.test_benchmark_orchestrator`
Run: `pytest tests/test_bench_pytorch.py`

Expected: PASS.

- [ ] **Step 5.3: If any test fails, fix the minimal cause before continuing**

Do not continue to performance claims while correctness is failing.

### Task 6: Capture benchmark and profiling evidence

**Files:**
- Modify if needed: `docs/performance/issue-30-profiling.md`

- [ ] **Step 6.1: Build and run the representative warm benchmark before/after comparison**

Run at minimum:

```bash
cargo run --release --bin benchmark_ltembed -- \
  --mode warm \
  --scenario single/medium \
  --model-dir assets \
  --warmup 20 \
  --iters 100
```

and:

```bash
cargo run --release --bin benchmark_ltembed -- \
  --mode warm \
  --scenario batch/medium/8 \
  --model-dir assets \
  --warmup 20 \
  --iters 100
```

Record `mean_ms`, `median_ms`, and `p95_ms` for the issue notes.

- [ ] **Step 6.2: Capture a macOS ARM64 flame graph or equivalent hotspot chart**

Use the documented profiler command from `docs/performance/issue-30-profiling.md` against the `single/medium` scenario. Save the output artifact in a local working directory for issue attachment.

- [ ] **Step 6.3: Extract the top hotspot summary**

Produce a short table with:

- function name
- self time or sample share
- interpretation

The table must call out whether the chosen optimized hotspot moved down the ranking after the change.

- [ ] **Step 6.4: Update docs if the actual profiler command differs from the planned one**

Keep the committed profiling document aligned with the command sequence that was actually used.

- [ ] **Step 6.5: Commit**

```bash
git add docs/performance/issue-30-profiling.md
git commit -m "docs: capture issue 30 profiling reproduction details"
```

### Task 7: Final verification before completion

**Files:**
- No code changes expected

- [ ] **Step 7.1: Run the full targeted verification set one more time**

Run:

```bash
cargo test bert::tests
cargo test --test benchmarking_support_tests
cargo test --test integration_tests
python -m unittest tests.test_benchmark_orchestrator
```

Expected: PASS.

- [ ] **Step 7.2: Collect the final change summary**

Prepare a concise summary covering:

- profiling path added
- hotspot identified
- optimization shipped
- benchmark delta
- any environment limitations such as missing Linux ARM64 profiling

- [ ] **Step 7.3: Final commit if needed**

Only create an additional commit if verification required a code or doc fix.
