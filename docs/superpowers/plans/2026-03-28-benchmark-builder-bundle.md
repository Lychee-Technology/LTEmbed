# Benchmark Builder Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `benchmark-arm64` consume a pinned `minimal-ort-builder` release bundle for LTEmbed while keeping HuggingFace weights for the PyTorch reference path.

**Architecture:** The workflow will prepare two separate benchmark inputs: `ort_bundle/` from the pinned builder release and `MODEL_DIR` from HuggingFace. The Python harness will route LTEmbed invocations to `--ort-bundle-dir` and PyTorch invocations to `--model-dir`, and the Rust benchmark binary will initialize LTEmbed from the bundle directory.

**Tech Stack:** GitHub Actions, Python 3.11, Rust, `gh` CLI, `huggingface_hub`

---

### Task 1: Lock the benchmark input contract with tests

**Files:**
- Modify: `tests/test_benchmark_orchestrator.py`
- Test: `tests/test_benchmark_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_ltembed_commands_use_ort_bundle_contract(self):
    ...
    self.assertIn("--ort-bundle-dir", command)
    self.assertNotIn("--model-dir", ltembed_slice)

def test_pytorch_commands_keep_model_dir_contract(self):
    ...
    self.assertIn("--model-name-or-path", command)

def test_benchmark_workflow_downloads_builder_bundle_and_hf_weights(self):
    ...
    self.assertIn("minimal-ort-builder", workflow)
    self.assertIn("jinaai__jina-embeddings-v5-text-nano-retrieval_q4f16_linux-arm64.tar.gz", workflow)
    self.assertIn("snapshot_download(", workflow)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_benchmark_orchestrator -v`
Expected: FAIL because the LTEmbed command builder and workflow still use the current `model_dir` contract only.

- [ ] **Step 3: Write minimal implementation**

```text
No production code in this task. Only add the regression tests.
```

- [ ] **Step 4: Run test to verify it still fails for the expected reason**

Run: `python3 -m unittest tests.test_benchmark_orchestrator -v`
Expected: FAIL with assertions showing the missing builder bundle / `--ort-bundle-dir` wiring.

- [ ] **Step 5: Commit**

```bash
git add tests/test_benchmark_orchestrator.py
git commit -m "test: lock benchmark bundle contracts"
```

### Task 2: Update the workflow to prepare both bundle and HF inputs

**Files:**
- Modify: `.github/workflows/benchmark-arm64.yml`
- Test: `tests/test_benchmark_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
def test_benchmark_workflow_downloads_builder_bundle_and_hf_weights(self):
    ...
    self.assertIn("gh release download", workflow)
    self.assertIn("--ort-bundle-dir", workflow)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_benchmark_orchestrator -v`
Expected: FAIL because the workflow does not yet download or extract the builder bundle.

- [ ] **Step 3: Write minimal implementation**

```yaml
env:
  BUILDER_REPO: Lychee-Technology/minimal-ort-builder
  BUILDER_RELEASE_TAG: v1.0.9
  BUILDER_ASSET_NAME: jinaai__jina-embeddings-v5-text-nano-retrieval_q4f16_linux-arm64.tar.gz
  ORT_BUNDLE_DIR: ${{ github.workspace }}/ort_bundle

- name: Download ORT Bundle
  run: |
    mkdir -p "$ORT_BUNDLE_DIR"
    gh release download "$BUILDER_RELEASE_TAG" \
      --repo "$BUILDER_REPO" \
      --pattern "$BUILDER_ASSET_NAME" \
      --dir "$RUNNER_TEMP/ort-bundle-download"
    tar -xzf "$RUNNER_TEMP/ort-bundle-download/$BUILDER_ASSET_NAME" -C "$GITHUB_WORKSPACE"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_benchmark_orchestrator -v`
Expected: PASS for workflow contract tests.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/benchmark-arm64.yml tests/test_benchmark_orchestrator.py
git commit -m "ci: fetch benchmark ort bundle from builder release"
```

### Task 3: Route LTEmbed to the bundle directory

**Files:**
- Modify: `scripts/run_embedding_benchmarks.py`
- Modify: `src/bin/benchmark_ltembed.rs`
- Test: `tests/test_benchmark_orchestrator.py`
- Test: `src/bin/benchmark_ltembed.rs`

- [ ] **Step 1: Write the failing tests**

```python
def test_ltembed_commands_use_ort_bundle_contract(self):
    ...
```

```rust
#[test]
fn test_parse_args_accepts_ort_bundle_dir_for_warm_mode() {
    let args = parse_args_from([
        "benchmark_ltembed",
        "--mode",
        "warm",
        "--scenario",
        "single/medium",
        "--ort-bundle-dir",
        "ort_bundle",
    ])
    .unwrap();

    assert_eq!(args.ort_bundle_dir, PathBuf::from("ort_bundle"));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_benchmark_orchestrator -v`
Expected: FAIL because LTEmbed commands still use `--model-dir`.

Run: `cargo test --bin benchmark_ltembed`
Expected: FAIL because the parser still expects `--model-dir`.

- [ ] **Step 3: Write minimal implementation**

```python
def ltembed_warm_command(args):
    return [
        "cargo", "run", "--quiet", "--release",
        "--bin", "benchmark_ltembed", "--",
        "--mode", "warm",
        "--ort-bundle-dir", str(args.ort_bundle_dir),
        "--output-dimension", str(args.output_dimension),
        "--l2-normalize", "true" if args.l2_normalize else "false",
        ...
    ]
```

```rust
struct Args {
    ort_bundle_dir: PathBuf,
    output_dimension: usize,
    l2_normalize: bool,
    ...
}

let engine = OnnxEngine::from_bundle_dir(
    Path::new(&args.ort_bundle_dir),
    OnnxEngineConfig {
        output_dimension: args.output_dimension,
        l2_normalize: args.l2_normalize,
    },
)?;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_benchmark_orchestrator -v`
Expected: PASS

Run: `cargo test --bin benchmark_ltembed`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_embedding_benchmarks.py src/bin/benchmark_ltembed.rs tests/test_benchmark_orchestrator.py
git commit -m "bench: consume builder ort bundle for ltembed benchmarks"
```

### Task 4: Final verification

**Files:**
- Modify: `tests/test_benchmark_orchestrator.py`
- Modify: `.github/workflows/benchmark-arm64.yml`
- Modify: `scripts/run_embedding_benchmarks.py`
- Modify: `src/bin/benchmark_ltembed.rs`

- [ ] **Step 1: Run targeted verification**

Run: `python3 -m unittest tests.test_benchmark_orchestrator`
Expected: PASS

Run: `cargo test --bin benchmark_ltembed`
Expected: PASS

- [ ] **Step 2: Review diff for unrelated changes**

Run: `git status --short`
Expected: Only the benchmark workflow, harness, runner, plan doc, and test files are modified. Do not touch unrelated untracked docs.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/benchmark-arm64.yml scripts/run_embedding_benchmarks.py src/bin/benchmark_ltembed.rs tests/test_benchmark_orchestrator.py
git commit -m "ci: align benchmark inputs with builder ort bundle"
```
