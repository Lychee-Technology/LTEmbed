# CN/EN-only Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every benchmark measurement derive from `tests/CN_EN_Data.csv` — per-language latency scenarios, correctness derived from the retrieval embeddings, PyTorch retrieval-only, quality gate on mean cosine — and drop the jane-austen corpus.

**Architecture:** One dataset. The generator emits the retrieval eval plus a pre-resolved two-scenario latency fixture (`single/zh`, `single/en`). Per quant, `ltembed` runs warm/cold (latency) + retrieval; the orchestrator derives both retrieval quality (`both@3`/recall) and fidelity (cosine vs FP32) from the single retrieval embedding pass against a retrieval-only PyTorch reference.

**Tech Stack:** Python 3.13 (stdlib + numpy), Rust 1.94 (llama.cpp FFI), GitHub Actions.

## Global Constraints

- Scenario names are the cross-process contract; they must match byte-for-byte across `scripts/run_embedding_benchmarks.py`, `src/benchmarking.rs`, and the generated fixture: exactly `single/zh` and `single/en`.
- Embeddings are L2-normalized by default; cosine == dot after normalization.
- The orchestrator (`run_embedding_benchmarks.py`) stays pure-stdlib except numpy, which is imported lazily inside functions that need it.
- Rust cannot link test binaries locally (macOS `stdc++`); the local Rust gate is `cargo clippy --all-targets -- -D warnings` with `STATIC_LLAMA_DIR="$(pwd)/.llama-artifacts/extracted"`. Full `cargo test` runs on CI (Linux).
- Run all commands from the worktree root.

---

### Task 1: Generator emits the CN/EN latency fixture

**Files:**
- Modify: `scripts/build_cn_en_retrieval_cases.py`
- Test: `tests/test_build_cn_en_retrieval_cases.py` (create if absent)

**Interfaces:**
- Produces: `build_fixture(pairs: list[tuple[str,str]]) -> dict` returning `{"scenarios": {"single/zh": [{"kind":"query","text":<zh>}], "single/en": [{"kind":"query","text":<en>}]}}`; `pick_representative(pairs) -> tuple[str,str]` (median English length); new CLI flag `--fixture-output PATH`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_cn_en_retrieval_cases.py`:

```python
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_cn_en_retrieval_cases.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_cn_en", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_fixture_has_two_language_scenarios():
    gen = load_module()
    pairs = [("短", "short"), ("中等长度的句子", "a medium length sentence"), ("最长的一句话在这里", "the longest sentence here")]
    fixture = gen.build_fixture(pairs)
    scenarios = fixture["scenarios"]
    assert set(scenarios.keys()) == {"single/zh", "single/en"}
    assert len(scenarios["single/zh"]) == 1
    assert scenarios["single/zh"][0]["kind"] == "query"
    assert len(scenarios["single/en"]) == 1
    # zh and en come from the SAME representative pair
    zh = scenarios["single/zh"][0]["text"]
    en = scenarios["single/en"][0]["text"]
    assert (zh, en) == gen.pick_representative(pairs)


def test_pick_representative_is_median_english_length_and_deterministic():
    gen = load_module()
    pairs = [("a", "x"), ("b", "yyy"), ("c", "zz")]
    # sorted by len(en): "x"(1), "zz"(2), "yyy"(3) -> median index 1 -> ("c","zz")
    assert gen.pick_representative(pairs) == ("c", "zz")
    assert gen.pick_representative(pairs) == gen.pick_representative(pairs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_build_cn_en_retrieval_cases.py -q`
Expected: FAIL with `AttributeError: module ... has no attribute 'build_fixture'`

- [ ] **Step 3: Add `pick_representative` and `build_fixture`**

In `scripts/build_cn_en_retrieval_cases.py`, add after `build_case`:

```python
def pick_representative(pairs: list[tuple[str, str]]) -> tuple[str, str]:
    """The median-English-length pair — a representative, non-degenerate sentence."""
    ranked = sorted(pairs, key=lambda pair: len(pair[1]))
    return ranked[len(ranked) // 2]


def build_fixture(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    zh, en = pick_representative(pairs)
    return {
        "scenarios": {
            "single/zh": [{"kind": "query", "text": zh}],
            "single/en": [{"kind": "query", "text": en}],
        }
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_build_cn_en_retrieval_cases.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Wire `--fixture-output` into the CLI**

In `parse_args`, add:

```python
    parser.add_argument("--fixture-output", type=Path, default=None,
                        help="Optional path to also write the single/zh + single/en latency fixture.")
```

In `main`, after writing the retrieval payload, add (using the already-loaded sampled `pairs`):

```python
    if args.fixture_output is not None:
        pairs = sample_pairs(load_pairs(args.csv), args.num_pairs)
        args.fixture_output.parent.mkdir(parents=True, exist_ok=True)
        args.fixture_output.write_text(
            json.dumps(build_fixture(pairs), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote fixture {args.fixture_output}")
```

- [ ] **Step 6: Verify end-to-end + commit**

Run: `python3 scripts/build_cn_en_retrieval_cases.py --output /tmp/ret.json --fixture-output /tmp/fix.json --num-pairs 500 && python3 -c "import json; f=json.load(open('/tmp/fix.json')); print(list(f['scenarios'])); print(f['scenarios']['single/zh'][0]['text']); print(f['scenarios']['single/en'][0]['text'])"`
Expected: prints `['single/zh', 'single/en']` and a Chinese then an English sentence.

```bash
git add scripts/build_cn_en_retrieval_cases.py tests/test_build_cn_en_retrieval_cases.py
git commit -m "feat: generate single/zh + single/en latency fixture from CN_EN"
```

---

### Task 2: Orchestrator scenarios → single/zh, single/en; fixture pass-through

**Files:**
- Modify: `scripts/run_embedding_benchmarks.py` (`SCENARIOS`, remove `resolve_fixture`/`load_corpus_texts`, simplify `resolve_fixture_if_present`)
- Test: `tests/test_benchmark_orchestrator.py`

**Interfaces:**
- Produces: `SCENARIOS` = `[Scenario("single/zh",1,"zh"), Scenario("single/en",1,"en")]`; `resolve_fixture_if_present(args)` sets `args.resolved_fixture_path = args.fixture_path` verbatim (the fixture is already resolved). `load_corpus_texts`/`resolve_fixture` are removed.

- [ ] **Step 1: Update the scenario test (failing)**

In `tests/test_benchmark_orchestrator.py`, replace `test_scenarios_are_single_only` body with:

```python
    def test_scenarios_are_single_zh_and_en(self):
        bench = load_module()
        self.assertEqual([s.name for s in bench.SCENARIOS], ["single/zh", "single/en"])
        self.assertTrue(all(s.batch_size == 1 for s in bench.SCENARIOS))
        self.assertEqual(bench.scenario_from_name("single/zh").text_profile, "zh")
```

Delete `test_load_corpus_texts_sorts_by_length_and_skips_empty` and `test_resolve_fixture_selects_single_texts_and_kinds` (the functions they cover are being removed).

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_benchmark_orchestrator.py::CommandBuilderTests::test_scenarios_are_single_zh_and_en -q`
Expected: FAIL (`SCENARIOS` still contains `single/short` …)

- [ ] **Step 3: Replace `SCENARIOS`**

In `scripts/run_embedding_benchmarks.py`:

```python
SCENARIOS = [
    Scenario(name="single/zh", batch_size=1, text_profile="zh"),
    Scenario(name="single/en", batch_size=1, text_profile="en"),
]
```

- [ ] **Step 4: Remove jane-austen resolution**

Delete the `load_corpus_texts` and `resolve_fixture` functions entirely. Replace `resolve_fixture_if_present` with:

```python
def resolve_fixture_if_present(args: argparse.Namespace) -> None:
    """Point the runners at the already-resolved CN/EN latency fixture.

    The generator writes the fixture in the runners' resolved format
    (``{"scenarios": {name: [{"kind","text"}]}}``), so there is nothing to resolve here.
    """
    if getattr(args, "fixture_path", None):
        args.resolved_fixture_path = args.fixture_path
```

Update the `--fixture-path` help text to: `"Pre-resolved CN/EN latency fixture (scenarios -> texts) produced by build_cn_en_retrieval_cases.py --fixture-output."`

- [ ] **Step 5: Run tests to verify pass**

Run: `python3 -m pytest tests/test_benchmark_orchestrator.py -q`
Expected: PASS (the golden `_run` tests use scenario names via mocks; update any that referenced `single/long`/`single/short` in this file to `single/zh`/`single/en` if they now fail — the mock scenario lists in `RunTests` should read `["single/zh", "single/en"]`).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_embedding_benchmarks.py tests/test_benchmark_orchestrator.py
git commit -m "feat: orchestrator uses single/zh + single/en; fixture pass-through"
```

---

### Task 3: Derive correctness from retrieval; PyTorch reference retrieval-only

**Files:**
- Modify: `scripts/run_embedding_benchmarks.py` (add `derive_correctness_rows`, remove `collect_correctness_rows`, `--include-correctness`; `_emit_reference` retrieval-only; `summary_lines` version source; `_run` flow)
- Test: `tests/test_benchmark_orchestrator.py`

**Interfaces:**
- Consumes: retrieval payloads produced by `collect_retrieval_eval_rows` — each `payload["results"][*]["documents"]` is `[{"id","embedding"}]`.
- Produces: `derive_correctness_rows(*, ctx, args, ltembed_retrieval, reference_retrieval) -> list[dict]` — one correctness row per retrieval document, `implementation="ltembed"`, `mode="correctness"`, `scenario="cn-en/zh"|"cn-en/en"` (by id suffix), `cosine_similarity_vs_pytorch` set, `status` gated by `args.correctness_threshold`. Reference JSON shape becomes `{"retrieval": <pytorch retrieval payload>}` (no `correctness` key).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_benchmark_orchestrator.py` in `ReferenceModeTests`:

```python
    def test_derive_correctness_rows_from_retrieval(self):
        bench = load_module()
        ltembed = {"results": [{"dataset_name": "cn-en-crosslingual-v1", "documents": [
            {"id": "pair_0_zh", "embedding": [1.0, 0.0]},
            {"id": "pair_0_en", "embedding": [0.0, 1.0]},
        ]}]}
        reference = {"results": [{"dataset_name": "cn-en-crosslingual-v1", "documents": [
            {"id": "pair_0_zh", "embedding": [1.0, 0.0]},   # identical -> cos 1.0
            {"id": "pair_0_en", "embedding": [1.0, 0.0]},   # orthogonal -> cos 0.0
        ]}]}
        ctx = bench.RunContext(run_id="r", timestamp_utc="t", model_id="m",
                               model_source="hf", git_revision="sha", host={
                                   "host_os": "linux", "host_arch": "arm64",
                                   "cpu_model": "c", "runner_labels": ""})
        args = SimpleNamespace(threads=1, correctness_threshold=0.98)
        rows = bench.derive_correctness_rows(ctx=ctx, args=args,
                                             ltembed_retrieval=ltembed, reference_retrieval=reference)
        by_scenario = {r["scenario"]: r for r in rows}
        self.assertEqual(len(rows), 2)
        self.assertEqual(by_scenario["cn-en/zh"]["cosine_similarity_vs_pytorch"], "1.000000")
        self.assertEqual(by_scenario["cn-en/zh"]["status"], "pass")
        self.assertEqual(by_scenario["cn-en/en"]["cosine_similarity_vs_pytorch"], "0.000000")
        self.assertEqual(by_scenario["cn-en/en"]["status"], "fail")
        self.assertTrue(all(r["implementation"] == "ltembed" and r["mode"] == "correctness" for r in rows))
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_benchmark_orchestrator.py::ReferenceModeTests::test_derive_correctness_rows_from_retrieval -q`
Expected: FAIL (`derive_correctness_rows` undefined)

- [ ] **Step 3: Implement `derive_correctness_rows`**

In `scripts/run_embedding_benchmarks.py`, add (and delete the old `collect_correctness_rows` function):

```python
def _language_from_doc_id(doc_id: str) -> str:
    return "cn-en/zh" if doc_id.endswith("_zh") else "cn-en/en"


def derive_correctness_rows(
    *,
    ctx: RunContext,
    args: argparse.Namespace,
    ltembed_retrieval: dict[str, Any],
    reference_retrieval: dict[str, Any],
) -> list[dict[str, str]]:
    """Cosine ltembed-vs-FP32 per retrieval document — fidelity for free from the retrieval pass."""
    reference_docs: dict[str, list[float]] = {}
    for result in reference_retrieval["results"]:
        for doc in result["documents"]:
            reference_docs[str(doc["id"])] = doc["embedding"]

    rows: list[dict[str, str]] = []
    version = resolved_implementation_version("ltembed", {}, ctx.git_revision)
    for result in ltembed_retrieval["results"]:
        for doc in result["documents"]:
            doc_id = str(doc["id"])
            similarity = cosine_similarity(doc["embedding"], reference_docs[doc_id])
            scenario = Scenario(name=_language_from_doc_id(doc_id), batch_size=1, text_profile="correctness")
            base_fields = base_row_fields(
                ctx=ctx, implementation="ltembed", implementation_version=version,
                scenario=scenario, mode="correctness", threads=args.threads,
                warmup_iters=0, timed_iters=0,
            )
            rows.append(build_correctness_row(
                base_fields=base_fields, cosine_similarity=similarity,
                threshold=args.correctness_threshold,
            ))
    return rows
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_benchmark_orchestrator.py::ReferenceModeTests::test_derive_correctness_rows_from_retrieval -q`
Expected: PASS

- [ ] **Step 5: Rewire `_run`, `_emit_reference`, `summary_lines`, and flags**

In `_run`, remove the correctness block and derive after retrieval:

```python
    retrieval_payloads = None
    if args.include_retrieval_eval:
        retrieval_rows, retrieval_payloads = collect_retrieval_eval_rows(
            args=args, ctx=ctx, implementations=embedding_impls, reference=reference
        )
        rows.extend(retrieval_rows)
        rows.extend(derive_correctness_rows(
            ctx=ctx, args=args,
            ltembed_retrieval=retrieval_payloads["ltembed"],
            reference_retrieval=retrieval_payloads["pytorch"],
        ))
```

Delete the `correctness_payloads` block and the `if args.include_correctness:` call. In the `summary_lines(...)` call, drop `correctness_payloads=...`.

In `parse_args`, delete the `--include-correctness` argument.

In `_emit_reference`, make it retrieval-only:

```python
def _emit_reference(*, args: argparse.Namespace) -> int:
    """Run only PyTorch retrieval and write an embeddings-only reference."""
    resolve_fixture_if_present(args)
    reference = {
        "retrieval": run_json_command(
            build_benchmark_command("pytorch", "retrieval", args), "pytorch retrieval"
        ),
    }
    args.emit_reference.parent.mkdir(parents=True, exist_ok=True)
    args.emit_reference.write_text(json.dumps(reference), encoding="utf-8")
    print(f"wrote PyTorch retrieval reference to {args.emit_reference}")
    return 0
```

In `summary_lines`, change the signature to drop `correctness_payloads` and source the PyTorch version from `reference.get("retrieval")`:

```python
    pytorch_payload = warm_payloads.get("pytorch")
    if pytorch_payload is None and reference is not None:
        pytorch_payload = reference.get("retrieval")
    pytorch_payload = pytorch_payload or {}
```

Remove the `if correctness_payloads is not None: lines.append("correctness=enabled")` line; add `lines.append("correctness=derived")` after the retrieval-eval line when `retrieval_payloads is not None`.

- [ ] **Step 6: Update the golden `_run` tests**

In `tests/test_benchmark_orchestrator.py` `RunTests`, the mock `mock_run` must no longer be called with a `correctness` label. Update `test_standalone_run_runs_both_impls` and `test_reference_mode_skips_pytorch_latency`: remove the `" correctness"` branch usage, and change correctness assertions to derived rows — after `_run`, `modes.count("correctness")` equals the number of retrieval documents (2 in the fixtures used), with `implementation == "ltembed"`. Ensure the retrieval mock payload documents use ids ending `_zh`/`_en` so the derived rows get scenarios `cn-en/zh`/`cn-en/en`. Update the summary assertion to expect `correctness=derived`.

- [ ] **Step 7: Run the full orchestrator suite**

Run: `python3 -m pytest tests/test_benchmark_orchestrator.py -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add scripts/run_embedding_benchmarks.py tests/test_benchmark_orchestrator.py
git commit -m "feat: derive correctness from retrieval; pytorch reference retrieval-only"
```

---

### Task 4: Slim `bench_pytorch.py` to retrieval-only

**Files:**
- Modify: `scripts/bench_pytorch.py` (remove `SCENARIOS`, `SHORT`/`MEDIUM`/`LONG`, `apply_fixture`, `warm_payload`, `cold_payload`, `correctness_payload`, warm/cold measurement helpers; keep retrieval)
- Test: `tests/test_bench_pytorch.py`

**Interfaces:**
- Produces: `main()` handles only `--mode retrieval`; any other mode raises `SystemExit`/`ValueError`. `progress_label`, `compute_stats`, `load_model`, `embed_texts`, and the retrieval path remain.

- [ ] **Step 1: Replace the scenario test with a retrieval-only test (failing)**

In `tests/test_bench_pytorch.py`, delete `test_scenarios_are_representative_set`, `test_apply_fixture_overrides_scenario_texts`, and the `progress_label` test's scenario name; add:

```python
    def test_scenarios_and_fixture_machinery_removed(self):
        bench = load_module()
        self.assertFalse(hasattr(bench, "SCENARIOS"))
        self.assertFalse(hasattr(bench, "apply_fixture"))
        self.assertFalse(hasattr(bench, "warm_payload"))

    def test_progress_label_includes_mode_scenario_and_state(self):
        bench = load_module()
        self.assertEqual(bench.progress_label("retrieval", "cn-en-crosslingual-v1", "start"),
                         "retrieval cn-en-crosslingual-v1 start")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_bench_pytorch.py::BenchPyTorchTests::test_scenarios_and_fixture_machinery_removed -q`
Expected: FAIL (`SCENARIOS` still defined)

- [ ] **Step 3: Remove the warm/cold/correctness machinery**

In `scripts/bench_pytorch.py`: delete `SHORT`, `MEDIUM`, `LONG`, `SCENARIOS`, `apply_fixture`, the warm/cold measurement helpers (`measure_warm_stats`, `measure_cold_stats`), `warm_payload`, `cold_payload`, `correctness_payload`. In `main`, keep only the retrieval branch:

```python
def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    if args.mode != "retrieval":
        raise SystemExit(f"bench_pytorch now supports only --mode retrieval (got {args.mode})")
    payload = retrieval_payload(args)
    print(json.dumps(payload))
```

Leave `parse_args` accepting the existing flags (`--warmup`, `--iters`, `--scenario`, `--fixture-path`) so the orchestrator's shared args remain compatible; they are simply unused. Remove the `if args.fixture_path is not None: apply_fixture(...)` call.

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m pytest tests/test_bench_pytorch.py -q`
Expected: PASS (remaining tests cover `compute_stats`, `load_model`, retrieval, `progress_label`)

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_pytorch.py tests/test_bench_pytorch.py
git commit -m "refactor: bench_pytorch is retrieval-only (drops scenario machinery)"
```

---

### Task 5: Rust scenarios → single/zh, single/en

**Files:**
- Modify: `src/benchmarking.rs` (`BENCHMARK_SCENARIOS`, text consts, `scenario_inputs`)
- Test: `tests/benchmarking_support_tests.rs`

**Interfaces:**
- Produces: `BENCHMARK_SCENARIOS` = two entries `single/zh` (profile `zh`), `single/en` (profile `en`); `ZH_TEXT`/`EN_TEXT` consts; `scenario_inputs` returns `query_input(ZH_TEXT)` / `query_input(EN_TEXT)`.

- [ ] **Step 1: Update the Rust support test (failing)**

In `tests/benchmarking_support_tests.rs`, replace `test_benchmark_scenarios_are_representative_set` and `test_selected_scenarios_returns_requested_scenario` and `test_single_scenarios_carry_expected_kinds` with:

```rust
#[test]
fn test_benchmark_scenarios_are_single_zh_and_en() {
    let names: Vec<_> = benchmark_scenarios().iter().map(|s| s.name).collect();
    assert_eq!(names, vec!["single/zh", "single/en"]);
    assert_eq!(scenario_by_name("single/zh").unwrap().batch_size, 1);
    assert_eq!(scenario_by_name("single/en").unwrap().text_profile, "en");
    assert!(scenario_by_name("missing/scenario").is_none());
}

#[test]
fn test_selected_scenarios_returns_requested_scenario() {
    let selected = selected_scenarios(Some("single/zh")).unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].name, "single/zh");
}

#[test]
fn test_single_scenarios_carry_query_kind() {
    let zh = scenario_inputs(scenario_by_name("single/zh").expect("exists"));
    assert_eq!(zh.len(), 1);
    assert_eq!(zh[0].kind, EmbeddingInputKind::Query);
    let en = scenario_inputs(scenario_by_name("single/en").expect("exists"));
    assert_eq!(en.len(), 1);
    assert_eq!(en[0].kind, EmbeddingInputKind::Query);
}
```

Also update `test_scenario_token_lengths_follow_tokenizer_outputs` to use `"single/zh"` instead of `"single/medium"`.

- [ ] **Step 2: Replace scenarios + inputs in `src/benchmarking.rs`**

Replace the text consts and `BENCHMARK_SCENARIOS`:

```rust
pub const ZH_TEXT: &str = "他感冒了";
pub const EN_TEXT: &str = "He caught a cold.";
pub const BENCHMARK_MAX_LENGTH: usize = 8192;

const BENCHMARK_SCENARIOS: [BenchmarkScenario; 2] = [
    BenchmarkScenario { name: "single/zh", batch_size: 1, text_profile: "zh" },
    BenchmarkScenario { name: "single/en", batch_size: 1, text_profile: "en" },
];
```

Replace `scenario_inputs`:

```rust
pub fn scenario_inputs(scenario: &BenchmarkScenario) -> Vec<BenchmarkInput> {
    match scenario.name {
        "single/zh" => vec![query_input(ZH_TEXT)],
        "single/en" => vec![query_input(EN_TEXT)],
        _ => Vec::new(),
    }
}
```

Delete the now-unused `SHORT_TEXT`, `MEDIUM_TEXT`, `long_text`, and `document_input` **only if** they have no remaining references (check with grep in Step 3).

- [ ] **Step 3: Fix leftover references + lint**

Run: `grep -rn "SHORT_TEXT\|MEDIUM_TEXT\|long_text\|document_input\|single/short\|single/medium\|single/long" src/`
Expected: no hits in `src/benchmarking.rs`. If `document_input`/`long_text` are referenced elsewhere (e.g. `src/bin/benchmark_ltembed.rs`), keep them; otherwise delete to avoid dead-code warnings.

Run: `STATIC_LLAMA_DIR="$(pwd)/.llama-artifacts/extracted" cargo clippy --all-targets -- -D warnings`
Expected: `Finished` with no warnings.

- [ ] **Step 4: Commit**

```bash
git add src/benchmarking.rs tests/benchmarking_support_tests.rs
git commit -m "feat: ltembed benchmark scenarios are single/zh + single/en"
```

---

### Task 6: Remove the correctness mode from the ltembed binary

**Files:**
- Modify: `src/bin/benchmark_ltembed.rs` (drop `correctness` mode dispatch + `run_correctness_mode`; fix inline tests)
- Test: inline `#[cfg(test)]` in the same file

**Interfaces:**
- Consumes: nothing new. The orchestrator no longer invokes `--mode correctness` for ltembed.
- Produces: `main` handles `warm`, `cold`, `retrieval`; `correctness` becomes an unknown mode error.

- [ ] **Step 1: Update inline tests (failing at compile)**

In `src/bin/benchmark_ltembed.rs`, change `test_progress_label_includes_mode_scenario_and_state` to use `"single/zh"`:

```rust
        assert_eq!(progress_label("warm", "single/zh", "start"), "warm single/zh start");
```

In `test_scenario_inputs_resolved_uses_fixture_when_present`, change the fixture keys to `single/zh` / `single/en` and the calls accordingly:

```rust
        let fixture: ResolvedFixture = serde_json::from_str(
            r#"{ "scenarios": {
                "single/zh": [{"kind": "query", "text": "他感冒了"}],
                "single/en": [{"kind": "document", "text": "He caught a cold."}]
            } }"#,
        ).unwrap();
        let zh = scenario_inputs_resolved("single/zh", Some(&fixture)).unwrap();
        assert_eq!(zh.len(), 1);
        assert_eq!(zh[0].kind, EmbeddingInputKind::Query);
        let en = scenario_inputs_resolved("single/en", Some(&fixture)).unwrap();
        assert_eq!(en.len(), 1);
        assert_eq!(en[0].kind, EmbeddingInputKind::Document);
```

In `test_scenario_inputs_resolved_falls_back_to_builtin_without_fixture`, change `"single/short"` to `"single/zh"`. Delete any test that invokes `run_correctness_mode` or asserts a correctness payload.

- [ ] **Step 2: Remove the correctness mode**

Delete `run_correctness_mode` and its match arm in `main` (the `Mode::Correctness` / `"correctness"` dispatch). If `Mode` is an enum with a `Correctness` variant, remove that variant and any exhaustiveness fallout. Keep `warm`, `cold`, `retrieval`.

- [ ] **Step 3: Lint (local gate)**

Run: `STATIC_LLAMA_DIR="$(pwd)/.llama-artifacts/extracted" cargo clippy --all-targets -- -D warnings`
Expected: `Finished` with no warnings. (Full `cargo test` runs on CI.)

- [ ] **Step 4: Commit**

```bash
git add src/bin/benchmark_ltembed.rs
git commit -m "refactor: drop unused correctness mode from ltembed binary"
```

---

### Task 7: Report gate on mean cosine

**Files:**
- Modify: `scripts/render_benchmark_report.py` (`recommend`, gate wording)
- Test: `tests/test_render_benchmark_report.py`

**Interfaces:**
- Produces: `recommend()` selects the smallest GGUF with `mean_cosine >= QUALITY_GATE` (was `min_cosine`).

- [ ] **Step 1: Write the failing test**

In `tests/test_render_benchmark_report.py`, add a case where a small quant has a low-outlier min but a high mean and should now be recommended:

```python
    def test_gate_uses_mean_not_min(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "bench-results"
            base.mkdir()
            # IQ4_NL: one low outlier (0.90) but mean 0.985 -> passes mean gate 0.98
            _write_quant(base, "IQ4_NL", 190_000_000, warm=8.0, cosines=[0.90, 0.99, 0.99, 0.99, 0.995])
            _write_quant(base, "Q5_K_M", 230_000_000, warm=9.5, cosines=[0.99, 0.99])
            results = mod.collect_results(base)
            recommended, _ = mod.recommend(results)
            self.assertEqual(recommended, "IQ4_NL")  # smallest with mean >= gate
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_render_benchmark_report.py::RenderBenchmarkReportTests::test_gate_uses_mean_not_min -q`
Expected: FAIL (min gate rejects IQ4_NL because min 0.90 < 0.98)

- [ ] **Step 3: Gate on mean in `recommend`**

In `scripts/render_benchmark_report.py`, change the gate in `recommend`:

```python
    gated = [
        r for r in results if r.get("mean_cosine") is not None and r["mean_cosine"] >= QUALITY_GATE
    ]
    if gated:
        best = min(gated, key=lambda r: (r.get("gguf_size_bytes") or float("inf"), r.get("warm_ms") or float("inf")))
        reason = (
            f"smallest GGUF whose mean cosine vs FP32 stays ≥ {QUALITY_GATE:.2f} "
            f"(mean_cosine={_fmt('cos', best['mean_cosine'])}, size={_fmt('mb', best['size_mb'])} MB)"
        )
        return best["quant"], reason
```

Update the report footer/gate wording in `build_report` from `worst-case cosine` to `mean cosine`.

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m pytest tests/test_render_benchmark_report.py -q`
Expected: PASS (existing `test_recommends_smallest_quant_passing_quality_gate` still passes — its Q5_K_M mean 0.989 clears the gate and is smallest passing).

- [ ] **Step 5: Commit**

```bash
git add scripts/render_benchmark_report.py tests/test_render_benchmark_report.py
git commit -m "feat: report quality gate uses mean cosine"
```

---

### Task 8: CI — one dataset, retrieval-only reference

**Files:**
- Modify: `.github/workflows/benchmark-arm64.yml`
- Test: `tests/test_benchmark_orchestrator.py` (`WorkflowTests`)

**Interfaces:**
- Consumes: `build_cn_en_retrieval_cases.py --fixture-output`, orchestrator `--emit-reference`/`--reference-path`/`--fixture-path`.

- [ ] **Step 1: Update workflow assertions (failing)**

In `tests/test_benchmark_orchestrator.py` `WorkflowTests`, update/add:

```python
    def test_reference_job_generates_both_files_and_no_jane_austen(self):
        workflow = self._workflow()
        self.assertIn("--fixture-output reference/cn_en_fixture.json", workflow)
        self.assertIn("--emit-reference reference/reference.json", workflow)
        self.assertNotIn("jane-austen", workflow)

    def test_matrix_jobs_consume_fixture_and_reference(self):
        workflow = self._workflow()
        self.assertIn("--fixture-path reference/cn_en_fixture.json", workflow)
        self.assertIn("--reference-path reference/reference.json", workflow)
        self.assertNotIn("--no-include-correctness", workflow)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_benchmark_orchestrator.py::WorkflowTests -q`
Expected: FAIL (jane-austen still present; `--fixture-output` absent)

- [ ] **Step 3: Edit the workflow**

In `.github/workflows/benchmark-arm64.yml`:
- Remove the two "Download jane-austen benchmark corpus" steps and the `FIXTURE_URL`/`FIXTURE_JSONL` env in both `reference` and `benchmark` jobs.
- In the `reference` job's "Generate CN/EN cross-lingual retrieval eval" step, add `--fixture-output reference/cn_en_fixture.json`.
- In the `reference` job's emit step, drop `--fixture-path`/`--output-csv` jane-austen wiring; keep `--retrieval-eval-path reference/cn_en_retrieval_cases.json --emit-reference reference/reference.json` (plus model/dim/threads/l2 flags).
- Add `reference/cn_en_fixture.json` to the `Upload reference artifact` `path:` list.
- In the `benchmark` job's harness step, replace `--fixture-path "$FIXTURE_JSONL"` with `--fixture-path reference/cn_en_fixture.json`; delete the `--no-include-correctness` branch from `EXTRA_ARGS`.

- [ ] **Step 4: Validate YAML + run workflow tests**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/benchmark-arm64.yml')); print('YAML OK')" && python3 -m pytest tests/test_benchmark_orchestrator.py::WorkflowTests -q`
Expected: `YAML OK` then PASS

- [ ] **Step 5: Full Python suite + Rust lint**

Run: `python3 -m pytest tests/ -q` (the pre-existing `test_compare_embedding_outputs.py::...raw_compare_only` failure is unrelated and expected to remain)
Run: `STATIC_LLAMA_DIR="$(pwd)/.llama-artifacts/extracted" cargo clippy --all-targets -- -D warnings`
Expected: Python — only the known unrelated failure; Rust — `Finished`, no warnings.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/benchmark-arm64.yml tests/test_benchmark_orchestrator.py
git commit -m "ci: one CN/EN dataset, retrieval-only pytorch reference"
```

---

## Notes for the implementer

- **Correctness rows are per-document** (~1000 per quant): accurate `mean`/`min` for the report with zero new report logic. If CSV size ever matters, collapse to per-language aggregates later — not now (YAGNI).
- **`min_cosine` stays a display column**; only the *gate* moved to `mean_cosine`.
- The pre-existing failing test `tests/test_compare_embedding_outputs.py::...raw_compare_only` is unrelated (it asserts a `raw_compare_only` workflow feature that has never existed) — do not try to fix it here.
- After Task 8, the scenario contract lives in exactly **two** places: `run_embedding_benchmarks.py` `SCENARIOS` and `src/benchmarking.rs` `BENCHMARK_SCENARIOS`.
