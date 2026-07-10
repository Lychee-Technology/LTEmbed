# CN/EN-only benchmark redesign

## Context & goal

The embedding benchmark currently draws on **two** datasets: warm/cold latency
and correctness-vs-FP32 use the jane-austen English corpus (via `--fixture-path`,
split into `single/short|medium|long` by chunk length), while only the retrieval
eval uses `tests/CN_EN_Data.csv`. This split exists because short/medium/long is a
*length* profile that CN_EN — a set of uniformly short CN↔EN translation pairs —
cannot provide.

The goal is to make **all** measurements derive from `tests/CN_EN_Data.csv` (the
actual multilingual use case), collapse the length scenarios into per-language
latency scenarios, and drop jane-austen. Two decisions were taken during design:

1. **Correctness (cosine vs FP32) is derived from the retrieval embeddings**, not a
   separate pass — retrieval already embeds ~1000 CN/EN sentences with both
   ltembed (per quant) and PyTorch (the reference).
2. **The quality gate switches from `min_cosine` to `mean_cosine`** (min stays as a
   display-only column). Over ~1000 sentences a hard min is too easily tripped by a
   single outlier.

## Resulting architecture

One dataset (`CN_EN_Data.csv`), one embedding pass per implementation (retrieval),
plus a tiny latency pass:

```
reference job (once):   pytorch  → retrieval embeddings on CN/EN → reference.json
quant job (per quant):  ltembed  → warm/cold latency (single/zh, single/en)
                        ltembed  → retrieval embeddings on CN/EN
        derive:  both@3 / recall   (quality, ltembed vs reference ranking)
                 cosine vs FP32     (fidelity, ltembed docs vs reference docs)
```

PyTorch runs **once**, retrieval-only. Each quant runs ltembed warm + cold +
retrieval. Correctness is no longer a subprocess mode.

## Components

### 1. Generator — `scripts/build_cn_en_retrieval_cases.py`

Produces **two** deterministic files from the CSV (`--output` for the retrieval eval,
`--fixture-output` for the latency fixture):

- `cn_en_retrieval_cases.json` — the retrieval eval (unchanged: ~500 sampled pairs
  → both-language docs + bidirectional queries, each query relevant to both docs).
- `cn_en_fixture.json` — an already-**resolved** fixture (existing runner format
  `{"scenarios": {name: [{"kind","text"}]}}`) with exactly two entries:
  - `single/zh`: one deterministically-picked Chinese sentence (`kind: "query"`).
  - `single/en`: its English counterpart (`kind: "query"`).

  Picked from a fixed pair index (e.g. the median-length pair) so every job uses the
  identical two sentences.

### 2. Scenarios — now two, latency-only

`single/zh`, `single/en` replace `single/short|medium|long`. They exist only to
drive ltembed warm/cold latency, split by language (Chinese vs English tokenize
differently). Defined in **two** places now (was three):

- `scripts/run_embedding_benchmarks.py` `SCENARIOS` — for cold iteration + mapping
  ltembed warm/cold results to CSV rows.
- `src/benchmarking.rs` `BENCHMARK_SCENARIOS` — the ltembed runner's built-in set,
  with `scenario_inputs` returning a built-in zh/en sample as the no-fixture
  fallback.

`scripts/bench_pytorch.py` **no longer needs `SCENARIOS`**: PyTorch is retrieval-only,
so its warm/cold/correctness modes (and their scenario list) are removed.

### 3. Correctness derived from retrieval — `run_embedding_benchmarks.py`

- Remove the `correctness` subprocess mode from the orchestrator flow, the
  `--include-correctness` flag, and the `collect_correctness_rows` subprocess calls.
- After retrieval, for each retrieval **document** (~1000 CN/EN sentences), compute
  cosine between the ltembed embedding and the reference (FP32) embedding for the
  same document id. Emit one correctness row per document (`implementation=ltembed`,
  `mode=correctness`, `scenario="cn-en/zh"|"cn-en/en"`, `cosine_similarity_vs_pytorch`
  = that document's cosine). This reuses the existing per-row aggregation in the
  report unchanged and yields an accurate mean and min over all sentences.
- The **PyTorch reference shrinks to retrieval-only**: `--emit-reference` runs pytorch
  retrieval and writes `{"retrieval": <payload>}` (no `correctness` key). Version
  strings for the summary come from the retrieval payload.

### 4. Report — `scripts/render_benchmark_report.py`

- `recommend()` gates on **`mean_cosine >= QUALITY_GATE`** (was `min_cosine`). Pick the
  smallest GGUF whose mean cosine clears the gate. `min_cosine` remains a displayed
  column. Update the gate wording accordingly.
- Columns already surface `both@3` / `recall@3` / `mrr@3`; no schema change beyond the
  gate metric.

### 5. CI — `.github/workflows/benchmark-arm64.yml`

- **Drop** the jane-austen corpus download in both the reference and quant jobs.
- Reference job: generate `cn_en_retrieval_cases.json` **and** `cn_en_fixture.json`,
  run pytorch retrieval → `reference.json`, upload all three.
- Quant jobs: download the reference artifact; pass `--fixture-path cn_en_fixture.json`
  (now an already-resolved fixture) for ltembed warm/cold, `--retrieval-eval-path
  cn_en_retrieval_cases.json`, and `--reference-path reference.json`. No jane-austen,
  no `--include-correctness`.

## Data flow (per quant job)

1. ltembed warm on `single/zh` + `single/en` (fixture texts) → latency rows.
2. ltembed cold on each of `single/zh`, `single/en` → cold rows.
3. ltembed retrieval on `cn_en_retrieval_cases.json` → doc/query embeddings.
4. Derive from (3) + reference: both@3/recall rows (quality) and per-document
   correctness rows (cosine vs FP32).
5. Write CSV + summary.

## Removed / simplified

- jane-austen corpus, `resolve_fixture` (length-based picking), `--fixture-path` raw
  JSONL resolution → replaced by the pre-resolved `cn_en_fixture.json`.
- `correctness` subprocess mode (orchestrator + Rust binary; `bench_pytorch.py`
  warm/cold/correctness modes) and the `--include-correctness` flag.
- `bench_pytorch.py` `SCENARIOS` (PyTorch is retrieval-only).
- Scenario definitions drop from three sources to two.

## Testing

- **Generator**: emits both files deterministically; `cn_en_fixture.json` has exactly
  `single/zh` + `single/en` with a Chinese and an English sentence.
- **Derived correctness**: given a fabricated reference + ltembed retrieval payload with
  known embeddings, the orchestrator emits per-document correctness rows with the
  expected cosines; identical embeddings → cosine 1.0.
- **Report gate**: `recommend()` picks the smallest quant whose `mean_cosine` clears the
  gate (add a case where the smallest quant has a low outlier but high mean — it should
  now be recommended, unlike the old min gate).
- **Reference-only shape**: `--emit-reference` writes `{"retrieval": ...}` only; a quant
  run with `--reference-path` launches no PyTorch, emits latency + retrieval +
  derived-correctness rows.
- **Scenarios**: orchestrator + Rust expose `single/zh`, `single/en`; no `short/medium/long`
  or `batch/*` remain anywhere.
- **Rust**: `cargo clippy --all-targets -D warnings` clean (link/test runs on CI Linux).

## Verification (end-to-end)

1. `python3 scripts/build_cn_en_retrieval_cases.py --output <retrieval> --fixture-output <fixture>`
   → inspect both files.
2. Emit reference: `run_embedding_benchmarks.py --emit-reference ref.json
   --retrieval-eval-path <retrieval> ...` → `{"retrieval": ...}` only.
3. Consume for one quant with `--reference-path ref.json --fixture-path <fixture>
   --retrieval-eval-path <retrieval>` → CSV has warm/cold (single/zh, single/en),
   retrieval both@3/recall, and per-document correctness rows; no PyTorch launched.
4. `render_benchmark_report.py` → report gates on mean cosine.
