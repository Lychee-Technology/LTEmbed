# Evaluation: replace `ort` with `llama-cpp-rs`

Status: **exploratory / non-binding decision record** (no code change). `main` remains
ORT-only / `OnnxEngine` / `ort_bundle` (see `README.md`, `docs/design.md`,
`docs/development.md`); this document does **not** commit the project to a migration.
Captures the pros, cons, cost, and a phased migration plan for swapping LTEmbed's
inference backend from ONNX Runtime (`ort`) to `llama-cpp-rs` (`llama-cpp-2` /
`llama-cpp-sys-2`, binding llama.cpp/ggml + GGUF).

## Context

LTEmbed is a narrow Rust **embedding library** (`ltembed`, lib crate) wrapping ONNX
Runtime via the `ort` crate (`2.0.0-rc.12`, `load-dynamic`). It runs exactly one model —
`jinaai/jina-embeddings-v5-text-nano-retrieval` — as a fixed 2-input (`input_ids`,
`attention_mask`) / 1-output (`last_hidden_state [B,S,768]`) graph, then does **last-token
pooling → 768→512 Matryoshka truncation → optional L2 normalize**. Target: **CPU-only,
ARM64 (Graviton / Apple Silicon)**, intended to be embedded in an AWS Lambda `bootstrap`.
The model + `libonnxruntime.so` ship in an `ort_bundle/` tarball produced by an external
builder repo (`Lychee-Technology/minimal-ort-builder`, `model.ort` in INT8/q4f16).

Drivers for considering the swap: **GGUF ecosystem**, **ARM perf/cost**, and **future
multi-model** support (generative / rerankers / jina-v5-omni).

## Key findings that shape the decision

- **The backend seam is clean and small.** Only three files import `ort`:
  `src/engine/mod.rs` (session build + run), `src/engine/ort_init.rs` (dylib init),
  `src/engine/session_io.rs` (graph introspection — type-only). Everything else — tokenizer
  (`src/traits/tokenizer.rs`, HF `tokenizers` crate), pooling/pack/normalize
  (`src/engine/inference.rs`), config/input/error types, and the whole benchmark harness
  (`src/bin/benchmark_ltembed.rs`, `src/benchmarking.rs`) — is backend-agnostic and
  reusable. No `ort` type appears in any public signature.
- **The exact model already ships as GGUF (confirmed).** `jinaai/jina-embeddings-v5-text-nano-retrieval`
  publishes GGUF siblings including **F16, Q8_0, Q5_K_M, Q4_K_M**, and llama.cpp runs it with
  `--embedding --pooling last` — matching LTEmbed's last-token pooling. Model convertibility
  is therefore **not** an open risk; Phase 1's job is to validate a *specific* quant on
  correctness, performance, size, and runtime contract.
- **A correctness gate already exists**: golden fixtures (`tests/fixtures/test_fixtures.json`),
  the parity test (`test_golden_parity_cosine_similarity`, cosine > 0.99), and Python
  reference scripts (`scripts/bench_pytorch.py`, `compare_embedding_outputs.py`,
  `compare_q4f16_onnx_vs_pytorch.py`). These are directly reusable to validate a GGUF backend.

### Prior-work salvage (Phase 0 findings)

Investigated whether an existing llama.cpp experiment could be reused. Conclusion: **there
is no reusable llama.cpp embedding backend to salvage.**

- `benches/build/*.o` are *untracked* compiled ggml/llamafile objects (`sgemm.o`,
  `vendor_llama.cpp_ggml_*`). No `bench.c`/`bench.cpp` source exists in git history — they
  are leftovers from an ad-hoc GEMM/kernel micro-benchmark (see the `matrixmultiply`
  branch's `bench-kernel-compare.yml`, which runs Criterion kernel microbenchmarks on
  Neoverse N1 against `e5-small-v2`), **not** an embedding backend. Recommend
  gitignoring/cleaning them.
- The `matrixmultiply` branch is a *different* backend entirely: it deletes `src/engine/`
  (ORT) and replaces it with a hand-rolled pure-Rust BERT (`src/models/bert.rs`) over
  `src/gemm.rs`/`matrixmultiply`. There is also an `examples/benchmark_candle.rs` there — a
  **Candle** backend was explored too. Reusable references only: `src/traits/pooling.rs` and
  the kernel-benchmark infra.
- Net: the team has already tried two other backend directions (NEON GEMM, Candle) before
  landing on ORT — perf on ARM has been the recurring motivation. **The Phase 1 spike must
  be built essentially from scratch.** Reusable: the Python orchestration + scenario
  definitions (`scripts/run_embedding_benchmarks.py`, `bench_pytorch.py`) and the pooling /
  postprocess logic. **Not** fully backend-agnostic yet: the Rust `benchmark_ltembed` binary
  is still built around `OnnxEngine`, `--ort-bundle-dir`, and `model.ort`, so it must be
  generalized (see cost table).

## Pros of switching to `llama-cpp-rs`

1. **Self-contained artifact.** llama.cpp compiles from source and statically links ggml →
   one binary, no `libonnxruntime.so` to ship, no `ORT_DYLIB_PATH` resolution, and the whole
   external `minimal-ort-builder` dylib pipeline goes away. Simpler runtime contract for Lambda.
2. **GGUF ecosystem (a stated driver).** Native, well-tooled quantization (Q4_K_M, Q5_K_M,
   Q8_0), Jina's official GGUF releases, `convert_hf_to_gguf.py`. Replaces the bespoke
   q4f16/INT8 ONNX quantization path.
3. **Future multi-model (a stated driver).** The same backend runs generative LLMs,
   rerankers (`--pooling rank`), and jina-v5-omni multimodal. If LTBase grows past this one
   embedding model, llama.cpp is one backend for all of them; ORT would need a separate
   generative stack.
4. **ARM perf/cost (a stated driver).** ggml's ARM NEON quant kernels are extremely mature —
   the motivation behind the earlier `matrixmultiply` GEMM experiment — and with llama.cpp
   you get those optimizations maintained upstream for free. **Must be measured**, not
   assumed (see Phase 1).
5. **Hardware portability later.** CPU/Metal/CUDA/Vulkan are cargo features in llama-cpp-2,
   vs ORT execution providers which are heavier to enable.
6. **Larger, faster-moving community** than `ort` (which is still `2.0.0-rc`).

## Cons / risks

1. **Build complexity moves *into* the crate.** Today LTEmbed's build is trivially fast and
   pure-Rust (ort is `load-dynamic`; nothing C compiles). llama-cpp-sys-2 compiles llama.cpp
   from a **git submodule** and needs **CMake + a C/C++ toolchain + libclang (bindgen)** in
   every build environment. Build time goes from seconds to minutes; **cross-compiling for
   aarch64 Lambda with a C++ toolchain is the main new friction.** This partially offsets the
   "deploy simplicity" win.
2. **Static binary grows.** ggml is compiled into the Rust binary (vs the current separate
   dylib). Total footprint vs the Lambda 250 MB uncompressed limit needs re-measuring; likely
   fine with a small GGUF quant, but not free.
3. **Quantization parity is a real validation task.** GGUF K-quants ≠ ONNX q4f16/INT8, so
   embeddings *will* differ numerically. Must re-confirm cosine > 0.99 vs the PyTorch
   reference and, ideally, retrieval quality (`scripts/retrieval_eval_cases.json`).
   **Fixture policy:** keep the existing PyTorch/F32 golden fixtures
   (`tests/fixtures/test_fixtures.json`, produced by `scripts/generate_fixtures.py`) as the
   **immutable reference** — do *not* regenerate golden from GGUF output, or quantization
   drift gets fossilized as "correct." Each GGUF quant is validated *against* that reference
   via a comparison report / backend-specific snapshot.
4. **API-paradigm mismatch (manageable).** llama.cpp returns a *pooled* embedding directly
   (pooling type set at context creation, `LLAMA_POOLING_TYPE_LAST`) and can normalize
   internally. LTEmbed wants the **raw, un-normalized 768-d** vector so it can do its own
   Matryoshka 768→512 truncation *then* optional L2. Need to confirm llama-cpp-2 can emit the
   full-dim un-normalized embedding (use `embd_normalize = -1` / raw path) so
   `postprocess_embedding` in `src/engine/inference.rs` stays unchanged.
5. **Tokenizer parity.** LTEmbed uses the HF `tokenizers` crate against the bundle's
   `tokenizer.json`. llama.cpp has its own GGUF-embedded tokenizer. Cheapest path: keep the HF
   tokenizer and feed token ids into llama.cpp (it accepts token-id input), then validate
   token parity.
6. **Dependency churn.** llama-cpp-2 is `0.1.x`, versioned lockstep with a pinned llama.cpp
   submodule and ships breaking changes often. Ongoing cost: keeping the submodule current.
7. **Public-API naming churn.** `OnnxEngine`, `OnnxEngineConfig`, `InferenceError::OrtRun`
   become misnomers. Either a breaking rename (affects callers) or keep misleading names.
8. **Sunk cost.** The team just invested in the ort path (recent commits: dropped
   `download-binaries`/`tls-native`, wired `--threads` to intra-op threads,
   `minimal-ort-builder` v1.0.9). Switching discards that.

## Cost (effort estimate)

Dominated by the build/deploy pipeline and validation, **not** the Rust code.

| Work item | Rough effort |
|---|---|
| Backend swap in `engine/mod.rs` + `ort_init.rs` + `session_io.rs` (behind a trait) | 2–4 days |
| Quant selection + pooled 768-d emit + tokenizer parity | 3–5 days |
| Generalize `benchmark_ltembed` to `--backend ort\|llama`, `--bundle-dir`, `--model-path/--gguf-path` (+ quant metadata in result rows) | 1–2 days |
| Prebuilt llama/ggml static-library builder repo + artifact contract (recommended — see Phase 4) | 1–2 weeks |
| CI changes: consume prebuilt artifact (or add CMake/C++/libclang + submodule), re-measure Lambda cold-start & size | 2–3 days |
| Correctness validation: cosine parity + retrieval eval against the **immutable PyTorch golden** (comparison reports per quant) | 2–4 days |
| **Total to validated parity + deployable artifact** | **~3–5 weeks focused** |

## Recommendation

Given all three drivers (GGUF, ARM perf/cost, future multi-model) genuinely favor
llama.cpp, and given the clean backend seam + the model already existing as GGUF, this is
**worth pursuing — but gate the commitment on a short spike first.** The one thing that can
sink it is that a *stated* driver is performance, and there are **no measured llama.cpp vs
ORT numbers yet** on the target ARM64 hardware. Do not migrate the pipeline before Phase 1
proves both parity and a perf win.

## Migration plan (phased — each phase is a go/no-go gate)

### Phase 0 — Salvage prior work (done)
See "Prior-work salvage" above. No reusable llama.cpp embedding backend exists; the Python
benchmark orchestration + pooling/postprocess logic are reusable, but the Rust
`benchmark_ltembed` binary must be generalized beyond ORT.

### Phase 1 — De-risk spike (throwaway, ~3–4 days) **[GATE]**
- Pull the already-published GGUF (`v5-nano-retrieval` F16 / Q8_0 / Q5_K_M / Q4_K_M); pin the
  exact file + SHA. Start Q8_0 for parity, then Q5_K_M / Q4_K_M for size. (Convert via
  `convert_hf_to_gguf.py` only as a fallback if an official quant is missing.)
- Standalone binary using `llama-cpp-2`: load GGUF, `with_pooling_type(Last)`, feed the
  HF-tokenized ids, extract the **raw un-normalized pooled 768-d** embedding
  (`embeddings_seq_ith`, `embd_normalize = -1`).
- Reuse `scripts/compare_embedding_outputs.py` + the immutable PyTorch golden to check
  **cosine > 0.99** after 768→512 truncation + L2.
- Benchmark on ARM64 against the current ORT path using existing scenarios
  (`single/medium`, `single/long`, `batch/medium/8`).
- **Quantitative gate (all must pass to proceed):**
  - **Hardware/config:** Graviton2 / Neoverse N1 runner, `threads=1` (LTEmbed default), warm
    + cold measured separately.
  - **Correctness:** cosine ≥ **0.99** vs PyTorch reference for the chosen quant; retrieval
    eval (`scripts/retrieval_eval_cases.json`) within an agreed tolerance of the ORT baseline.
  - **Performance:** warm p95 latency **≤ current ORT** on each scenario (record p95/p99);
    batch throughput not worse than ORT.
  - **Size / cold-start:** stripped binary + GGUF within the Lambda 250 MB uncompressed
    limit; cold-start (init + model load + first inference) recorded and acceptable.

### Phase 2 — Backend abstraction (2–4 days)
- Introduce an internal `EmbeddingBackend` trait whose output is a **raw, un-normalized
  pooled 768-d embedding per input** (roughly `load(...)`,
  `embed(tokenized) -> Vec<[f32; 768]>`) — *not* ORT-style `[B,S,768]` hidden states. This
  matches how llama.cpp naturally emits embeddings and keeps the shared
  `postprocess_embedding` (768→512 truncation + optional L2) unchanged. Last-token pooling
  becomes a backend responsibility: the ORT backend runs the existing `pool_last_token`
  internally; the llama backend uses `with_pooling_type(Last)`.
- Move the current `ort` calls behind an `OrtBackend`; keep it the default so nothing breaks.
  Critical files: `src/engine/mod.rs`, `src/engine/session_io.rs`, `src/engine/ort_init.rs`;
  relocate `pool_last_token` from the shared path into `OrtBackend`.
- Keep `src/traits/tokenizer.rs`, `src/engine/{config,input}.rs`, `src/error.rs`, and the
  truncation/normalize half of `src/engine/inference.rs` unchanged (already backend-agnostic).

### Phase 3 — llama.cpp backend behind the trait (3–5 days)
- Add `llama-cpp-2`/`llama-cpp-sys-2` (CPU/NEON feature only) and a `LlamaBackend`
  implementing `EmbeddingBackend`: model load from GGUF, context with
  `with_embeddings(true)` + `with_pooling_type(Last)`, batch tokenize via the existing
  `HFTokenizer`, extract the raw pooled 768-d via `embeddings_seq_ith` (`embd_normalize = -1`).
- Feature-flag the backend (`--features llama` / `ort`) so both compile. Defer the
  `OnnxEngine` rename to a later breaking release.
- Replace the `.ort`-specific bundle contract (`src/engine/bundle.rs`) with a GGUF path; drop
  `ORT_DYLIB_PATH`/`libonnxruntime.so` requirements on the llama backend.

### Phase 4 — Build & release pipeline (1–2 weeks)
- **Do not compile llama.cpp/ggml from source in the LTEmbed repo on every build.** Mirror
  the existing `minimal-ort-builder` pattern: stand up a separate **prebuilt static-library
  builder repo** (e.g. `ltembed-llama-builder`) that owns the CMake/bindgen/submodule work
  and publishes verified artifacts. LTEmbed then only *consumes* a pinned artifact.
  - Fully pin inputs: `llama-cpp-2` version/commit, llama.cpp submodule commit, Rust
    toolchain, CMake, compiler image, target triple (`aarch64-unknown-linux-gnu`), CPU
    profile (Neoverse N1 / `-mcpu=neoverse-n1`; avoid `target-cpu=native` for release), and
    enabled features.
  - Publish `libllama.a` / `libggml*.a` (+ bindings/headers), `build-info.json`, `SHA256SUMS`,
    licenses. Consume via a small sys-layer override (a `LLAMA_CPP_PREBUILT_DIR`-style patch of
    `llama-cpp-sys-2` that skips CMake/bindgen when prebuilt artifacts are present).
  - Add a smoke binary in the builder that links the `.a` files and runs a tiny embedding
    init; LTEmbed CI verifies SHA before use. Benchmark prebuilt vs a local
    `llama-cpp-sys-2` build once to confirm no perf regression. Version the artifact contract
    (llama.cpp / `llama-cpp-rs` offer weak semver stability).
- Produce the GGUF bundle (`model.gguf` + `tokenizer.json` + `build-info.json` + `SHA256SUMS`)
  and update `.github/workflows/release-bundles.yml` to consume the prebuilt lib + GGUF and
  cross-compile the `bootstrap`. Re-measure Lambda cold-start and package size vs 250 MB.

### Phase 5 — Validate, then flip the default (2–4 days)
- Run parity + retrieval-eval against the **immutable PyTorch golden** (do *not* replace the
  reference golden with GGUF output — see fixture policy in Cons #3); emit per-quant
  comparison reports; run the benchmark suite on ARM64.
- Flip the default backend to llama.cpp once green; schedule the public rename
  (`OnnxEngine` → e.g. `EmbeddingEngine`) as an explicit breaking release.

## Verification (for the eventual implementation)

- **Parity:** `cargo test` incl. `test_golden_parity_cosine_similarity` with
  `LTEMBED_TEST_BUNDLE_DIR` pointing at a GGUF bundle; require cosine > 0.99 **against the
  unchanged PyTorch/F32 golden** (`tests/fixtures/test_fixtures.json`). Cross-check with
  `scripts/compare_embedding_outputs.py` against `scripts/bench_pytorch.py`.
- **Retrieval quality:** evaluate against `scripts/retrieval_eval_cases.json`.
- **Performance:** run `src/bin/benchmark_ltembed.rs` (warm/cold) on ARM64 for both backends
  across `single/medium`, `single/long`, `batch/medium/8`; compare latency + per-stage
  `EmbedBatchProfile`.
- **Footprint/cold-start:** measure stripped binary + GGUF size vs the 250 MB Lambda limit,
  and cold-start init time.

## Open questions to resolve during Phase 1

- **Which GGUF file/quant do we select and pin** (with SHA)? Use an official published quant
  (F16 / Q8_0 / Q5_K_M / Q4_K_M) or self-convert via `convert_hf_to_gguf.py`?
- Does the chosen quant clear the quantitative gate — cosine ≥ 0.99 vs PyTorch, retrieval
  eval within tolerance, warm p95 ≤ ORT, and size/cold-start within Lambda limits?
- Can `llama-cpp-2` emit the **un-normalized pooled 768-d** vector (`embeddings_seq_ith`,
  `embd_normalize = -1`) so the existing Matryoshka 768→512 + L2 logic is preserved?
- Prebuilt static-library builder repo vs in-repo `llama-cpp-sys-2` compile — confirm the
  prebuilt path links cleanly and matches perf.

## References

- llama-cpp-rs: <https://github.com/utilityai/llama-cpp-rs> · crates `llama-cpp-2` /
  `llama-cpp-sys-2`
- llama.cpp embedding + pooling types:
  <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- Jina — Multimodal Embeddings in llama.cpp and GGUF:
  <https://jina.ai/news/multimodal-embeddings-in-llama-cpp-and-gguf/>
- ort: <https://github.com/pykeio/ort>
