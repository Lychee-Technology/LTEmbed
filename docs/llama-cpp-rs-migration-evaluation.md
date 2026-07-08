# Evaluation: replace `ort` with `llama-cpp-rs`

Status: **assessment / decision document** (no code change). Captures the pros, cons,
cost, and a phased migration plan for swapping LTEmbed's inference backend from ONNX
Runtime (`ort`) to `llama-cpp-rs` (`llama-cpp-2` / `llama-cpp-sys-2`, binding
llama.cpp/ggml + GGUF).

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
- **The exact model is already available as GGUF.** Jina officially publishes GGUF for
  jina-embeddings-v5, and llama.cpp supports it with `--embedding --pooling last` — which
  matches LTEmbed's last-token pooling. This removes the single biggest migration risk
  (model convertibility).
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
  be built essentially from scratch**, though the benchmark harness and pooling logic are
  directly reusable.

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
   reference and, ideally, retrieval quality (`scripts/retrieval_eval_cases.json`). Golden
   fixtures need regenerating.
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
| GGUF conversion + quant selection + un-normalized 768-d emit + tokenizer parity | 3–5 days |
| New GGUF bundle build pipeline replacing `minimal-ort-builder` (incl. aarch64 cross-compile with C++ toolchain) | 1–2 weeks |
| CI changes: add CMake/C++/libclang, submodule, longer builds; re-measure Lambda cold-start & size | 2–3 days |
| Correctness re-validation: regenerate golden fixtures, cosine parity, retrieval eval | 2–4 days |
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
See "Prior-work salvage" above. No reusable llama.cpp embedding backend exists; benchmark
harness + pooling logic are reusable.

### Phase 1 — De-risk spike (throwaway, ~3–4 days) **[GATE]**
- Obtain/convert `jina-embeddings-v5-text-nano-retrieval` to GGUF (prefer Jina's official
  GGUF; fall back to `convert_hf_to_gguf.py`). Pick a quant (start Q8_0 for parity, then
  Q4_K_M/Q5_K_M for size).
- Standalone binary using `llama-cpp-2`: load GGUF, `pooling = last`, feed the HF-tokenized
  ids, emit **raw un-normalized 768-d** embedding.
- Reuse `scripts/compare_embedding_outputs.py` + golden fixtures to check **cosine > 0.99**
  vs PyTorch reference after 768→512 truncation + L2.
- Benchmark on ARM64 against the current ORT path using existing scenarios
  (`single/medium`, `single/long`, `batch/medium/8`).
- **Gate:** proceed only if cosine parity holds **and** latency/size is at least competitive.

### Phase 2 — Backend abstraction (2–4 days)
- Introduce an internal `EmbeddingBackend` trait (roughly `load(...)`,
  `run(ids, mask) -> raw_hidden`) so `OnnxEngine` orchestration (`embed`/`embed_batch`,
  pooling, truncation, norm, error types) is backend-neutral.
- Move the current `ort` calls behind an `OrtBackend`; keep it the default so nothing breaks.
  Critical files: `src/engine/mod.rs`, `src/engine/session_io.rs`, `src/engine/ort_init.rs`.
- Keep `src/engine/inference.rs`, `src/traits/tokenizer.rs`, `src/engine/{config,input}.rs`,
  and `src/error.rs` unchanged (already backend-agnostic).

### Phase 3 — llama.cpp backend behind the trait (3–5 days)
- Add `llama-cpp-2`/`llama-cpp-sys-2` (CPU/NEON feature only) and a `LlamaBackend`
  implementing `EmbeddingBackend`: model load from GGUF, context with `pooling = last`, batch
  tokenize via the existing `HFTokenizer`, extract raw 768-d.
- Feature-flag the backend (`--features llama` / `ort`) so both compile. Defer the
  `OnnxEngine` rename to a later breaking release.
- Replace the `.ort`-specific bundle contract (`src/engine/bundle.rs`) with a GGUF path; drop
  `ORT_DYLIB_PATH`/`libonnxruntime.so` requirements on the llama backend.

### Phase 4 — Build & release pipeline (1–2 weeks)
- Stand up a GGUF bundle builder (new workflow or repo) replacing `minimal-ort-builder`:
  produce `model.gguf` + `tokenizer.json` + `build-info.json` + `SHA256SUMS`.
- Update `.github/workflows/release-bundles.yml` and CI to install CMake/C++/libclang, pull
  the llama.cpp submodule, and cross-compile for `linux-arm64`. Re-measure Lambda cold-start
  and package size against the 250 MB limit.

### Phase 5 — Validate, then flip the default (2–4 days)
- Regenerate golden fixtures for the chosen GGUF quant; run parity + retrieval-eval suites;
  run the benchmark suite on ARM64.
- Flip the default backend to llama.cpp once green; schedule the public rename
  (`OnnxEngine` → e.g. `EmbeddingEngine`) as an explicit breaking release.

## Verification (for the eventual implementation)

- **Parity:** `cargo test` incl. `test_golden_parity_cosine_similarity` with
  `LTEMBED_TEST_BUNDLE_DIR` pointing at a GGUF bundle; require cosine > 0.99. Cross-check with
  `scripts/compare_embedding_outputs.py` against `scripts/bench_pytorch.py`.
- **Retrieval quality:** evaluate against `scripts/retrieval_eval_cases.json`.
- **Performance:** run `src/bin/benchmark_ltembed.rs` (warm/cold) on ARM64 for both backends
  across `single/medium`, `single/long`, `batch/medium/8`; compare latency + per-stage
  `EmbedBatchProfile`.
- **Footprint/cold-start:** measure stripped binary + GGUF size vs the 250 MB Lambda limit,
  and cold-start init time.

## Open questions to resolve during Phase 1

- Does Jina's official jina-v5-**nano-retrieval** GGUF exist, or must we convert? (v5-omni
  GGUFs are published; confirm the text-nano-retrieval variant specifically.)
- Can `llama-cpp-2` emit the **un-normalized full 768-d** vector so the existing Matryoshka
  truncation + L2 logic is preserved bit-for-bit?
- Which quant meets both the cosine-parity bar and the size/latency targets?

## References

- llama-cpp-rs: <https://github.com/utilityai/llama-cpp-rs> · crates `llama-cpp-2` /
  `llama-cpp-sys-2`
- llama.cpp embedding + pooling types:
  <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- Jina — Multimodal Embeddings in llama.cpp and GGUF:
  <https://jina.ai/news/multimodal-embeddings-in-llama-cpp-and-gguf/>
- ort: <https://github.com/pykeio/ort>
