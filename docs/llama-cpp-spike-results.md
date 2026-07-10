# llama.cpp/GGUF migration — spike + backend results

Status: **correctness + size validated; ORT fully removed; `LlamaBackend` is the sole
backend behind the `EmbeddingBackend` trait.** The de-risk spike (below) is throwaway and
was subsumed by the real `src/engine/llama/` backend. Follow-on to
`docs/llama-cpp-rs-migration-evaluation.md`.

Decision (2026-07-08): drop ONNX Runtime entirely rather than keep a dual backend; keep the
`EmbeddingBackend` trait purely as the future extensibility seam. No ORT perf comparison —
compare GGUF quants (Q8_0 vs Q5_K_M) instead.

## What was proven

- **Linking works on aarch64.** The prebuilt static archives from
  `static-llama-cpp-rs-builder` `v0.1.151-1` (contract v2, llama.cpp `b9553`, no `common`/
  OpenMP) link cleanly with the documented link line — **no `--whole-archive` needed**.
  Consumed via raw FFI: `build.rs` (feature-gated on `llama`) adopts the release's
  `consume.build.rs`, and the crate does `include!(env!("STATIC_LLAMA_BINDINGS"))`.
- **Embeddings work.** Load GGUF → context with `embeddings=true`,
  `pooling_type=LLAMA_POOLING_TYPE_LAST` → feed HF-tokenized ids in a `llama_batch` →
  `llama_decode` → `llama_get_embeddings_seq` returns the **raw, un-normalized pooled 768-d**
  vector (L2 ≈ 90–113), exactly what LTEmbed's existing 768→512 truncate + L2
  `postprocess_embedding` expects. Model reports `n_embd = 768`,
  `general.architecture = eurobert`. (Embedding contexts auto-route `decode`→`encode`.)
- **Correctness parity** vs the PyTorch/F32 golden (`tests/fixtures/test_fixtures.json`,
  regenerated in fp32 — see below), cosine over the 4 fixtures after truncate+L2:

  | Quant  | Size    | min cosine | mean cosine | Gate |
  |--------|---------|------------|-------------|------|
  | Q8_0   | 233 MB  | 0.99970    | 0.99980     | parity ✅, too big for Lambda |
  | **Q5_K_M** | **169 MB** | **0.99663** | 0.99720 | **parity ✅, fits Lambda ✅ (pick)** |
  | Q4_K_M | 157 MB  | 0.98937    | 0.99169     | parity ✗ (dips to 0.9894) |

- **Size / Lambda.** Stripped release binary (static ggml linked in) = **7 MB**. With
  Q5_K_M (169 MB) + the real tokenizer.json (17 MB) → ≈ **193 MB**, comfortably under the
  250 MB uncompressed Lambda limit. Q8_0 (233 + 17 + 7 ≈ 257 MB) would exceed it.

## Recommended quant

**Q5_K_M** — the smallest quant that clears cosine ≥ 0.99 (0.9966) while fitting Lambda.
Q8_0 is the parity ceiling but too large; Q4_K_M is too lossy.
Pinned: `v5-nano-retrieval-Q5_K_M.gguf`
sha256 `46fbc0423862cb6a5d4ff776d885f349d2a87c36d821dd5630f9fa184c9b4b92`
(HF repo commit `ac5d898c8d382b17167c33e5c8af644a3519b47d`).

## Gotchas found

1. **`assets/tokenizer.json` is the wrong tokenizer** — a stale BERT WordPiece tokenizer
   (30k vocab). Feeding its ids into the model gave cosine ≈ 0. The real model uses a
   **128k-vocab Llama-style BPE** tokenizer (bos 128000/eos 128001); download `tokenizer.json`
   from the HF model repo. **The GGUF bundle contract must ship the real tokenizer.**
2. **Golden was a placeholder.** `tests/fixtures/test_fixtures.json` shipped empty
   (`dim: 0`), so the parity test was silently skipping. Regenerated from PyTorch via
   `scripts/generate_fixtures.py`, which also needed a fix: the model's native dtype is
   bfloat16, so it now loads with `torch_dtype=torch.float32` (the golden is the F32 reference;
   bf16 tensors also can't convert to numpy directly).

## Quant performance: Q8_0 vs Q5_K_M

Warm latency through the real `benchmark_ltembed` binary (`--mode warm`, `threads=1`,
50 iters). **Apple-Silicon aarch64 dev box — NOT Graviton**, so treat as relative signal:

| Scenario       | Q8_0 p50 | Q8_0 p95 | Q5_K_M p50 | Q5_K_M p95 |
|----------------|----------|----------|------------|------------|
| single/short   | 10.5 ms  | 11.0 ms  | 21.4 ms    | 33.6 ms    |
| single/medium  | 18.4 ms  | 19.5 ms  | 32.6 ms    | 33.1 ms    |
| single/long    | 466 ms   | 480 ms   | 741 ms     | 748 ms     |
| batch/medium/8 | 146 ms   | 147 ms   | 261 ms     | 283 ms     |

**Q8_0 is ~1.7–1.8× faster than Q5_K_M here** (simpler dequant than Q5_K's block structure).
So the trade-off is: **Q8_0 = faster but 233 MB (too big for Lambda); Q5_K_M = fits Lambda
but slower.** Caveats: this is not Graviton (the tuned Neoverse-N1 build with dotprod/i8mm
may narrow the Q5_K gap), `batch/*` is sequential decode (one `llama_decode` per input — true
multi-sequence batching is a future optimization), and `single/long` pays full non-causal
attention over a long sequence. Re-run on Graviton before committing to a quant.

## Reproduce (aarch64 Linux container)

```
.llama-artifacts/dev.sh cargo run --release --features llama --bin spike_llama -- \
    .llama-artifacts/gguf/v5-nano-retrieval-Q5_K_M.gguf
# then compare .llama-artifacts/spike_out.json vs tests/fixtures/test_fixtures.json by cosine
```
