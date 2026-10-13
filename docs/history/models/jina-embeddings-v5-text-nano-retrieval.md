# Jina Embeddings v5 Text Nano Retrieval on llama.cpp

> [!NOTE]
> **Status:** Historical (superseded)\
> **Applies to:** `jinaai/jina-embeddings-v5-text-nano-retrieval`, the model `main` ran from the llama.cpp migration (#149, 2026-07-09) until the EmbeddingGemma 2 migration\
> **Current implementation:** EmbeddingGemma 2 (`unsloth/embeddinggemma-2-GGUF`, `UD-Q4_K_XL`) on llama.cpp/GGUF\
> **Current documentation:** [docs/architecture.md](../../architecture.md), [docs/bundle-format.md](../../bundle-format.md), [docs/model-migration.md](../../model-migration.md)
>
> `main` no longer loads this model: its bundles declare `"pooling": "last_token"` and fail with `UnsupportedPooling`, and the engine pools with mean only. Its vectors are not comparable with EmbeddingGemma 2's. The commands below assembled a bundle for a `main` that no longer exists; they are kept as a record, not as current instructions.

This collects what the current documentation said about the model before the migration:
the model, its bundle, the quant choice and the results recorded for it. The text is kept
as it was, apart from headings and the links, which now point at the history.

## Model

The target model was
[`jinaai/jina-embeddings-v5-text-nano-retrieval`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval)
in GGUF form. The migration spike recorded `general.architecture = eurobert` and
`n_embd = 768` for it. LTEmbed ran it as an encoder: non-causal attention, last-token
pooling.

| Constant | Value |
|---|---|
| `RAW_EMBEDDING_DIMENSION` | `768` |
| `EMBEDDING_DIMENSION` | `512` (the `EngineConfig::default()` output size) |
| `MAX_LENGTH` | `8192` |
| `QUERY_PREFIX` / `DOCUMENT_PREFIX` | `"Query: "` / `"Document: "` |

### Why padding was stripped before the backend

Last-token pooling takes the hidden state of the final position fed to the model. In a
padded batch, shorter inputs end in pad tokens (pad id `128004`). Feeding them pooled the
pad position, which dropped cosine against the FP32 reference to about 0.31 and broke
cross-lingual retrieval. `TokenizerOutput::real_input_ids` removes padding so that each
sequence ends on its real last token.

## Bundle

### `model.gguf`

- A GGUF build of `jinaai/jina-embeddings-v5-text-nano-retrieval`. The Hugging Face repo
  publishes one file per quant as `v5-nano-retrieval-<QUANT>.gguf`, which the workflows
  renamed to `model.gguf`.
- Its embedding size (`n_embd`) and pooled output size (`n_embd_out`) both had to equal
  `raw_embedding_dimension` from `build-info.json`.
- `Q5_K_M` was the quant used by CI and by release tarballs. Benchmarks also ran `IQ4_NL`
  and `Q8_0`.
- Reference hash: at Hugging Face revision `ac5d898c8d382b17167c33e5c8af644a3519b47d`
  (the revision CI pinned), `v5-nano-retrieval-Q5_K_M.gguf` has sha256
  `46fbc0423862cb6a5d4ff776d885f349d2a87c36d821dd5630f9fa184c9b4b92` (about 169 MB).

### `tokenizer.json`

- The model's own Hugging Face tokenizer (`tokenizer.json` in the same Hugging Face repo): a
  BPE tokenizer with 128256 tokens (128000 BPE tokens and 256 added special tokens) and
  right padding with pad id `128004`. It is about 17 MB.
- At the pinned revision its sha256 is
  `98d4a1d32152d6cedf85b5e88f3b205106dca1fe72aaab34e0ac13c238421069`.
- Its vocabulary size, added tokens included, had to equal the number of tokens in the
  GGUF's own vocabulary (`tokenizer.ggml.tokens`, also 128256). That size check was the
  only check that the tokenizer belonged to the GGUF.

### `build-info.json`

```json
{
  "target_id": "jinaai/jina-embeddings-v5-text-nano-retrieval",
  "model_metadata": {
    "model_format": "gguf",
    "quant": "Q5_K_M",
    "pooling": "last_token",
    "input_kind": "retrieval",
    "query_prefix": "Query: ",
    "document_prefix": "Document: ",
    "raw_embedding_dimension": 768,
    "output_embedding_dimension": 768,
    "max_length": 8192
  }
}
```

`pooling` accepted `"last_token"` or `"lasttoken"`.

### Assembling a bundle by hand (obsolete)

This mirrored the CI step "Assemble GGUF bundle for model-backed tests" of the time. It
does not produce a bundle the current `main` loads.

```bash
GGUF_REPO=jinaai/jina-embeddings-v5-text-nano-retrieval
HF_REVISION=ac5d898c8d382b17167c33e5c8af644a3519b47d
QUANT=Q5_K_M
BUNDLE=gguf_bundle
mkdir -p "$BUNDLE"
curl -fsSL -o "$BUNDLE/model.gguf" \
  "https://huggingface.co/$GGUF_REPO/resolve/$HF_REVISION/v5-nano-retrieval-$QUANT.gguf"
curl -fsSL -o "$BUNDLE/tokenizer.json" \
  "https://huggingface.co/$GGUF_REPO/resolve/$HF_REVISION/tokenizer.json"
# build-info.json as above
```

The release and benchmark workflows downloaded `model.gguf` and `tokenizer.json` from
`resolve/main` instead of the pinned revision, so a release could contain files CI never
tested.

## Why Q5_K_M

The release and CI bundles used `v5-nano-retrieval-Q5_K_M.gguf`. The llama.cpp migration
spike (July 2026, see
[llama-cpp-spike-results.md](../migrations/llama-cpp-spike-results.md)) chose it as the
smallest quant that met both targets:

- **Parity**: cosine ≥ 0.99 against the PyTorch FP32 golden fixture. Q5_K_M scored a mean of
  0.99720 and a minimum of 0.99663. Q4_K_M dropped to 0.98937.
- **Size**: within the AWS Lambda unzipped package limit of 250 MB. Q5_K_M (169 MB) with the
  17 MB tokenizer and a 7 MB stripped binary came to about 193 MB. Q8_0 (233 MB) came to
  about 257 MB.

## Golden fixture

The golden fixture held four inputs (three queries, one document), each with its final
512-d, truncated and L2-normalized vector, produced by `scripts/generate_fixtures.py` with
`transformers` and `torch`. It loaded the model with `trust_remote_code=True` in float32
(the model's native dtype is bfloat16), then applied last-token pooling, truncation to 512
and L2 normalization. The script downloaded the model's latest Hugging Face revision; it
did not pin one. The parity gate was cosine ≥ 0.99 per input.

## Recorded results: llama.cpp migration spike, July 2026

Source: [llama-cpp-spike-results.md](../migrations/llama-cpp-spike-results.md), committed
with #149 on 2026-07-09.

**Golden parity** (cosine after truncation to 512 and L2 normalization, over the 4
golden fixtures):

| Quant | GGUF size | min cosine | mean cosine |
|---|---|---|---|
| Q8_0 | 233 MB | 0.99970 | 0.99980 |
| Q5_K_M | 169 MB | 0.99663 | 0.99720 |
| Q4_K_M | 157 MB | 0.98937 | 0.99169 |

**Warm latency, in ms.**

- Environment: an **Apple Silicon aarch64 development machine, not Graviton**. The exact
  chip, OS and llama.cpp build flags were not recorded.
- Settings: `--mode warm`, `threads=1`, 50 iterations.

Use these numbers only to compare quants with each other. They do not represent
Graviton or Lambda performance.

| Scenario | Q8_0 p50 | Q8_0 p95 | Q5_K_M p50 | Q5_K_M p95 |
|---|---|---|---|---|
| `single/short` | 10.5 | 11.0 | 21.4 | 33.6 |
| `single/medium` | 18.4 | 19.5 | 32.6 | 33.1 |
| `single/long` | 466 | 480 | 741 | 748 |
| `batch/medium/8` | 146 | 147 | 261 | 283 |

On that machine Q8_0 was about 1.7–1.8× faster than Q5_K_M. The scenario set changed
afterwards: `single/short` no longer exists, and the medium, long and batch texts have come
from `scripts/benchmark_corpus.json` since #153 (2026-07-19).
