# GGUF Bundle Format

This is the reference for the directory that `EmbeddingEngine::from_gguf_bundle_dir` loads.
It is checked against `src/engine/bundle.rs`, `src/engine/mod.rs`,
`src/engine/llama/mod.rs` and the bundle steps in `.github/workflows/`.

## Layout

```
gguf_bundle/
├── model.gguf        required
├── tokenizer.json    required
├── build-info.json   required
└── SHA256SUMS        release tarballs only; not read by the runtime
```

The directory name is up to you. `examples/api_usage.rs` and the docs use `gguf_bundle/`.
Every bundle-producing workflow assembles the directory itself; no tool in this repository
converts models.

## `model.gguf`

- A GGUF build of `jinaai/jina-embeddings-v5-text-nano-retrieval`. The Hugging Face repo
  publishes one file per quant as `v5-nano-retrieval-<QUANT>.gguf`, which the workflows
  rename to `model.gguf`.
- Its embedding size (`n_embd`) must equal `raw_embedding_dimension` from
  `build-info.json`. Otherwise loading fails with `ModelLoad(Runtime)`.
- `Q5_K_M` is the quant used by CI and by release tarballs; see
  [releases.md](./releases.md#why-q5_k_m) for why. Benchmarks also run `IQ4_NL` and `Q8_0`.
- Reference hash: at Hugging Face revision `ac5d898c8d382b17167c33e5c8af644a3519b47d`
  (the revision CI pins), `v5-nano-retrieval-Q5_K_M.gguf` has sha256
  `46fbc0423862cb6a5d4ff776d885f349d2a87c36d821dd5630f9fa184c9b4b92` (about 169 MB).

## `tokenizer.json`

- The model's own Hugging Face tokenizer (`tokenizer.json` in the same Hugging Face repo): a
  BPE tokenizer with a vocabulary of about 128k entries and right padding with pad id
  `128004`. It is about 17 MB.
- At the pinned revision its sha256 is
  `98d4a1d32152d6cedf85b5e88f3b205106dca1fe72aaab34e0ac13c238421069`, the value recorded in
  `tests/fixtures/token_ids.json`.
- **Do not use `assets/tokenizer.json`.** That tracked file is an old 30k-vocab BERT
  WordPiece tokenizer. Unit tests use it only for tokenizer shape and length checks. With
  it, embeddings are meaningless (cosine ≈ 0 against the reference).
- The engine does **not** check that the tokenizer matches the GGUF. A wrong tokenizer
  loads without error and produces wrong vectors.

## `build-info.json`

The engine reads only the fields below. Unknown fields such as `quant` are ignored.

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

This is exactly what `.github/workflows/ci.yml` writes. `release-bundles.yml` and
`benchmark-arm64.yml` write the same object and fill `quant` from their matrix.

| Field | Required | Rule | Used for |
|---|---|---|---|
| `target_id` | yes (string) | — | Error messages only |
| `model_metadata.model_format` | no | If present, must be `"gguf"`; else `UnsupportedModelFormat`. A missing field is accepted for older minimal bundles. | Rejecting ONNX-era bundles |
| `model_metadata.pooling` | yes | `"last_token"` or `"lasttoken"`; else `UnsupportedPooling` | Validation only; the backend always uses last-token pooling |
| `model_metadata.input_kind` | yes | `"retrieval"` or `"text"`; else `UnsupportedInputKind` | Validation only |
| `model_metadata.query_prefix` | yes (string) | — | Prepended to `EmbeddingInput::query` text |
| `model_metadata.document_prefix` | yes (string) | — | Prepended to `EmbeddingInput::document` text |
| `model_metadata.raw_embedding_dimension` | yes (integer) | Must equal the GGUF `n_embd` | Upper bound for `EngineConfig::output_dimension`; backend output length |
| `model_metadata.output_embedding_dimension` | yes (integer) | — | **Not used.** The output size comes from `EngineConfig`. |
| `model_metadata.max_length` | yes (integer) | — | Tokenizer limit (`InputTooLong`) and llama.cpp context size |
| `model_metadata.quant` | no | — | **Not read.** Informational. |

If the file is missing, the engine returns `ModelLoad(MissingFile)`. If it cannot be read or
parsed, or a required field is missing, the engine returns `ModelLoad(Metadata)`.

## `SHA256SUMS` (release tarballs)

`release-bundles.yml` runs
`sha256sum model.gguf tokenizer.json build-info.json > SHA256SUMS` inside `gguf_bundle/`.
To check a downloaded bundle:

```bash
cd gguf_bundle && sha256sum -c SHA256SUMS
```

## Assembling a bundle by hand

This mirrors the CI step "Assemble GGUF bundle for model-backed tests", including its pinned
revision:

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
cat > "$BUNDLE/build-info.json" <<JSON
{
  "target_id": "$GGUF_REPO",
  "model_metadata": {
    "model_format": "gguf",
    "quant": "$QUANT",
    "pooling": "last_token",
    "input_kind": "retrieval",
    "query_prefix": "Query: ",
    "document_prefix": "Document: ",
    "raw_embedding_dimension": 768,
    "output_embedding_dimension": 768,
    "max_length": 8192
  }
}
JSON
sha256sum "$BUNDLE/model.gguf" "$BUNDLE/tokenizer.json"   # compare with the hashes above
```

`gguf_bundle/` is not in `.gitignore`, but `*.gguf` is, so the model cannot be committed by
accident. The other two files can be, so keep the bundle out of commits.

## Where each workflow gets its bundle

| Workflow | GGUF source | Tokenizer source | Pinned? |
|---|---|---|---|
| `ci.yml` (`Test`) | HF `resolve/<HF_REVISION>` | HF `resolve/<HF_REVISION>` | Yes, `ac5d898c…` |
| `release-bundles.yml` | HF `resolve/main` | HF `resolve/main` | **No** |
| `benchmark-arm64.yml` | HF `resolve/main` | `snapshot_download` of the repo (latest) | **No** |

Because of this, a release or benchmark run can pick up an upstream Hugging Face change that
CI never tested. Compare the bundle's `SHA256SUMS` with the hashes above when that matters.

## ONNX-era bundles

Bundles with `model.ort`, `libonnxruntime.so` or `model_format` other than `gguf` belong
to the `ort` branch. `main` rejects them at load time; see
[history/ort/](./history/ort/lambda-s3-files-deployment.md).
