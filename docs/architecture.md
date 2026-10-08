# Architecture

How `main` turns text into an embedding, and why it is built this way. For the bundle files
see [bundle-format.md](./bundle-format.md); for build setup see
[development.md](./development.md).

## Components

```
EmbeddingEngine                      src/engine/mod.rs
├── ModelSpec      (build-info.json)  src/engine/bundle.rs
├── EngineConfig   (caller)           src/engine/config.rs
├── HFTokenizer    (tokenizer.json)   src/traits/tokenizer.rs
└── Box<dyn EmbeddingBackend>         src/engine/backend.rs   (pub(crate))
    └── LlamaBackend (model.gguf)     src/engine/llama/       (only implementation)
        └── raw FFI → static llama.cpp / ggml archives (build.rs)
```

- **`EmbeddingEngine`** is the public entry point. It owns the shared pipeline: prefixing,
  tokenization, truncation and normalization.
- **`EmbeddingBackend`** is a crate-private trait. A backend receives tokenized inputs and
  returns one raw, un-normalized, pooled vector of `raw_embedding_dimension` floats per
  input. `LlamaBackend` is the only implementation. The trait exists so that another backend
  could be added without touching the shared pipeline; there is no plan in the repository to
  add one.
- **llama.cpp** is not a crates.io dependency. `build.rs` links the prebuilt static archives
  from [`Lychee-Technology/static-llama-cpp-rs-builder`](https://github.com/Lychee-Technology/static-llama-cpp-rs-builder)
  (`libllama`, `libggml`, `libggml-cpu`, `libggml-base`, plus `stdc++`, `pthread`, `m`,
  `dl`; no OpenMP) and exposes the release's generated `bindings.rs` to
  `src/engine/llama/ffi.rs`. The pinned release is `v0.1.159-1` (llama.cpp `v0.6.0`,
  artifact contract `4`), which ships an `aarch64-graviton2` and an `x86_64-v3` variant.
  No single file holds the whole pin: the workflows set the tag, the llama.cpp commit and
  one SHA-256 per variant; `fetch-static-llama.sh` picks the variant for the runner's
  architecture and checks the contract version, variant and commit; `.githooks/pre-push`
  checks the contract version and target of its local copy. See
  [development.md](./development.md#static-llamacpp-artifacts) before bumping it.

## Model

The target model is
[`jinaai/jina-embeddings-v5-text-nano-retrieval`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval)
in GGUF form. The migration spike recorded `general.architecture = eurobert` and
`n_embd = 768` for it. LTEmbed runs it as an encoder: non-causal attention, last-token
pooling.

Public constants in `ltembed::engine` describe this model:

| Constant | Value |
|---|---|
| `RAW_EMBEDDING_DIMENSION` | `768` |
| `EMBEDDING_DIMENSION` | `512` (the `EngineConfig::default()` output size) |
| `MAX_LENGTH` | `8192` |
| `QUERY_PREFIX` / `DOCUMENT_PREFIX` | `"Query: "` / `"Document: "` |

At runtime the engine does **not** read the prefix, dimension or length constants. It uses
the values in the bundle's `build-info.json`. The constants match the bundles that CI and
the release workflow write.

## Load path

`EmbeddingEngine::from_gguf_bundle_dir(dir, config)` (1 thread) and
`from_gguf_bundle_dir_with_threads(dir, config, n_threads)` do the following, in order. Any
failure returns `LTEmbedError::ModelLoad(_)`:

1. Require `model.gguf`, `tokenizer.json` and `build-info.json` in `dir` (`MissingFile`).
2. Parse and validate `build-info.json` (see [bundle-format.md](./bundle-format.md)).
3. Validate `EngineConfig`: `0 < output_dimension <= raw_embedding_dimension` (`Config`).
4. Validate `n_threads > 0` (`Config`).
5. Load `tokenizer.json` with the Hugging Face `tokenizers` crate (`Runtime` on failure).
   The BPE word cache is turned off because `tokenizers` 0.23 leaks it when a tokenizer is
   dropped.
6. Load the GGUF on CPU (`n_gpu_layers = 0`) and check that its `n_embd` and `n_embd_out`
   both equal `raw_embedding_dimension` (`Runtime` on mismatch). `n_embd_out` is the length
   of each pooled vector, which a GGUF may set apart from `n_embd`.
7. Create one llama.cpp context: `embeddings = true`, `LLAMA_POOLING_TYPE_LAST`,
   `LLAMA_ATTENTION_TYPE_NON_CAUSAL`, `n_ctx = n_batch = n_ubatch = max_length`,
   `n_seq_max = 1`, `n_threads = n_threads_batch = n_threads`.

llama.cpp's process-wide backend is initialized once and never freed. Its log callback drops
INFO/DEBUG lines and forwards WARN and above to stderr.

## Inference path

`embed(input)` is `embed_batch(&[input])`. `embed_batch(&inputs)`:

1. **Prefix.** `EmbeddingInput::query(text)` becomes `query_prefix + text`, and
   `EmbeddingInput::document(text)` becomes `document_prefix + text`. Callers pass raw
   text; the engine adds the prefix.
2. **Tokenize** the whole batch with `HFTokenizer::encode_batch`. Special tokens are added,
   and the bundle tokenizer's padding settings apply. Nothing is truncated: if any input
   exceeds `max_length` tokens, the call returns `LTEmbedError::InputTooLong`.
3. **Backend.** Under one `Mutex`, for each input in order:
   - keep only real tokens (attention mask `1`) and drop right padding;
   - clear the KV memory;
   - run `llama_encode` on that single sequence;
   - copy the pooled vector from `llama_get_embeddings_seq`: `raw_embedding_dimension`
     floats, the buffer's `n_embd_out` length that load step 6 checked.

   Inputs are encoded **one at a time**. A batch call saves tokenizer and call overhead,
   not model compute; multi-sequence batching is not implemented.
4. **Truncate** each raw 768-d vector to `config.output_dimension` (Matryoshka).
5. **Normalize** to unit L2 norm if `config.l2_normalize`. An all-zero vector is returned
   unchanged.

An empty slice returns an empty `Vec` without touching the backend.
`embed_batch_profiled` returns the same embeddings plus an `EmbedBatchProfile`, which holds
per-stage milliseconds. The benchmark binary prints it when `LTEMBED_PROFILE=1`.

### Why padding is stripped before the backend

Last-token pooling takes the hidden state of the final position fed to the model. In a
padded batch, shorter inputs end in pad tokens (pad id `128004`). Feeding them pooled the
pad position, which dropped cosine against the FP32 reference to about 0.31 and broke
cross-lingual retrieval. `TokenizerOutput::real_input_ids` removes padding so that each
sequence ends on its real last token. The unit test
`test_real_input_ids_strips_right_padding` covers this.

## Concurrency

`EmbeddingEngine` holds one llama.cpp context behind a `Mutex`. Concurrent calls on a
shared engine are safe but run one at a time. If the mutex is poisoned, calls return
`InferenceError::MutexPoisoned`. For parallelism, either load several engines or raise
`n_threads` to parallelize inside each call.

## Errors

| Variant | When |
|---|---|
| `ModelLoad(MissingFile)` | A bundle file is missing |
| `ModelLoad(UnsupportedModelFormat / UnsupportedInputKind / UnsupportedPooling)` | `build-info.json` describes an incompatible model |
| `ModelLoad(Metadata)` | `build-info.json` cannot be read or parsed |
| `ModelLoad(Config)` | Bad `output_dimension` or `n_threads = 0` |
| `ModelLoad(Runtime)` | Tokenizer or GGUF load failure, `n_embd` or `n_embd_out` mismatch, context creation failure |
| `InputTooLong { tokens, max }` | An input tokenizes to more than `max_length` tokens |
| `Tokenization(_)` | The tokenizer rejects the input |
| `Inference(SequenceTooLong / AllPadding / Backend / Tensor / OutputShape / MutexPoisoned / Internal)` | Backend-side failures; `Backend` wraps a non-zero `llama_encode` return code |

## Platform constraint

The static archives are native Linux objects, one variant per target:

| Target | Variant | CPU requirement |
|---|---|---|
| `aarch64-unknown-linux-gnu` | `aarch64-graviton2` | `armv8.2-a+fp16+dotprod+rcpc` (Graviton2 / Neoverse N1 or newer) |
| `x86_64-unknown-linux-gnu` | `x86_64-v3` | x86-64-v3 (AVX2, BMI1/2, F16C, FMA, LZCNT, MOVBE) |

The crate builds and links only for these two targets, with `STATIC_LLAMA_DIR` pointing at
the matching variant. llama.cpp has no runtime ISA dispatch in these builds, so a CPU below
the variant's baseline faults with `SIGILL`; x86-64 CPUs below v3 are not supported.
`build.rs` panics unless `STATIC_LLAMA_DIR` points at an extracted release containing
`lib/libllama.a` and `bindings.rs`. It does not check the target triple itself; the other
variant's archives fail to link. The crate defines no Cargo features, so there is nothing to
switch off.

## What was replaced

- `main` used ONNX Runtime (`OnnxEngine`, `ort_bundle`, `model.ort`,
  `libonnxruntime.so` / `ORT_DYLIB_PATH`) until #149 (2026-07-09). That line is frozen on
  the `ort` branch. The rename map was `OnnxEngine` → `EmbeddingEngine`,
  `OnnxEngineConfig` → `EngineConfig`, `InferenceError::OrtRun` → `Backend`.
- An earlier pure-Rust BERT backend built on `matrixmultiply` lives on the
  `matrixmultiply` branch.
- The documents behind these decisions are in [history/](./README.md#history).
