# Architecture

How `main` turns text into an embedding, and why it is built this way. For the bundle files
see [bundle-format.md](./bundle-format.md); for build setup see
[development.md](./development.md).

## Components

```
EmbeddingEngine                      src/engine/mod.rs
├── ModelSpec      (build-info.json)  src/engine/bundle.rs
├── EngineConfig   (caller)           src/engine/config.rs
└── Box<dyn EmbeddingBackend>         src/engine/backend.rs   (pub(crate))
    └── LlamaBackend (model.gguf)     src/engine/llama/       (only implementation)
        └── raw FFI → static llama.cpp / ggml archives (build.rs)
```

- **`EmbeddingEngine`** is the public entry point. It owns the shared pipeline: prefixing,
  the input length limit, Matryoshka truncation and normalization.
- **`EmbeddingBackend`** is a crate-private trait. A backend tokenizes text with the
  model's own vocabulary (`tokenize`), and for tokenized inputs returns one raw,
  un-normalized, pooled vector of `raw_embedding_dimension` floats per input (`embed`).
  `LlamaBackend` is the only implementation; it tokenizes with the GGUF's vocabulary. The
  trait exists so that another backend could be added without touching the shared
  pipeline; there is no plan in the repository to add one.
- **llama.cpp** is not a crates.io dependency. `build.rs` links the prebuilt static archives
  from [`Lychee-Technology/static-llama-cpp-rs-builder`](https://github.com/Lychee-Technology/static-llama-cpp-rs-builder)
  (`libllama`, `libggml`, `libggml-cpu`, `libggml-base`, plus `stdc++`, `pthread`, `m`,
  `dl`; no OpenMP) and exposes the release's generated `bindings.rs` to
  `src/engine/llama/ffi.rs`. The pinned release is `v0.1.160-1` (llama.cpp `v0.7.0`,
  artifact contract `7`), which ships an `aarch64-graviton2` and an `x86_64-v3` variant.
  No single file holds the whole pin: the workflows set the tag, the llama.cpp commit and
  one SHA-256 per variant; `fetch-static-llama.sh` picks the variant for the runner's
  architecture and checks the contract version, variant and commit; `.githooks/pre-push`
  checks the contract version and target of the copy it builds against. See
  [development.md](./development.md#static-llamacpp-artifacts) before bumping it.

## Model

The target model is [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)
(`google/embeddinggemma-2`, 271M parameters, Apache-2.0), run from the text-only GGUF
`embeddinggemma-2-UD-Q5_K_XL.gguf` in
[`unsloth/embeddinggemma-2-GGUF`](https://huggingface.co/unsloth/embeddinggemma-2-GGUF).
The GGUF declares `general.architecture = gemma-embedding2`: a bidirectional Gemma encoder
with a hidden width of 512, mean pooling over every token, and a dense projection of the
pooled vector to 768 dimensions. The model card gives it an 8,192-token context and
Matryoshka widths of 768, 512, 256 and 128.

LTEmbed returns the projected vector, never the hidden state. llama.cpp applies the
projection inside its pooling step, so the pooled buffer that `llama_get_embeddings_seq`
returns is already 768 wide (`n_embd_out`), while `n_embd` stays 512.

Public constants in `ltembed::engine` describe this model:

| Constant | Value |
|---|---|
| `RAW_EMBEDDING_DIMENSION` | `768` (the projected width) |
| `EMBEDDING_DIMENSION` | `512` (the `EngineConfig::default()` output size) |
| `MAX_LENGTH` | `8192` |
| `QUERY_PREFIX` | `"task: search result \| query: "` |
| `DOCUMENT_PREFIX` | `"title: none \| text: "` |

The prefixes are the model's own `query` and `document` prompts. At runtime the engine does
**not** read the prefix, dimension or length constants. It uses the values in the bundle's
`build-info.json`, except that `max_length` must lie in `1..=MAX_LENGTH`. The constants
match the bundles that `scripts/assemble_gguf_bundle.py` writes, which a Python test checks.

The previous target, `jinaai/jina-embeddings-v5-text-nano-retrieval`, is no longer
supported; its vectors are not comparable with these. See
[model-migration.md](./model-migration.md).

## Load path

`EmbeddingEngine::from_gguf_bundle_dir(dir, config)` (1 thread) and
`from_gguf_bundle_dir_with_threads(dir, config, n_threads)` do the following, in order. Any
failure returns `LTEmbedError::ModelLoad(_)`:

1. Require `model.gguf` and `build-info.json` in `dir` (`MissingFile`). A `tokenizer.json`
   in `dir` is not read.
2. Parse and validate `build-info.json`, including `pooling = "mean"` and
   `1 <= max_length <= 8192` (see [bundle-format.md](./bundle-format.md#build-infojson)).
3. Validate `EngineConfig`: `0 < output_dimension <= raw_embedding_dimension` (`Config`).
4. Validate `n_threads > 0` (`Config`).
5. Load the GGUF on CPU (`n_gpu_layers = 0`) and check (`Runtime` on failure):
   - `n_embd_out`, the length of each pooled vector, equals `raw_embedding_dimension`, and
     `n_embd` is a valid width. For EmbeddingGemma 2 these are 768 and 512; a bundle that
     declared 512 would read the first 512 projected values, so it is rejected.
   - `<architecture>.pooling_type` in the GGUF is mean (1).
   - The GGUF's vocabulary tokenizes a fixed check text (`SPECIAL_TOKENS_CHECK_TEXT`) to ids
     that start with its BOS and end with its EOS token. This rejects a GGUF whose tokenizer
     metadata does not add both; it does not show that the vocabulary splits text like the
     model's own tokenizer (see [bundle-format.md](./bundle-format.md#tokenization)).

   It then measures the vocabulary's longest token text in bytes (93 for this GGUF), which
   sets the [input byte bound](#inputs-over-the-byte-bound). That is not a check and cannot
   fail.
6. Create one llama.cpp context: `embeddings = true`, `LLAMA_POOLING_TYPE_MEAN`,
   `LLAMA_ATTENTION_TYPE_NON_CAUSAL`, flash attention `ENABLED`,
   `n_ctx = n_batch = n_ubatch = max_length`, `n_seq_max = 1`,
   `n_threads = n_threads_batch = n_threads`; then check that the context reports mean
   pooling. The attention type has no effect: `llama_encode` always runs non-causal
   attention, and the GGUF declares it too. The engine sets it to state what the model
   needs.

llama.cpp's process-wide backend is initialized once and never freed. Its log callback drops
INFO/DEBUG lines and forwards WARN and above to stderr. Loading this GGUF prints three
warnings that need no action: two `control-looking token` lines (for `</s>` and
`<|tool_response>`) and `'</s>' is a normal token here, removing it from EOG list`. They
concern llama.cpp's token types. Those also decide which special-token text tokenization
matches, and the [tokenizer parity tests](./testing.md#tokenizer-parity) show that the
GGUF's vocabulary gives the same ids as the model's `tokenizer.json` on their corpus.

Flash attention is stated rather than left to llama.cpp's automatic choice, which also
enables it on CPU. It stores K and V in F16; the residual stream stays F32. With the BF16
GGUF, LTEmbed stays within cosine 1e-5 of the F32 reference on both architectures with flash
attention on or off, so it costs no measurable precision, and it halves the compute buffer
and the latency of an 8192-token input.

## Inference path

`embed(input)` is `embed_batch(&[input])`. `embed_batch(&inputs)`:

1. **Size check.** An input whose prefixed form would be longer than the
   [byte bound](#inputs-over-the-byte-bound) fails the call with
   `InputTooLong { tokens: None, .. }` before any input is prefixed or tokenized.
2. **Prefix.** `EmbeddingInput::query(text)` becomes `query_prefix + text`, and
   `EmbeddingInput::document(text)` becomes `document_prefix + text`, once. Callers pass raw
   text; the engine adds the prefix.
3. **Tokenize** each input with the GGUF's vocabulary (`llama_tokenize`, special tokens
   added and special-token text matched), without taking the backend's lock
   ([Concurrency](#concurrency)). `<bos>` and `<eos>` are added; nothing is padded.
   Nothing is truncated: if any input exceeds `max_length` tokens, the call returns
   `LTEmbedError::InputTooLong` and embeds none of the batch.
4. **Backend.** Under one `Mutex`, for each input in order:
   - clear the KV memory;
   - run `llama_encode` on that single sequence, every token marked as an output;
   - copy the pooled, projected vector from `llama_get_embeddings_seq`:
     `raw_embedding_dimension` floats, the buffer's `n_embd_out` length that load step 5
     checked.

   Inputs are encoded **one at a time**. A batch call saves call overhead, not model
   compute; multi-sequence batching is not implemented. A batch's outputs are
   bit-identical to embedding each input alone, and repeated calls are bit-identical.
5. **Check** that every raw value is finite. A NaN or infinity fails the call with
   `Inference(Backend)` rather than being normalized into a vector of NaNs.
6. **Truncate** each raw 768-d vector to its first `config.output_dimension` values
   (Matryoshka).
7. **Normalize** to unit L2 norm if `config.l2_normalize`, with the norm accumulated in
   `f64`. An all-zero vector is returned unchanged.

Truncation comes before normalization, so a 512-wide output has unit norm, as the model card
prescribes. With `l2_normalize = false` the caller gets the raw prefix of the projected
vector, whose norm is not 1.

An empty slice returns an empty `Vec` without touching the backend.
`embed_batch_profiled` returns the same embeddings plus an `EmbedBatchProfile`, which holds
per-stage milliseconds. The benchmark binary prints it when `LTEMBED_PROFILE=1`.

## Long inputs and memory

The memory an input needs grows with its length. Peak resident memory
(`/usr/bin/time -v`) of one fresh process that loads the release bundle (`UD-Q5_K_XL`,
`max_length` 8192, flash attention on) and embeds one golden input, with static release
`v0.1.159-3` and, for comparison, `v0.1.159-2`:

| Input tokens | x86-64-v3 | aarch64 | x86-64-v3, `v0.1.159-2` | aarch64, `v0.1.159-2` |
|---|---|---|---|---|
| 14 | 0.32 GiB | 0.41 GiB | 0.32 GiB | 0.41 GiB |
| 1024 | 0.45 GiB | 0.54 GiB | 1.47 GiB | 1.56 GiB |
| 4074 | 0.81 GiB | 0.90 GiB | 4.75 GiB | 4.84 GiB |
| 8192 | 1.38 GiB (1.48 GB) | 1.47 GiB (1.58 GB) | 9.38 GiB (10.1 GB) | 9.47 GiB (10.2 GB) |

- x86-64-v3: AMD Ryzen 5 5600G, Fedora Linux (kernel 7.2), 6 threads.
- aarch64: GitHub's `ubuntu-24.04-arm` runner, 4 Neoverse-N2 vCPUs and 16 GB, Ubuntu
  (kernel 6.17), 4 threads
  ([run](https://github.com/Lychee-Technology/LTEmbed/actions/runs/38688544781)).
- One thread gives the same peaks. Both releases return bit-identical embeddings for every
  input on each architecture.
- The pinned `v0.1.160-1` (llama.cpp `v0.7.0`) was measured the same way on x86-64-v3
  only. Its peaks are within 0.001 GiB of `v0.1.159-3`'s at every length, and the
  embeddings of all 25 golden inputs are bit-identical.

That is about 0.32 GiB (aarch64: 0.41 GiB) plus 0.13 MiB per token of the longest input
embedded so far. By llama.cpp's own account of this context, it goes to:

- **the model and runtime**, which is what a short input needs. Most of it is the
  memory-mapped GGUF;
- **the compute buffer**, nearly all of the growth with length. llama.cpp sizes it at load
  for `n_ubatch`, which is `max_length` (1104 MiB for 8192 tokens), but only the part an
  input uses becomes resident;
- **the output buffer**, 768 floats (3 KiB) per input token: 24 MiB at 8192.

`v0.1.159-2` (llama.cpp `b11577`) and earlier also reserved, on every `llama_encode`, room
for vocabulary logits for every output token (`n_vocab × n_tokens` floats), although a
pooled embedding model produces none. With 262144 tokens in the vocabulary that is 1 MiB
per input token, 8 GiB at 8192, and the whole difference between the columns. Since
llama.cpp `b11630` (upstream
[#30255](https://github.com/ggml-org/llama.cpp/pull/30255) and
[#30445](https://github.com/ggml-org/llama.cpp/pull/30445)), a context with pooling
reserves no host logits and `llama_get_logits` returns `NULL`; LTEmbed reads only the
pooled embedding (`llama_get_embeddings_seq`).

On the aarch64 runner at 4 threads with `v0.1.159-3`, the first call took 0.06, 2.4, 15.2
and 44.7 s for the four lengths, and later calls of the same length 0.03, 2.3, 15.1 and
44.5 s; at 1 thread the 8192-token input took 174 s. Under `v0.1.159-2` warm calls took the
same time to within 3%, and first calls up to 1.6 s longer, because they also allocated and
zeroed the logits. Loading took 0.6–0.8 s. On x86-64-v3 at 6 threads, the median warm call
under `v0.1.160-1` takes 40% less time than under `v0.1.159-3` at 14 tokens (15 instead of
25 ms), 12% less at 89 and 1% less at 1024, but 3% more at 4074 and 6% more at 8192 (16.8
instead of 15.8 s). llama.cpp `v0.7.0` again converts single FP16 values to FP32 on x86
through a lookup table instead of F16C (upstream
[#30343](https://github.com/ggml-org/llama.cpp/pull/30343)), the likely cause; it was not
isolated. [benchmarking.md](./benchmarking.md#recorded-results) has warm
latencies on x86-64-v3 and a Raspberry Pi 5, and how much long-input times vary between
processes. Three consequences:

- **An engine keeps its peak.** llama.cpp does not shrink these buffers, and the pages an
  input made resident stay resident until the engine is dropped. One engine that embeds
  8192, 14, 8192 and 1024 tokens in turn holds 1.38 GiB (aarch64: 1.47 GiB) from the first
  input on, and the later inputs add nothing (`v0.1.159-2`: 9.38 and 9.48 GiB).
- **`max_length` bounds it.** A bundle's `max_length` (`1..=8192`) sets
  `n_ctx = n_batch = n_ubatch` and the input token limit, and with them the compute buffer
  llama.cpp reserves: 110 MiB for 1024 tokens, 228 MiB for 2048, 1104 MiB for 8192. Inputs
  over it fail with `InputTooLong` instead of being truncated, so an engine never needs
  more than its peak for an input of `max_length` tokens. On x86-64-v3 that is 0.34 GiB for
  `max_length` 512, 0.40 GiB for 1024, 0.51 GiB for 2048, 0.77 GiB for 4096 and 1.38 GiB
  for 8192 (aarch64 was not measured this way; above, it peaks 0.09 GiB higher at every
  length). Inputs that fit get bit-identical embeddings under any `max_length`, and a
  smaller one lowers their peak slightly: the 1024-token input peaks at 0.40 GiB with
  `max_length` 1024, 0.41 GiB with 2048 and 0.45 GiB with 8192.
- **One sequence per micro-batch.** llama.cpp's encoder cannot split a sequence across
  micro-batches (it asserts `n_ubatch >= n_tokens`), so `n_ubatch` must equal the context
  length. LTEmbed checks the length before encoding, so the assertion cannot fire.

Latency grows faster than linearly with length, because attention is quadratic. Flash
attention keeps the attention's compute buffer small: for 8192 tokens llama.cpp reserves
1104 MiB with it and 2128 MiB without. Measured with llama.cpp's own `llama-embedding`
(`b11630`) on the same GGUF and settings at 6 threads on the x86-64-v3 host, turning it off
raises the 8192-token input's peak RSS from 1.63 GiB to 2.63 GiB and its run time, load
included, from 23.2 s to 45.1 s (4074 tokens: 1.07 to 1.48 GiB, 8.7 to 12.7 s). The tool
peaks higher than LTEmbed because its context, made for 256 sequences, reserves a 257 MiB
output buffer. Other jobs were loading the host during these runs; with `b11577` on the
idle host the run times were 20.5 and 39.2 s (4074 tokens: 8.6 and 13.2 s).

### Inputs over the byte bound

The engine counts an input's tokens only after tokenizing all of it, and tokenizing costs
time and memory in proportion to the text's size: llama.cpp builds the pre-tokenizer's
pieces and the whole token vector before `llama_tokenize` returns, about 30 to 140 bytes per
input byte at peak in the runs below, and the Gemma 4 pre-tokenizer splits only at
newlines, so one piece can be the whole input. A text that cannot fit is therefore rejected
before it is tokenized:

- At load, the backend measures the vocabulary's longest token text: 93 bytes for this
  GGUF, a run of 31 `▁` (U+2581), which an input of literal `▁` characters fills. The bound
  is `max_length` times that, 761,856 bytes for `max_length` 8192, and at most
  `i32::MAX - 2` bytes, the longest text `llama_tokenize` takes.
- `embed_batch` first compares each input's size with its prefix against the bound. If one
  is over it, the call returns `InputTooLong { tokens: None, bytes, max }` before any input
  of the batch is prefixed or tokenized: there is no token count to report. Otherwise the
  inputs are tokenized as before, and one of more than `max_length` tokens returns
  `InputTooLong { tokens: Some(count), bytes, max }`.

The bound rejects no text that fits, because this tokenizer never drops or contracts input.
llama.cpp (`llama_vocab::impl::tokenize` in `src/llama-vocab.cpp`, `v0.7.0`) turns
special-token text into its token, replaces each space in the rest with `▁` (3 bytes),
splits it at newlines without losing a character, BPE-merges each piece, and spells a
merged piece that is not a token as one `<0xXX>` byte token per byte. Every input byte so
ends up in exactly one token whose text is at least as long as the input it covers, and a
text of n tokens has at most n × 93 bytes. Two llama.cpp mechanisms could drop input, and
neither applies: a token with the `LSTRIP` or `RSTRIP` attribute removes the whitespace
beside its text, but llama.cpp sets those only for some other models, by name; and a byte
without a `<0xXX>` token is skipped, but this vocabulary has all 256. The loader checks
neither, just as it does not check that the vocabulary splits text like the model's own
tokenizer: with a GGUF whose tokenization drops input, the bound could reject a text that
fits. For the pinned GGUF, a document of 8183 copies of the longest token, 8192 tokens and
761,039 bytes, embeds, and the [tokenizer parity](./testing.md#tokenizer-parity) corpus fits
the bound at its own token count, at most 90.2 bytes per token.

Rejecting 64 MiB inputs on x86-64-v3 (the Ryzen 5 5600G above), 1 thread, `max_length`
8192, before (`59c6aee`) and with the bound. Each run is a fresh process that loads the
release bundle, builds one document and embeds it; the peak includes the engine (0.32 GiB)
and the input string itself:

| 64 MiB of | Before | With the bound |
|---|---|---|
| `abcdefghijklmnopqrstuvwxyz` repeated, no spaces | 78.2 s, 9.15 GiB | under 0.01 ms, 0.35 GiB |
| `lorem ipsum dolor sit amet ` repeated | 67.4 s, 5.70 GiB | under 0.01 ms, 0.35 GiB |
| `中文测试文本` repeated | 8.3 s, 2.11 GiB | under 0.01 ms, 0.35 GiB |
| `a\n` repeated | 8.3 s, 6.06 GiB | under 0.01 ms, 0.35 GiB |

With the bound, the call adds nothing to the peak of building the input without embedding
it. A text at the bound is still tokenized, so it sets what a rejected input can cost: the
same four kinds at 761,856 bytes with the prefix took 0.57, 0.54, 0.06 and 0.06 s to
reject, and peaked at 0.39 GiB at most. Ordinary text holds far fewer bytes per token than
93, so it reaches 8192 tokens long before the bound: these inputs had 87,915, 141,089,
126,982 and 761,844 tokens. The engine's own buffer for the token ids, `text.len() + 2` of
them, is now at most 2.9 MiB.

## Concurrency

`EmbeddingEngine` holds one llama.cpp context behind a `Mutex`. `llama_encode` writes to
the context, so concurrent calls on a shared engine encode their inputs one at a time. They
tokenize without the mutex, so a thread tokenizing a long input does not hold up another
thread's inference, and inference does not hold up tokenizing. If the mutex is poisoned,
calls that reach inference return `InferenceError::MutexPoisoned`. For parallel inference,
either load several engines or raise `n_threads` to parallelize inside each call.

Tokenizing needs no lock because, in the pinned llama.cpp (`v0.7.0`), `llama_tokenize` only
reads state that several threads share:

- It calls the `const` `llama_vocab::tokenize` (`src/llama-vocab.cpp`). Only
  `llama_vocab::impl::load` writes the vocabulary, while the model loads.
- Each call builds its own working state as locals: the token vector, the special-token
  fragments and the BPE session.
- The Gemma 4 pre-tokenizer splits at newlines without `std::regex`. The only statics on
  the path are `const` tables behind thread-safe initialization guards, and the load-time
  BOS/EOS check initializes them before any other thread can tokenize.
- `llama_encode` holds the model as a `const llama_model &` and only reads the vocabulary.

`tokenize_with_vocab` in `src/engine/llama/mod.rs` records the argument with its
references. Upstream's server relies on the same, tokenizing requests on its HTTP threads
while its main loop runs inference. A ThreadSanitizer build of llama.cpp `620d222` reported
no race over 4,320 `llama_tokenize` calls on 8 threads made while another thread ran
`llama_encode` with LTEmbed's context settings, and every call returned the single-thread
token ids. The argument holds for this llama.cpp only: recheck it when the static release
changes ([development.md](./development.md#static-llamacpp-artifacts)).
`test_concurrent_calls_match_a_single_thread` checks that 8 threads sharing an engine get
exactly the results one thread gets
([testing.md](./testing.md#integration-tests-tier-1-and-tier-2)).

Latency of one thread's calls on an engine that a second thread also uses, before
(`5b8ca0a`, tokenizing under the mutex) and now, on x86-64-v3 (the Ryzen 5 5600G above), 1
llama.cpp thread, `max_length` 8192. The measured thread sleeps 100 ms before each of its
calls; the other thread calls back to back. The builds alternated over two runs; Before
and Now give each run, Alone the range over all four:

| Measured thread | Other thread | Alone | Before | Now |
|---|---|---|---|---|
| 60 short queries (`hello`, 10 tokens with the prompt) | rejects a document of 761,856 bytes with its prefix and no spaces: tokenizes it, 0.6 s, then returns `InputTooLong` | median 52–55 ms | median 501, 501 ms; max 1.05, 1.05 s | median 51, 54 ms; max 56, 57 ms |
| 30 rejections of `hello world ` × 12,000 as a document (about 24,000 tokens) | embeds a 459-token document, 1.6 s | median 80–84 ms | median 4.9, 8.2 s; max 168, 252 s | median 81, 88 ms; max 84, 95 ms |

Before, the measured thread waited for the other thread's tokenizing or encoding to finish,
and sometimes longer: a thread that releases a `std::sync::Mutex` and at once locks it
again can keep it from a waiting thread. Now neither waits for the other, and the measured
thread's latency is the same as alone.

## Errors

| Variant | When |
|---|---|
| `ModelLoad(MissingFile)` | A bundle file is missing |
| `ModelLoad(UnsupportedModelFormat / UnsupportedInputKind / UnsupportedPooling)` | `build-info.json` describes an incompatible model |
| `ModelLoad(Metadata)` | `build-info.json` cannot be read or parsed, or `max_length` is outside `1..=8192` |
| `ModelLoad(Config)` | Bad `output_dimension` or `n_threads = 0` |
| `ModelLoad(Runtime)` | GGUF load failure, `n_embd` or `n_embd_out` mismatch, a GGUF that does not declare mean pooling or whose tokenization does not add BOS and EOS, context creation failure |
| `InputTooLong { tokens, bytes, max }` | An input is longer than `max_length` tokens. `bytes` is its size with the prefix; `tokens` is its token count, or `None` if it was over the [byte bound](#inputs-over-the-byte-bound) and not tokenized |
| `Tokenization(_)` | The input cannot be tokenized with the GGUF's vocabulary |
| `Inference(Backend / Tensor / OutputShape / MutexPoisoned / Internal)` | Backend-side failures; `Backend` wraps a non-zero `llama_encode` return code or a raw vector with a NaN or infinite value |

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
- Until the EmbeddingGemma 2 migration, the GGUF backend ran
  `jinaai/jina-embeddings-v5-text-nano-retrieval` (last-token pooling, `Query: ` /
  `Document: ` prefixes, Q5_K_M). Its bundle and results are archived in
  [history/models/jina-embeddings-v5-text-nano-retrieval.md](./history/models/jina-embeddings-v5-text-nano-retrieval.md).
- The documents behind these decisions are in [history/](./README.md#history).
