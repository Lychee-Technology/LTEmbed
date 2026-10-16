# GGUF Bundle Format

This is the reference for the directory that `EmbeddingEngine::from_gguf_bundle_dir` loads.
It is checked against `src/engine/bundle.rs`, `src/engine/mod.rs`,
`src/engine/llama/mod.rs` and `scripts/assemble_gguf_bundle.py`, which every workflow uses
to build its bundle.

## Layout

```
gguf_bundle/
├── model.gguf        required
├── build-info.json   required
├── SHA256SUMS        written by assemble_gguf_bundle.py; not read by the runtime
└── provenance.json   written by assemble_gguf_bundle.py; not read by the runtime
```

There is no `tokenizer.json`: the engine tokenizes with the GGUF's own vocabulary (see
[Tokenization](#tokenization)). A `tokenizer.json` left in the directory, for example by an
older bundle, is not read. The directory name is up to you. `examples/api_usage.rs` and the docs use `gguf_bundle/`.
No tool in this repository converts models: `scripts/assemble_gguf_bundle.py` downloads
published files from pinned Hugging Face revisions and checks their hashes.

## `model.gguf`

A text-only GGUF of [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) from
[`unsloth/embeddinggemma-2-GGUF`](https://huggingface.co/unsloth/embeddinggemma-2-GGUF),
pinned at revision `031f0d4b35536f69ab3509d4893c923264fcf253`. The repository publishes one
file per quant as `embeddinggemma-2-<QUANT>.gguf`, which the script renames to `model.gguf`.

| Quant | File size | sha256 | Used by |
|---|---|---|---|
| `UD-Q4_K_XL` | 175,673,856 B | `ea905fd08e8061db77a0031cbb5096d459d7cfd2bdd0ad2ba66e483f16719493` | benchmarks |
| `UD-Q5_K_XL` | 210,055,680 B | `a9d7a3b7eb421113f05e964e10882676f10fff83811b7876d5bb983db7c7ac41` | CI, release, golden gates |
| `UD-Q6_K_XL` | 248,812,032 B | `dc84f042bc3ffbe6122b3b38b2fb299d66f5f37b276c9d458db85d4a4f1e6ba3` | benchmarks |
| `Q8_0` | 309,855,520 B | `6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc` | benchmarks |
| `BF16` | 557,950,240 B | `f315cbbb30dd487e44d501c8902abe88808755e43753a96beed1964f0a48aa4f` | benchmarks, precision checks |

The hashes are those Hugging Face records for each file at the pinned revision. The
repository's `F16` GGUF is not pinned: nothing here has measured it, and the model card does
not support float16. Its `mmproj-*.gguf` files are the vision and audio projectors; a text
bundle never contains one, and the release workflow fails if one appears.

The pinned GGUFs declare `general.architecture = gemma-embedding2`, a hidden width
(`embedding_length`) of 512, a projected output width (`embedding_length_out`) of 768,
`pooling_type = 1` (mean), non-causal attention, and `add_bos_token`/`add_eos_token`. The
loader checks, failing with `ModelLoad(Runtime)`:

- the GGUF's pooled output width (`n_embd_out`) equals `raw_embedding_dimension` from
  `build-info.json`, and its hidden width (`n_embd`) is positive. A bundle that names the
  hidden width 512 as its raw dimension is rejected, rather than returning the first 512
  values of the projected vector;
- the GGUF declares `<architecture>.pooling_type` as mean. A GGUF that declares another
  pooling, or none, is rejected: the backend always pools with mean;
- the GGUF's tokenization adds BOS and EOS (see [Tokenization](#tokenization));
- the llama.cpp context reports mean pooling once created.

The engine needs llama.cpp support for `gemma-embedding2`. The pinned static release
(`v0.1.160-1`, llama.cpp `v0.7.0`) has it, as did `v0.1.159-3` (llama.cpp `b11630`) and
`v0.1.159-2` (llama.cpp `b11577`); `v0.1.159-1` (llama.cpp `d8123504`) does not know the
architecture.

## Tokenization

The engine tokenizes every input with the GGUF's own vocabulary (`tokenizer.ggml.*`, model
`gemma4`) through llama.cpp's `llama_tokenize`, as upstream llama.cpp's embedding tool does:
special tokens are added and special-token text in the input is matched. The vocabulary has
262144 tokens; tokenization adds `<bos>` (id 2) before and `<eos>` (id 1) after every input
and never pads or truncates. The library does not depend on the `tokenizers` crate, and
neither the bundle nor the release archive contains a `tokenizer.json`.

- The loader makes one tokenization check, failing with `ModelLoad(Runtime)`: a short fixed
  text (`SPECIAL_TOKENS_CHECK_TEXT` in `src/engine/llama/mod.rs`) must tokenize to ids that
  start with the vocabulary's BOS and end with its EOS. The model's own pipeline wraps every
  input in both and mean pooling averages over them, so a GGUF whose
  `tokenizer.ggml.add_eos_token` (or `add_bos_token`) is false would embed every input
  differently. The message names the missing token. (For this `gemma4` tokenizer llama.cpp
  overrides `add_bos_token` to true, so only a missing EOS can occur with these GGUFs.)
- The check does not show that the vocabulary splits text the way the model's
  SentenceTransformers pipeline does. That is shown by tests, for the pinned GGUF only: the
  model's `tokenizer.json` from
  [`unsloth/embeddinggemma-2`](https://huggingface.co/unsloth/embeddinggemma-2) at revision
  `c8f89cc7ecb4ba5122cd15fdd487422b2b487daf` (Unsloth's safetensors copy of the model; the
  GGUF repository has no `tokenizer.json`), about 32 MB with sha256
  `4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4`, gives the same token
  ids as the GGUF's vocabulary on a fixed corpus of about 25,000 texts, and the GGUF's
  vocabulary matches the token-id and llama.cpp reference fixtures
  ([testing.md](./testing.md#tokenizer-parity)). A GGUF of another model, or one converted
  with a different vocabulary, loads if it passes the checks above and produces wrong
  vectors.
- `scripts/assemble_gguf_bundle.py --reference-tokenizer PATH` writes that pinned
  `tokenizer.json` to `PATH`, outside the bundle, for those tests.
- The loader also measures the vocabulary's longest token text: 93 bytes, a run of 31 `▁`.
  No text of more than `max_length` times that many bytes, prefix included, fits in
  `max_length` tokens, so the engine rejects such a text with `InputTooLong` without
  tokenizing it. This relies on the `gemma4` tokenization never dropping input; the loader
  does not check that, and with a GGUF whose tokenization drops input the bound could reject
  a text that fits ([architecture.md](./architecture.md#inputs-over-the-byte-bound)).

## `build-info.json`

The engine reads only the fields below. Unknown fields such as `quant` are ignored.

```json
{
  "target_id": "unsloth/embeddinggemma-2-GGUF",
  "model_metadata": {
    "model_format": "gguf",
    "quant": "UD-Q5_K_XL",
    "pooling": "mean",
    "input_kind": "retrieval",
    "query_prefix": "task: search result | query: ",
    "document_prefix": "title: none | text: ",
    "raw_embedding_dimension": 768,
    "max_length": 8192
  }
}
```

This is exactly what `scripts/assemble_gguf_bundle.py` writes for the default quant; for
another quant only `quant` changes. Bundles assembled by earlier versions of the script also
have `model_metadata.output_embedding_dimension`. The engine ignores it, like any other
unknown field: the output size comes from `EngineConfig`.

| Field | Required | Rule | Used for |
|---|---|---|---|
| `target_id` | yes (string) | — | Error messages only |
| `model_metadata.model_format` | no | If present, must be `"gguf"`; else `UnsupportedModelFormat`. A missing field is accepted for older minimal bundles. | Rejecting ONNX-era bundles |
| `model_metadata.pooling` | yes | `"mean"`; else `UnsupportedPooling` | Validation only; the backend always uses mean pooling |
| `model_metadata.input_kind` | yes | `"retrieval"` or `"text"`; else `UnsupportedInputKind` | Validation only |
| `model_metadata.query_prefix` | yes (string) | — | Prepended to `EmbeddingInput::query` text |
| `model_metadata.document_prefix` | yes (string) | — | Prepended to `EmbeddingInput::document` text |
| `model_metadata.raw_embedding_dimension` | yes (integer) | Must equal the GGUF `n_embd_out` | Upper bound for `EngineConfig::output_dimension`; backend output length |
| `model_metadata.max_length` | yes (integer) | `1..=8192` (`MAX_LENGTH`, the model's context length); else `Metadata` | Input token limit (`InputTooLong`), the input byte bound (`max_length` × the vocabulary's longest token, 93 bytes) and llama.cpp context size |
| `model_metadata.quant` | no | — | **Not read.** Informational. |

If the file is missing, the engine returns `ModelLoad(MissingFile)`. If it cannot be read or
parsed, or a required field is missing, the engine returns `ModelLoad(Metadata)`.

The prefixes are read from this file, not from the constants in `src/engine/mod.rs`, and
are not compared with them. They must be EmbeddingGemma 2's own prompts, written above:
with any other prefix the bundle loads and returns vectors that do not match the model's
training (see [testing.md](./testing.md#how-sensitive-the-gates-are)).

`max_length` below 8192 is a supported way to bound the memory a long input can take; see
[architecture.md](./architecture.md#long-inputs-and-memory).

## `SHA256SUMS` and `provenance.json`

`scripts/assemble_gguf_bundle.py` writes both after every other check has passed:

- `SHA256SUMS`: the sha256 of `model.gguf` and `build-info.json`.
- `provenance.json`: for each file, the Hugging Face repository, revision, file name, URL
  and sha256, plus the upstream model (`google/embeddinggemma-2`), its license
  (`apache-2.0`), the quant, the LTEmbed commit that assembled the bundle and, in GitHub
  Actions, the workflow run. The license text itself is not a bundle file: it is in
  [`licenses/embeddinggemma-2/`](../licenses/embeddinggemma-2/NOTICE.md), which release
  tarballs ship next to the bundle.

To check a downloaded bundle:

```bash
cd gguf_bundle && sha256sum -c SHA256SUMS
```

## Assembling a bundle

```bash
python3 scripts/assemble_gguf_bundle.py gguf_bundle                 # UD-Q5_K_XL
python3 scripts/assemble_gguf_bundle.py gguf_bundle-q8 --quant Q8_0
# Also the tests' reference tokenizer.json, outside the bundle:
python3 scripts/assemble_gguf_bundle.py gguf_bundle --reference-tokenizer reference-tokenizer.json
# Copies already on disk are checked against the same pins instead of downloaded:
python3 scripts/assemble_gguf_bundle.py gguf_bundle --gguf PATH \
  --reference-tokenizer reference-tokenizer.json --tokenizer PATH
```

The script needs only Python 3. It downloads from `resolve/<revision>` URLs, never from a
branch, rejects a file whose sha256 differs from its pin, and requires the destination to be
absent or empty. It assembles in a temporary directory and moves the files into place only
after every check passed, so a failed run leaves no partial bundle. The reference tokenizer
replaces an existing file at `PATH` only once it matches its pin. CI, the release workflow and
the ARM64 benchmark all call the script, so they use the same files.

`/gguf_bundle/`, `/reference-tokenizer.json` and `*.gguf` are in `.gitignore`. A bundle under
another name is not ignored except for its GGUF, so keep it out of commits.

## Bundles for the previous model

Bundles built for `jinaai/jina-embeddings-v5-text-nano-retrieval` do not load: their
`build-info.json` declares `"pooling": "last_token"` (`UnsupportedPooling`), and their GGUF
declares last-token pooling and a 128256-token vocabulary. Embeddings from that model are
not comparable with EmbeddingGemma 2's; see [model-migration.md](./model-migration.md). Its
bundle details are kept in
[history/models/jina-embeddings-v5-text-nano-retrieval.md](./history/models/jina-embeddings-v5-text-nano-retrieval.md).

## ONNX-era bundles

Bundles with `model.ort`, `libonnxruntime.so` or `model_format` other than `gguf` belong
to the `ort` branch. `main` rejects them at load time; see
[history/ort/](./history/ort/lambda-s3-files-deployment.md).
