# Migrating from Jina v5 Nano to EmbeddingGemma 2

LTEmbed now embeds with EmbeddingGemma 2 (`unsloth/embeddinggemma-2-GGUF`, `UD-Q5_K_XL`)
and no longer loads `jinaai/jina-embeddings-v5-text-nano-retrieval`. There is no dual-model
mode. This page lists what changes for an application that stored vectors from the previous
model, and how to move an index without mixing the two. The previous model's bundle and
results are archived in
[history/models/jina-embeddings-v5-text-nano-retrieval.md](./history/models/jina-embeddings-v5-text-nano-retrieval.md).

## What changed

| | Jina v5 Nano (before) | EmbeddingGemma 2 (now) |
|---|---|---|
| Bundle `target_id` | `jinaai/jina-embeddings-v5-text-nano-retrieval` | `unsloth/embeddinggemma-2-GGUF` |
| GGUF | `v5-nano-retrieval-Q5_K_M.gguf`, about 169 MB | `embeddinggemma-2-UD-Q5_K_XL.gguf`, 210 MB |
| Tokenizer | the bundle's `tokenizer.json`: 128256 tokens, right padding, about 17 MB | the GGUF's own vocabulary: 262144 tokens, no padding; the bundle has no `tokenizer.json` |
| Pooling | last token | mean over all tokens, then a 512→768 projection |
| Query prefix | `Query: ` | `task: search result \| query: ` |
| Document prefix | `Document: ` | `title: none \| text: ` |
| Raw width / default output | 768 / 512 | 768 / 512 |
| Maximum input | 8192 tokens | 8192 tokens |
| Peak memory for an 8k input | not measured | about 1.6 GB ([architecture.md](./architecture.md#long-inputs-and-memory)) |

## Rust API

The embedding API keeps its signatures: `EmbeddingEngine`, `EmbeddingInput::query` /
`EmbeddingInput::document` and `EngineConfig` work as before, and the engine still adds the
prefix itself. Code that embeds raw text only through them compiles unchanged; it needs a
new bundle, and its stored vectors must be recomputed (below). Other public items changed,
and code that names a removed one no longer compiles:

- **The `ltembed::traits` module is removed**: `traits::tokenizer::HFTokenizer`, with its
  `from_file` and `encode_batch`, the `Tokenizer` trait and `TokenizerOutput`. The engine
  tokenizes with the GGUF's own vocabulary, the library no longer depends on `tokenizers`,
  and nothing replaces them.
- **`ltembed::benchmarking::scenario_token_lengths` and `BENCHMARK_MAX_LENGTH` are
  removed.** The function counted tokens with a `Tokenizer`.
- **`InferenceError::SequenceTooLong` and `InferenceError::AllPadding` are removed.** An
  input longer than the bundle's `max_length` is `LTEmbedError::InputTooLong`, which the
  engine returns before inference runs, as before.
- **`InferenceError::Tensor` is removed.** Its one remaining cause, llama.cpp returning no
  pooled embedding, is now `InferenceError::Backend`.
- **`LTEmbedError::InputTooLong` is `{ tokens: Option<usize>, bytes: usize, max: usize }`**,
  no longer `{ tokens: usize, max: usize }`. `bytes` is the input's size with its prefix.
  `tokens` is `None` when the input is too large to fit in `max` tokens at all, which the
  engine now rejects without tokenizing it
  ([architecture.md](./architecture.md#inputs-over-the-byte-bound)); otherwise it is the
  token count, as before.
- **`EmbedBatchProfile::tensorize_ms` and `extract_ms` are removed.** They were named for
  stages of the earlier ONNX Runtime backend and were always `0.0` with llama.cpp. `run_ms`
  covers running the model and copying out each pooled vector, as before.
- **`ltembed::engine::QUERY_PREFIX` and `DOCUMENT_PREFIX`** hold EmbeddingGemma 2's prompts
  from the table above. Code that reads them still compiles and gets the new strings.

Code that used `HFTokenizer` to count or inspect tokens, for example to chunk text under the
token limit, can load the model's pinned `tokenizer.json` with the
[`tokenizers`](https://crates.io/crates/tokenizers) crate directly.
`scripts/assemble_gguf_bundle.py --reference-tokenizer PATH` writes that file outside the
bundle. Encode the text with the prefix the engine adds, the bundle's `query_prefix` or
`document_prefix` (in a release bundle, `QUERY_PREFIX` and `DOCUMENT_PREFIX`), and with
special tokens, which adds `<bos>` and `<eos>` as the engine does:

```rust
use ltembed::engine::QUERY_PREFIX; // DOCUMENT_PREFIX for a document

let tokenizer = tokenizers::Tokenizer::from_file("reference-tokenizer.json")?;
let encoding = tokenizer.encode(format!("{QUERY_PREFIX}{text}"), true)?;
// The count the engine compares with the bundle's `max_length` (`InputTooLong`).
let token_count = encoding.get_ids().len();
```

These ids equal the GGUF vocabulary's on the [tokenizer parity](./testing.md#tokenizer-parity)
corpus. That covers the corpus, not every possible input. Load the tokenizer once and reuse
it: `tokenizers` 0.23 keeps each loaded tokenizer's BPE word cache in thread-locals that
dropping it does not free, which `HFTokenizer` avoided by turning that cache off.

## Old vectors cannot be reused

The two models map text into unrelated vector spaces. A cosine between a Jina vector and an
EmbeddingGemma 2 vector means nothing, even though both are 512 wide by default and both
are unit length. **Every stored vector must be recomputed**, documents and any cached
queries alike.

The equal width is the main hazard. An index whose schema checks only the dimension accepts
the new vectors next to the old ones without an error, and search then returns poor results
for no visible reason. Nothing in LTEmbed can detect a mixed index; it has to be prevented
when the index is rebuilt.

## Moving an index

1. **Record the model with every vector**, or at least with every index: the bundle's
   `target_id` and quant (`build-info.json`), or the model `sha256` from the bundle's
   `provenance.json`. This makes a mixed index detectable later.
2. **Build a new index beside the old one.** Re-embed every document with the new engine
   into a separate collection or table, keeping the old one serving.
3. **Switch queries and documents together.** Queries must be embedded with the same model
   as the index they search: switch the query path to the new engine at the moment the
   reader switches to the new index, not before.
4. **Keep the old deployment until the switch is done.** This LTEmbed version cannot embed
   with the previous model, so rolling back means returning to the previous LTEmbed version,
   its Jina bundle and the old index together.
5. **Delete the old index** only after the new one has served for long enough to compare.

Do not run old and new engines behind one query path, and do not backfill the new index
incrementally while serving it: until every document is re-embedded, the documents that are
still missing cannot be found.

## Things to re-check

- **Score thresholds.** Absolute similarity values differ between the models. Any cut-off
  tuned for Jina (a minimum score, a deduplication threshold, a "no good match" rule) has
  to be re-tuned on EmbeddingGemma 2 scores.
- **Output width.** EmbeddingGemma 2 is trained for 768, 512, 256 and 128 dimensions; other
  widths are accepted by `EngineConfig` but were not trained. If the index changes width,
  set it before re-embedding. Each width needs its own index: a 256-wide vector is not
  comparable with a 512-wide one.
- **Token budgets and chunking.** The limit is still 8192 tokens, but counted with a
  different tokenizer, so the same text can have a different count. Inputs over the limit
  fail with `InputTooLong`; nothing is truncated. A chunker that counts tokens must count
  them with the new model's tokenizer, which the bundle no longer contains; the
  [Rust API](#rust-api) section shows how. The engine's count includes `<bos>` and `<eos>`
  and the prefix.
- **Memory.** Peak memory grows with input length, and an engine keeps its peak for its
  lifetime: an 8192-token input needs about 1.6 GB. Size hosts for the longest input you
  accept, or set a lower `max_length` in `build-info.json` to reject longer inputs
  ([architecture.md](./architecture.md#long-inputs-and-memory)).
- **Hand-written prefixes.** If any other component embedded text with its own prefixes (for
  example a Python job using the model directly), it must now use the model's
  `query` and `document` prompts. Passing text that already carries a prefix to
  `EmbeddingInput` applies the prefix twice.
- **Retrieval quality.** The release quant `UD-Q5_K_XL` has cosine ≥ 0.996 against the F32
  model on every golden test input at every trained width, on x86-64-v3 and on aarch64,
  and each test query ranks its document first, as with the F32 model. Over 2000 short Chinese and English
  sentences, 0.5% drift below 0.995, down to 0.991, while recall stays at the F32 model's
  level (see
  [testing.md](./testing.md#golden-fixture-testsfixturestest_fixturesjson) and
  [benchmarking.md](./benchmarking.md#recorded-results)). That is no substitute for
  evaluating recall on your own data before the switch.
- **Bundles and releases.** Bundles built for the previous model fail to load
  (`UnsupportedPooling`); build a new one with `scripts/assemble_gguf_bundle.py`
  ([bundle-format.md](./bundle-format.md#assembling-a-bundle)). Release tarballs are now
  named `ltembed-<version>-unsloth__embeddinggemma-2-GGUF-UD-Q5_K_XL-linux-arm64.tar.gz`.
- **License.** The model cards of `google/embeddinggemma-2` and of both Unsloth repositories
  the bundle comes from declare `apache-2.0`. The previous model had its own terms. Check
  that the new license fits your distribution before shipping the bundle. Release tarballs
  carry the license text and the model's attribution in `licenses/embeddinggemma-2/`
  ([NOTICE.md](../licenses/embeddinggemma-2/NOTICE.md)); if you ship the bundle, ship
  them with it.
