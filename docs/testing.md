# Testing

What each test suite checks, what it needs, and exactly what CI runs. Every Rust target
links the static llama.cpp archives, so all `cargo test` and `cargo clippy` commands need
aarch64 Linux or x86-64-v3 Linux, with `STATIC_LLAMA_DIR` set to the matching variant. On
another ARM64 host, such as macOS on Apple silicon, run them in the container from
[development.md](./development.md#container-builds), never under emulation on x86_64.

## Suites

| Suite | Command | Needs beyond the build | Run by CI |
|---|---|---|---|
| Library unit tests (`src/**`) | `cargo test --lib` | — | Yes |
| Benchmark support and binary unit tests | `cargo test --test benchmarking_support_tests --bin benchmark_ltembed` | — | Yes |
| Example compiles | `cargo check --example api_usage` | — | Yes |
| Example unit test | `cargo test --example api_usage` | — | **No** |
| Tokenizer parity (`src/engine/tokenizer_parity.rs`) | `cargo test --lib tokenizer_parity -- --ignored` | A GGUF bundle in `LTEMBED_TEST_BUNDLE_DIR` and the reference `tokenizer.json` in `LTEMBED_TEST_REFERENCE_TOKENIZER` | Yes (required) |
| Integration tests, Tier 1 | `cargo test --test integration_tests` | — | Yes |
| Integration tests, Tier 2 | same | A GGUF bundle in `LTEMBED_TEST_BUNDLE_DIR` | Yes (required) |
| Python script tests | `python3 -m pytest tests/ -q` | `pytest`, `numpy`; `torch` (and `sentence-transformers` for one) for two modules | Yes, without `torch` |
| Format | `cargo fmt --check` | Nothing (no build script) | Yes |
| Clippy | `cargo clippy --all-targets -- -D warnings` | — | Yes |
| Rustdoc | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` | — | Yes |
| Markdown links and `AGENTS.md` paths | `python3 scripts/check_doc_references.py` | Git and Python 3 only (no build) | Yes |

## What CI runs

`.github/workflows/ci.yml` runs on every push and pull request to `main`. It has three
jobs. `Test` runs twice, as a matrix: `Test` on `ubuntu-24.04-arm` with the
`aarch64-graviton2` variant, and `Test (x86_64-v3)` on `ubuntu-24.04` with the `x86_64-v3`
variant. `Lint` and `Python script tests` run on `ubuntu-24.04-arm` only.

**`Test`** and **`Test (x86_64-v3)`** (job env `LTEMBED_REQUIRE_TEST_BUNDLE=1`):

1. `.github/scripts/fetch-static-llama.sh`: download, verify and extract the pinned static
   llama.cpp release for the runner's architecture, then export `STATIC_LLAMA_DIR`. On
   x86_64 it first checks that the runner's CPU supports x86-64-v3 and fails the job if
   not. The `x86_64` runner label alone does not guarantee that.
2. `mkdir -p assets`
3. `cargo test --lib`
4. `cargo test --test benchmarking_support_tests --bin benchmark_ltembed`
5. `cargo check --example api_usage`
6. `python3 scripts/assemble_gguf_bundle.py ci_gguf_bundle --reference-tokenizer
   ci_reference_tokenizer.json`: the UD-Q5_K_XL GGUF at its pinned Hugging Face revision,
   and the model's `tokenizer.json` at its own, outside the bundle, each checked against its
   pinned sha256 (see [bundle-format.md](./bundle-format.md#assembling-a-bundle)); then
   export `LTEMBED_TEST_BUNDLE_DIR` and `LTEMBED_TEST_REFERENCE_TOKENIZER`.
7. `cargo test --lib tokenizer_parity -- --ignored`: the
   [tokenizer parity tests](#tokenizer-parity), which
   plain `cargo test --lib` (step 3) skips as ignored.
8. `cargo test --test integration_tests`. The llama.cpp reference test and
   `test_input_of_the_longest_tokens_that_fits_embeds` each embed an 8192-token input while
   other tests run their own engines; the whole test process peaks at 3.8–4.1 GiB on a
   12-thread x86-64-v3 host, with the default of one test thread per CPU
   ([architecture.md](./architecture.md#long-inputs-and-memory)). Fewer test threads lower it.

**`Python script tests`**: Python 3.13, `pip install pytest numpy`, then
`python3 -m pytest tests/ -q`. `torch` and `sentence-transformers` are not installed, so
the 3 tests in `tests/test_bench_pytorch.py` that run the PyTorch runner's `load_model` and
`main` with real torch skip. The runner's other tests need only `numpy`
([Python tests](#python-tests)).

**`Lint`**: `python3 scripts/check_doc_references.py` with the runner's own `python3`
(see [Documentation references](#documentation-references)), then fetch static llama,
`cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` and
`RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`.

CI does not run plain `cargo test`, doc tests, the example's unit test, or any benchmark.
Benchmarks run only by hand; see [benchmarking.md](./benchmarking.md).

## Integration tests: Tier 1 and Tier 2

`tests/integration_tests.rs` mixes two kinds of tests.

**Tier 1** never needs model weights. Most of these tests build throwaway bundles in a
temp directory and check load-time validation:

- missing `model.gguf` or `build-info.json` → `ModelLoad(MissingFile)`
- malformed `build-info.json` → `ModelLoad(Metadata)`
- unsupported `input_kind` → `ModelLoad(UnsupportedInputKind)`, and the previous model's
  `"pooling": "last_token"` → `ModelLoad(UnsupportedPooling)`
- `output_dimension` larger than raw → `ModelLoad(Config)`
- the skip and require logic itself (`bundle_or_skip`)
- the committed fixtures agree with the engine and with each other, so a fixture left stale
  by a model, prompt or width change fails every run, not only runs with a bundle:
  `test_golden_fixture_matches_engine_constants` checks
  [the golden fixture](#golden-fixture-testsfixturestest_fixturesjson) against
  `QUERY_PREFIX`, `DOCUMENT_PREFIX`, `RAW_EMBEDDING_DIMENSION`, `MAX_LENGTH` and the trained
  widths; `test_golden_fixture_problems`, `test_stale_golden_fixture_fails` and
  `test_golden_fixture_without_prompts_fails` show the check rejects synthetic stale
  fixtures; `test_llama_cpp_reference_matches_golden_inputs` checks that
  [the llama.cpp reference](#llamacpp-reference-testsfixturesllama_cpp_referencejson) covers
  the same inputs and has vectors for the variant the tests were built for.
- the llama.cpp reference's two variants agree as closely as the
  [embedding space](./architecture.md#embedding-space), which does not name the variant,
  assumes: `test_llama_cpp_reference_variants_agree_within_one_space` requires a cosine of at
  least 0.999 and norms within 2% for every input at every trained width.

The `build-info.json` range check for `max_length` (`1..=8192`) is covered by unit tests in
`src/engine/bundle.rs`. Unit tests in `src/engine/mod.rs` and `src/engine/llama/mod.rs`
cover the [byte bound](./architecture.md#inputs-over-the-byte-bound) without a model: an
input over it fails its batch before any input is tokenized, one at it is tokenized, and a
text over `i32::MAX` bytes is `InputTooLong` for any vocabulary, checked on its length
alone. Unit tests in `src/engine/mod.rs` cover `token_count` and `max_length()` without a
model: the count is that of the prefixed text, BOS and EOS included, a count over
`max_length` is returned and is the one `embed`'s `InputTooLong` then names, a text over the
bound is rejected without being tokenized, and `max_length()` is the bundle's limit. Unit
tests in `src/engine/space.rs` and `src/engine/mod.rs` cover the
[embedding space](./architecture.md#embedding-space) without a model: the release bundle's
exact string form, a different string for a change to any field, the escaping of `%`, `;`
and non-ASCII bytes in prompts, the file hash against FIPS 180-2 test vectors, a hash that
stays that of the opened file after its path is replaced or removed, the load check that
fails once the path names another file (a symlink renamed over it) or none (removed), and
an engine that reports its own file, prompts and config and keeps the first result.

**Tier 2** needs a real bundle in `LTEMBED_TEST_BUNDLE_DIR`. Every Tier 2 test reads its
`model.gguf` and `build-info.json`:

| Test | Checks |
|---|---|
| `test_golden_parity_at_every_trained_dimension` | The 22 golden inputs of at most 512 tokens, at widths 768, 512, 256 and 128: width, finite values, unit norm, cosine ≥ 0.995 for each input against the golden truncated to that width, and each query ranks its relevant document first |
| `test_llama_cpp_reference_parity` | All 25 golden inputs, up to 8192 tokens: the GGUF's sha256 is the one the reference was generated from; the raw 768-wide vector has cosine ≥ 0.99999 and a norm within 1e-4 of the reference for this variant; and each input passes the per-input golden gate at 768 and 512. The token ids the reference recorded are checked by the [tokenizer parity tests](#tokenizer-parity). |
| `test_embed_batch_matches_single_embeds_and_repeats_exactly` | `embed_batch` returns one vector per input, in order, bit-identical to embedding each alone, and repeating the call returns bit-identical vectors |
| `test_embed_batch_profiled_measures_every_stage` | On the short golden inputs, `embed_batch_profiled` returns `embed_batch`'s vectors and a profile whose every field is measured: the input count, the longest input's token count from the fixture, non-negative stage times that sum to at most `total_ms`, and non-zero `tokenize_ms` and `run_ms` |
| `test_concurrent_calls_match_a_single_thread` | 8 threads share one engine and each makes the same 5 calls twice, thread `t` starting at call `t` modulo 5: a short query, a batch of two short inputs, a 99-token document, and two inputs of about 24,000 tokens that are tokenized and rejected. Every call returns the bit-identical vectors, or the same `InputTooLong` token count, that it returns on one thread. Tokenizing takes no lock ([architecture.md](./architecture.md#concurrency)), so this runs `llama_tokenize` on several threads while another runs `llama_encode` |
| `test_query_and_document_prompts_differ` | The same text as a query and as a document gives different vectors |
| `test_default_config_output_is_normalized_and_512_wide` | `EngineConfig::default()` output has 512 values and unit norm |
| `test_bundle_without_tokenizer_json_loads_and_ignores_one` | A directory of only `model.gguf` and `build-info.json` loads and gives the same vector as the test bundle, and still does with an unparseable `tokenizer.json` beside them |
| `test_long_input_returns_input_too_long_error` | An input of more than 8192 tokens returns `InputTooLong { max: 8192 }` with its full token count and its size with the prefix |
| `test_token_count_matches_the_golden_fixtures` | `token_count` equals the count the [golden fixture](#golden-fixture-testsfixturestest_fixturesjson) recorded with the model's own tokenizer, prompt, BOS and EOS included, for each of its 25 inputs, the 8192-token one included; `max_length()` is the fixture's `max_length` |
| `test_input_of_the_longest_tokens_that_fits_embeds` | The [byte bound](./architecture.md#inputs-over-the-byte-bound) rejects no input that fits: a document of 8183 copies of the vocabulary's longest token (31 × `▁`), exactly 8192 tokens and 761,039 bytes, embeds, and one more copy returns `InputTooLong { tokens: Some(8193) }`: it is tokenized, not rejected by its size. `token_count` gives 8192 and 8193 for the two, and `max_length()` 8192 |
| `test_input_over_the_byte_bound_is_rejected_without_tokenizing` | Inputs one byte over the bound (8192 × 93 bytes, prefix included) and of 64 MiB return `InputTooLong { tokens: None }` with their size, from `embed` and from `token_count`; one at the bound is tokenized and counted, and `embed`'s `InputTooLong` names `token_count`'s count |
| `test_bundle_max_length_is_enforced_without_truncation` | With `max_length` 64, `max_length()` is 64, an input that `token_count` counts as exactly 64 tokens embeds, and one it counts as 65 returns `InputTooLong { tokens: Some(65), max: 64 }`. `token_count` finds the two texts, counting past the limit. |
| `test_gguf_width_mismatch_returns_model_load_error` | With `raw_embedding_dimension` 1024, loading fails with `ModelLoad(Runtime)` |
| `test_hidden_width_as_raw_dimension_returns_model_load_error` | With `raw_embedding_dimension` 512, the hidden width, loading fails: the engine must return the projected 768-wide vector |
| `test_gguf_without_mean_pooling_returns_model_load_error` | A copy of the GGUF with `<arch>.pooling_type` changed to 3 (last) fails to load with `ModelLoad(Runtime)` |
| `test_gguf_without_eos_returns_model_load_error` | A copy of the GGUF with `tokenizer.ggml.add_eos_token` set to false fails the load-time tokenization check, naming the missing EOS. A copy without `add_bos_token` would still load: llama.cpp forces it on for this tokenizer, so the BOS half of the check has unit tests only. |
| `test_embedding_space_names_the_release_gguf_and_settings` | The test bundle's [embedding space](./architecture.md#embedding-space) has the release GGUF's sha256 (from the llama.cpp reference), the model's prompts and the engine's config; a different `output_dimension` or `l2_normalize` changes it. 4 threads instead of 1 do not, and move the vector by rounding at most (cosine ≥ 0.99999) |
| `test_edited_bundle_metadata_cannot_claim_another_space` | With `target_id` and `quant` edited in `build-info.json` and a `provenance.json` that names another model, an engine gives the same vector and the same space; with `max_length` 64, the same space and a vector within rounding (cosine ≥ 0.99999); with an edited `query_prefix`, another vector and a space with that prefix |
| `test_embedding_space_names_the_gguf_file_the_engine_loaded` | A copy of the GGUF with `quantize.imatrix.chunks_count` changed, which llama.cpp ignores, embeds identically but has a space with the copy's sha256. An engine loaded before `model.gguf` was replaced by the copy still reports the release GGUF's sha256 |

The GGUF copies in `test_gguf_without_mean_pooling_returns_model_load_error`,
`test_gguf_without_eos_returns_model_load_error` and
`test_embedding_space_names_the_gguf_file_the_engine_loaded`, about 210 MB each, go to Cargo's
integration-test scratch directory (`CARGO_TARGET_TMPDIR`, `target/tmp/` by default) rather
than `/tmp`, and are removed afterwards, even when the test fails. No crafted
GGUF is committed. The tests that swap one bundle file symlink the others instead of
copying them.

### Skipping vs. failing without a bundle

By default a Tier 2 test **skips**: it returns early, passes, and prints `Skipping: …`. That
happens when `LTEMBED_TEST_BUNDLE_DIR` is unset, or when the directory lacks `model.gguf`.
Local runs without a bundle therefore pass.

`build-info.json` is not part of that check. If the directory has `model.gguf` but no
`build-info.json`, the tests do not skip. Each one fails when it builds the engine, with
`ModelLoad(MissingFile)`, whether or not `LTEMBED_REQUIRE_TEST_BUNDLE` is set.

`LTEMBED_REQUIRE_TEST_BUNDLE=1` turns each skip into a failure that names the missing
variable or file. `0`, empty or unset keeps the skip. Any other value fails, so a typo such
as `true` cannot quietly turn the check off. CI sets it at job level, so a broken
`$GITHUB_ENV` hand-off fails the integration step instead of passing with no model-backed
assertions.

```bash
# Local run without a bundle: Tier 2 tests skip.
cargo test --test integration_tests

# As in CI: Tier 2 tests must run.
LTEMBED_REQUIRE_TEST_BUNDLE=1 LTEMBED_TEST_BUNDLE_DIR="$PWD/gguf_bundle" \
  cargo test --test integration_tests
```

## Golden fixture: `tests/fixtures/test_fixtures.json`

This is the **immutable PyTorch F32 reference**. `scripts/generate_fixtures.py` produces it
from the original safetensors model ([`unsloth/embeddinggemma-2`](https://huggingface.co/unsloth/embeddinggemma-2)
at revision `c8f89cc7…`, a copy of `google/embeddinggemma-2`), run by SentenceTransformers
on the CPU in float32, text tower only, with the model's own pipeline: its `query` and
`document` prompts, mean pooling over every token (prompt, `<bos>` and `<eos>` included),
the 512→768 projection, and L2 normalization. It never runs LTEmbed, llama.cpp or a GGUF.
The fixture records the generator, the model revision, the prompts and the library
versions (`sentence-transformers` 6.1.0, `transformers` 5.19.0, `torch` 2.14.1+cpu).

```json
{
  "model": {"repo": "unsloth/embeddinggemma-2", "revision": "c8f89cc7…", "upstream": "google/embeddinggemma-2"},
  "prompts": {"query": "task: search result | query: ", "document": "title: none | text: "},
  "raw_dim": 768, "dims": [768, 512, 256, 128], "max_length": 8192, "short_tokens": 512,
  "fixtures": [{"name": "query_zh", "kind": "query", "text": "如何重置密码？", "tokens": 14,
                "relevant": "document_zh", "embedding": [/* 768 floats, unit norm */]}]
}
```

- 25 inputs: 21 short ones (10 to 68 tokens) in English, Chinese, Japanese, Korean, French,
  German, Russian and Arabic, mixed Chinese and English, Python and Rust code, emoji, an
  empty query and irregular whitespace; and four from `scripts/benchmark_corpus.json` of
  89, 1024, 4074 and 8192 tokens, the last exactly `MAX_LENGTH`.
- `text` is raw caller text; the prompt is applied by the generator and by the engine,
  never stored. `tokens` counts the prompted input with `<bos>` and `<eos>`.
- `relevant` names the document a query must rank first among the documents. The generator
  checks the reference itself does.
- Each vector is the full 768-wide embedding. A smaller width is its first values,
  renormalized, which is how the model card says to use the widths it was trained for.
- **Never regenerate it from GGUF or LTEmbed output.** The golden measures how far the
  quantized llama.cpp path drifts from the original model; output captured from the
  implementation would make that check meaningless.
- Regenerate it, from PyTorch, only when the reference itself changes: a different model
  or revision, different inputs, or a change to pooling, prompts or widths. Requirements and
  usage are in the script's docstring; it needs `torchvision` and `pillow` besides `torch`.

### Gates and how they were set

Quantization moves a vector away from the F32 reference; an implementation mistake (wrong
prompt, pooling, tokenizer or width) moves it further. The gate sits between the two: cosine
≥ **0.995** for every input, at every width. There is no separate mean gate: every input
must reach 0.995, so the mean does too.

Cosine of LTEmbed's output against the golden, over all 25 inputs at 768 dimensions, for
each GGUF of the pinned revision (x86-64-v3, flash attention on):

| GGUF | Minimum | Mean |
|---|---|---|
| `UD-Q4_K_XL` | 0.98944 | 0.99471 |
| `UD-Q5_K_XL` (release) | 0.99690 | 0.99849 |
| `UD-Q6_K_XL` | 0.9983 | 0.9992 |
| `Q8_0` | 0.9996 | 0.99985 |
| `BF16` | 0.99999 | 0.99999+ |

`BF16` agrees with the F32 reference on every input, including the 8192-token one, so
tokenization, prompts, pooling, projection and LTEmbed's attention settings are right and
the remaining gap is weight quantization, which shrinks with each larger quant. For the
release quant at the other widths: 512, minimum 0.99702 and mean 0.99852; 256, minimum
0.99730 and mean 0.99869; 128, minimum 0.99829 and mean 0.99911. Its worst input is a Rust
code snippet at 768 and 512, and a Python one at 256 and 128. On aarch64 the release
quant's vectors differ slightly (see the llama.cpp reference below). From the aarch64
reference vectors, which LTEmbed must match, its minimum and mean are 0.99622 and 0.99847
at 768, 0.99640 and 0.99851 at 512, 0.99700 and 0.99869 at 256, and 0.99807 and 0.99909 at
128, with the same Rust snippet worst at every width.

The gate was set just below that measured error. Until the release moved from `UD-Q4_K_XL`
to `UD-Q5_K_XL`, the gates were 0.985 per input and 0.993 on average, just below the 4-bit
quant's error; the previous model's gate was 0.99 per input. The gate holds for these 25
inputs, not for every text: over 2000 short CN/EN sentences at 512 dimensions, 10 of the
release quant's vectors (0.5%) fall below 0.995, the lowest at 0.9911
([benchmarking.md](./benchmarking.md#recorded-results)). The llama.cpp reference test below
catches implementation errors with a far tighter gate.

### How sensitive the gates are

Cosine of the F32 reference's own output for deliberately wrong pipelines against the
correct embedding, over the 21 short inputs:

| Mistake | Highest cosine | Caught by |
|---|---|---|
| No prompt | 0.938 | per-input gate |
| Query and document prompts swapped | 0.982 | per-input gate |
| The previous model's `Query: `/`Document: ` prompts | 0.929 | per-input gate |
| Last-token pooling | 0.880 | per-input gate |
| First-token pooling | 0.871 | per-input gate |
| Prompt applied twice | 0.9985 (median 0.991) | per-input gate; llama.cpp reference |
| `<eos>` missing | 0.9856 (median 0.973) | per-input gate; load-time tokenization check |

Every mistake also fails the llama.cpp reference test, which uses the correct prompt and
the GGUF's own tokenizer.

## llama.cpp reference: `tests/fixtures/llama_cpp_reference.json`

The golden can only show LTEmbed is within quantization error of the F32 model. This
fixture shows LTEmbed computes what llama.cpp itself computes on the same GGUF: upstream
llama.cpp's `llama-embedding` example (`examples/embedding/embedding.cpp`), built from the
commit the pinned static release compiles and linked against that release's own archives,
run on the release GGUF for every golden input. It runs the same tokenizer and kernels as
LTEmbed but none of LTEmbed's code: it tokenizes the prompted text and builds its batch
itself. The settings are LTEmbed's: mean pooling, non-causal attention, flash
attention on, `n_ctx = n_batch = n_ubatch = 8192`, no normalization.

The kernels differ between the variants (their raw vectors agree to a mean cosine of
0.99961, and 0.99901 on the worst input, with norms up to 1.55% apart), so the fixture holds
one set of raw 768-wide vectors per variant, plus the token ids the GGUF
gave, the GGUF identity, the llama.cpp commit and the tool's arguments. LTEmbed matches its
own variant's vectors to the tool's 7-decimal output: the gate is cosine ≥ 0.99999 and a
norm within 1e-4. Measured on x86-64-v3, the two agree to cosine 1 − 4·10⁻¹⁶.

`scripts/generate_llama_cpp_reference.py` writes one variant's section and keeps the
other's. `scripts/build_llama_cpp_reference_tool.sh` builds the tool from a llama.cpp
checkout at the release's commit and an extracted release (see its header for the
details):

```bash
COMMIT=620d2225cc436ab0b001e3200e590e3b286ef9ab   # STATIC_LLAMA_LLAMA_CPP_COMMIT
git init .llama-artifacts/llama.cpp && git -C .llama-artifacts/llama.cpp fetch --depth 1 \
  https://github.com/ggml-org/llama.cpp "$COMMIT" && git -C .llama-artifacts/llama.cpp checkout FETCH_HEAD
python3 scripts/assemble_gguf_bundle.py .llama-artifacts/bundle

# x86-64-v3 host, x86_64-v3 release in .llama-artifacts/extracted-x86_64-v3:
scripts/build_llama_cpp_reference_tool.sh .llama-artifacts/llama.cpp \
  .llama-artifacts/extracted-x86_64-v3 .llama-artifacts/reftool-x86_64-v3
python3 scripts/generate_llama_cpp_reference.py --variant x86_64-v3 \
  --tool .llama-artifacts/reftool-x86_64-v3/llama-embedding --gguf .llama-artifacts/bundle/model.gguf

# aarch64 Linux host, aarch64-graviton2 release in .llama-artifacts/extracted:
scripts/build_llama_cpp_reference_tool.sh .llama-artifacts/llama.cpp \
  .llama-artifacts/extracted .llama-artifacts/reftool-aarch64
python3 scripts/generate_llama_cpp_reference.py --variant aarch64-graviton2 \
  --tool .llama-artifacts/reftool-aarch64/llama-embedding --gguf .llama-artifacts/bundle/model.gguf
```

The aarch64 section needs ARM64 hardware, such as a Raspberry Pi 5: never build or run the
tool under emulation on an x86_64 host. The generator checks the GGUF's sha256, that the
tool reports the pinned commit, that each input gives the token count the golden recorded,
and that both variants saw the same token ids. Without `--reset`, a run keeps the other
variant's section, so that section must come from the same inputs: the generator checks its
input names before running the tool, and its token ids as each input runs. When the golden
inputs change, run with `--reset`, which keeps neither the other section nor its token ids,
then generate the other variant.

## Token-id fixture: `tests/fixtures/token_ids.json`

`scripts/generate_token_ids.py` encodes a fixed set of inputs with the Python `tokenizers`
library, singly and as one batch. It uses the model's `tokenizer.json` at the revision
`scripts/assemble_gguf_bundle.py` pins; the bundle itself has no `tokenizer.json`. The
fixture records:

- `tokenizer`: repo, `revision` (`c8f89cc7…`) and the file's `sha256`
- `tokenizers_version` (`0.23.2`): the script refuses to run unless the installed Python
  `tokenizers` equals the Rust `tokenizers` version in `Cargo.lock`, the dev-dependency the
  [tokenizer parity test](#tokenizer-parity) reads its reference with
- `padding`: `null`. EmbeddingGemma 2's tokenizer pads nothing, so the script requires each
  batch row to equal the single encoding and stores only that.
- `cases`: 15 inputs, as literal `text` or as a `repeat` spec (long inputs are never stored
  literally): prompted English and CJK text, emoji, mixed whitespace, byte fallback, the
  empty string, added-token text in several forms, a run of 100,000 spaces (3230 tokens),
  and 6009 tokens of repeated mixed Chinese and English.

`test_gguf_vocabulary_matches_reference_token_ids` in `src/engine/tokenizer_parity.rs`
requires the GGUF's vocabulary, which the engine tokenizes with, to give each case's ids
exactly. The parity test requires its reference `tokenizer.json` to have the recorded
`sha256`, and fails with the expected repo and revision if it does not.

**Regenerate it** after changing the script's `INPUTS`, bumping `REVISION`, or bumping
`tokenizers` in `Cargo.lock`:

```bash
pip install "tokenizers==<version of the tokenizers crate in Cargo.lock>"   # 0.23.2 today
python3 scripts/generate_token_ids.py
```

If the installed version is wrong, the script exits and prints the `pip install` command to
run. No model weights are needed.

`REPO` and `REVISION` in `scripts/generate_token_ids.py` and `scripts/generate_fixtures.py`
must equal `TOKENIZER_REPO` and `TOKENIZER_REVISION` in `scripts/assemble_gguf_bundle.py`,
and its `TOKENIZER_SHA256` must equal the hash the fixture records.
`tests/test_reference_generators.py` checks all of them.

## Tokenizer parity

The engine tokenizes every input with the GGUF's own vocabulary (`tokenizer.ggml.*`)
through `llama_tokenize`, as upstream llama.cpp does
([bundle-format.md](./bundle-format.md#tokenization)). The load-time check only shows that
this adds BOS and EOS. `src/engine/tokenizer_parity.rs` checks the token ids themselves
against references that do not come from llama.cpp or LTEmbed, and the input byte bound on
the same texts:

- `test_tokenizer_json_matches_gguf_vocabulary`: the model's pinned `tokenizer.json`, run
  by the `tokenizers` crate, must give the same ids as the GGUF's vocabulary on a fixed
  corpus of 25,401 texts. `tokenizers` is a dev-dependency for the tests only; the library
  does not use it.
- `test_gguf_vocabulary_matches_reference_token_ids`: the GGUF's vocabulary must give the
  ids of the 15 [token-id fixture](#token-id-fixture-testsfixturestoken_idsjson) cases,
  written by Python `tokenizers`, and the ids upstream `llama-embedding` recorded for the 25
  prompted golden inputs in the
  [llama.cpp reference](#llamacpp-reference-testsfixturesllama_cpp_referencejson). That is
  40 inputs, up to 8192 tokens.
- `test_byte_bound_admits_every_text_that_fits`: the
  [input byte bound](./architecture.md#inputs-over-the-byte-bound) rejects no text that
  fits. The GGUF's longest token, which sets the bound, must be as long as the longest token
  of `tokenizer.json` (93 bytes). Each encoding of the corpus below, of 64 copies of that
  token, of runs of six whitespace characters, and of every special token's text between
  whitespace runs, must be within the bound for a `max_length` of its own token count. On
  the pinned GGUF that is 76,656 encodings, at most 90.2 bytes per token. With the bound's
  per-token length divided by 3, as if `▁` counted one byte, the test fails.

The corpus of the first test:

| Source | Texts |
|---|---|
| `tests/CN_EN_Data.csv`, both columns | 5268 |
| [Golden fixture](#golden-fixture-testsfixturestest_fixturesjson) inputs | 25 |
| [Token-id fixture](#token-id-fixture-testsfixturestoken_idsjson) inputs, `repeat` specs expanded | 15 |
| `text` fields of `scripts/benchmark_corpus.json` and `scripts/retrieval_eval_cases.json` | 28 |
| Hand-picked edge cases (`EDGE_CASES`, plus five long runs without a word boundary) | 65 |
| Seeded random strings | 20,000 |

- **Edge cases**:
  - whitespace and invisible characters: no-break and ideographic spaces, zero-width
    characters, BOM, NEL, line and paragraph separators, NUL, control characters;
  - composed and decomposed Unicode, and compatibility characters;
  - scripts written without spaces;
  - emoji ZWJ sequences, flags and keycaps;
  - digits in several scripts;
  - code and URLs;
  - special-token text in exact, nested, unterminated and wrong-case forms.
- **Random strings**: each has 1 to 12 segments. A segment is either a special-token or
  whitespace fragment, or a run of code points from one of 30 ranges (ASCII, controls, Latin,
  combining marks, Greek, Cyrillic, Hebrew, Arabic, Devanagari, Thai, Hangul, kana, CJK and
  CJK Extension B, private use, variation selectors, fullwidth forms, hieroglyphs, regional
  indicators, emoji). A fixed seed (`RANDOM_SEED`, xorshift64*) makes them the same on every
  run.

Each text is tokenized three times: as is, with the query prefix and with the document
prefix, through the engine's own `prefixed_text`. That makes 76,203 encodings, about 4.1
million tokens. Any difference fails the test, which lists the first 20 with the position of
their first differing token. There is no tolerance.

On the pinned GGUF all 76,203 encodings and all 40 reference inputs are identical. On
x86-64-v3 this was measured locally, and the three tests take about 20 s together in a debug
build. CI runs them on both variants.

**What a pass shows.** The GGUF's vocabulary agrees with the model's `tokenizer.json` on
this corpus, and with the recorded ids, for the pinned GGUF, `tokenizer.json`, `tokenizers`
version and static llama.cpp release. It is not a proof for every input. Neither side is
checked against the tokenizer the model was trained with.

**How sensitive it is.** With the engine's `llama_tokenize` call changed to not match
special-token text (`parse_special = false`), the corpus test finds 20,148 differing
encodings and the reference-id test 3 differing inputs. Comparing the real GGUF with a
deliberately altered copy of `tokenizer.json` (measured before the reference's sha256 was
checked, which now rejects such a copy):

| Change to `tokenizer.json` | Encodings that differ |
|---|---|
| `<eos>` removed from the template | 76,203 (all) |
| NFC normalization added before the space replacement | 14,916 |
| `<mask>` no longer an added token | 2,712 |
| One BPE merge removed (`▁th` + `e`) | 0 |

Removing one merge changes nothing because the BPE reaches the same tokens through other
merges. Python `tokenizers` also gives unchanged ids for all 5268 Chinese and English
sentences after removing any one of nine merges tried.

**Running them.** The tests need the GGUF, so they are `#[ignore]`d and plain
`cargo test --lib` stays model-free. The reference `tokenizer.json` is written outside the
bundle by `scripts/assemble_gguf_bundle.py --reference-tokenizer`
([bundle-format.md](./bundle-format.md#assembling-a-bundle)):

```bash
LTEMBED_REQUIRE_TEST_BUNDLE=1 LTEMBED_TEST_BUNDLE_DIR="$PWD/gguf_bundle" \
  LTEMBED_TEST_REFERENCE_TOKENIZER="$PWD/reference-tokenizer.json" \
  cargo test --lib tokenizer_parity -- --ignored
```

- **Skip or fail**: they follow the same rules as the Tier 2 integration tests
  ([Skipping vs. failing](#skipping-vs-failing-without-a-bundle)). All three need
  `model.gguf` in `LTEMBED_TEST_BUNDLE_DIR`, and the corpus and byte-bound tests also the
  file `LTEMBED_TEST_REFERENCE_TOKENIZER` names; with `LTEMBED_REQUIRE_TEST_BUNDLE=1` a
  missing variable or file fails instead of skipping. None reads `build-info.json`.
- **Corpus check**: the model-free unit test `test_corpus_reads_every_source`, part of
  `cargo test --lib`, checks that every source loads and that the `repeat` specs are
  expanded.
- **Expected warnings**: texts that start with `<bos>` or end with `<eos>` make llama.cpp
  print `check_double_bos_eos` warnings, about 560 per run. Both tokenizers add a second BOS
  or EOS there.

The corpus test has no fixture to regenerate: it computes both sides on every run. If either
test fails after a bump of the static llama.cpp release, the GGUF or the tokenizer revision,
that is a finding: the engine no longer tokenizes the way the model's own tokenizer does.

## Python tests

`tests/test_*.py` cover the scripts in `scripts/`: the bundle assembler, the benchmark
orchestrator, report renderer, metadata writer, retrieval-case builder, the three reference
generators, embedding comparison, documentation reference check and the PyTorch runner.
They do not build LTEmbed or need a bundle or model weights. The orchestrator tests check
the `cargo run` command line without running it. `tests/test_assemble_gguf_bundle.py` and
`tests/test_reference_generators.py` also check that the committed fixtures, the generators,
the bundle pins and the static release pins in `ci.yml` still name the same files, and that
the release archive check requires the model's license text and attribution in
`licenses/embeddinggemma-2/`, which name the pinned GGUF.
`tests/test_reference_generators.py` also checks that `licenses/llama.cpp/NOTICE.md` names the
static release and llama.cpp commit that `ci.yml` pins, that its license copies have the
sha256 it records, and that `benchmark-arm64.yml` pins the same static release as `ci.yml`.
`tests/CN_EN_Data.csv` is the source data for the CN/EN retrieval evaluation.

`tests/test_bench_pytorch.py` checks the PyTorch runner with a fake model. The runner
imports `torch`, `transformers` and `sentence_transformers` only in the functions that load
and run the model. The tests that do not call those functions block the three imports, so
on any machine they fail if the rest of the runner needs one of them. The tests of
`load_model`, which compare real torch dtypes, and of `main` need torch and skip without
it.

`tests/test_release_bundles.py` covers the version check of
`.github/workflows/release-bundles.yml`. It runs `.github/scripts/release-version.sh` on
scratch `Cargo.toml` files with the tag of each trigger, and the workflow's own version step
under the shell GitHub runs it with, so it fails if a tag that is not `v<version>` no longer
fails the step. It also checks that the step runs on every trigger before anything is
downloaded or published, and that the tarball is named with the version it checked.

`tests/test_pre_push_hook.py` covers `.githooks/pre-push` and the x86-64-v3 check in
`.github/scripts/x86-64-v3.sh`. It runs the hook in a scratch Git repository with stub
`uname`, `cargo` and `docker` commands and a stub CPU check, so on any machine it checks
which hosts run clippy natively, which use the container, and which the hook refuses
because the container would run under emulation, without compiling anything.
Its releases carry the artifact contract `.github/scripts/fetch-static-llama.sh` accepts
(`EXPECTED_CONTRACT`), so it fails if the hook expects another one or names another in its
error. The real CPU check runs against a fake dynamic loader and `/proc/cpuinfo`. It needs
`bash` and `git`.

## Documentation references

`scripts/check_doc_references.py` runs offline over the files Git tracks and fails if a
reference does not resolve:

- **Links in every `*.md` file.** The target of each inline link, image and reference
  definition must be a tracked file or directory. A target that starts with `/` resolves
  from the repository root, as on GitHub; any other resolves from the linking file's
  directory, and one with no path (`#anchor`, or an empty `()`) is the linking file itself.
  A `#fragment` on a link to a Markdown file must match one of its headings, slugged the
  way GitHub does it (`## What CI runs` is `#what-ci-runs`). Targets with a URL scheme, such
  as `https:` or `mailto:`, are skipped and never fetched. Link syntax in a code span or a
  fenced or indented code block is code, not a link. HTML `<a href>` links are not checked,
  and an HTML `<a id>` does not count as a heading.
- **Paths in `AGENTS.md` files.** Each token in a code span or code block that contains `/`
  or ends in a file extension must name a tracked file or directory. A token resolves from
  the repository root, or else from the `AGENTS.md` file's own directory. Each
  `file` § "Section" reference must match a heading of that file. A § "Section" refers to
  the last Markdown file named before it in the same paragraph, or to the `AGENTS.md`
  itself if there is none. One that directly follows a code span naming no Markdown file,
  as in the notation `file` § "Section" itself, is not checked.

The path check covers only `AGENTS.md` files, so the historical documents under
`docs/history/`, which may name files that no longer exist, get only the link check.
Placeholders such as `docs/history/<category>/` are skipped. Names that are not repository
paths, such as the bundle files `model.gguf` and `tokenizer.json`, are listed in
`NOT_REPO_PATHS` in the script. Add a name there when an `AGENTS.md` has to mention one.

It prints `file:line: problem` for each broken reference and exits 1. It needs Git and
Python 3.10 or newer, and nothing else: no `pip install`, no build, no network.
