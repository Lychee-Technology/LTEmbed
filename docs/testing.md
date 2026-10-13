# Testing

What each test suite checks, what it needs, and exactly what CI runs. Every Rust target
links the static llama.cpp archives, so all `cargo test` and `cargo clippy` commands need
aarch64 Linux or x86-64-v3 Linux, with `STATIC_LLAMA_DIR` set to the matching variant. On
other hosts, run them in the container from
[development.md](./development.md#container-builds).

## Suites

| Suite | Command | Needs beyond the build | Run by CI |
|---|---|---|---|
| Library unit tests (`src/**`) | `cargo test --lib` | `assets/tokenizer.json` (tracked) for tokenizer shape tests | Yes |
| Tokenizer reload memory test | `cargo test --test tokenizer_reload_tests` | — (writes its own BPE tokenizer to a temp file) | Yes |
| Benchmark support and binary unit tests | `cargo test --test benchmarking_support_tests --bin benchmark_ltembed` | — | Yes |
| Example compiles | `cargo check --example api_usage` | — | Yes |
| Example unit test | `cargo test --example api_usage` | — | **No** |
| Tokenizer parity (`src/engine/tokenizer_parity.rs`) | `cargo test --lib tokenizer_parity -- --ignored` | A GGUF bundle in `LTEMBED_TEST_BUNDLE_DIR` | Yes (required) |
| Integration tests, Tier 1 | `cargo test --test integration_tests` | — | Yes |
| Integration tests, Tier 2 | same | A GGUF bundle in `LTEMBED_TEST_BUNDLE_DIR` | Yes (required) |
| Python script tests | `python3 -m pytest tests/ -q` | `pytest`, `numpy`; `torch` (and `sentence-transformers` for one) for two modules | Yes, without `torch` |
| Format | `cargo fmt --check` | Nothing (no build script) | Yes |
| Clippy | `cargo clippy --all-targets -- -D warnings` | — | Yes |
| Rustdoc | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` | — | Yes |
| Markdown links and `AGENTS.md` paths | `python3 scripts/check_doc_references.py` | Git and Python 3 only (no build) | Yes |

`tokenizer_reload_tests` is its own test binary because it installs a counting global
allocator. It checks that loading and dropping `HFTokenizer` repeatedly does not keep
growing memory, which guards the BPE-cache workaround in `src/traits/tokenizer.rs`.

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
4. `cargo test --test tokenizer_reload_tests`
5. `cargo test --test benchmarking_support_tests --bin benchmark_ltembed`
6. `cargo check --example api_usage`
7. `python3 scripts/assemble_gguf_bundle.py ci_gguf_bundle`: the UD-Q4_K_XL GGUF and its
   tokenizer at the pinned Hugging Face revisions, checked against their pinned sha256
   (see [bundle-format.md](./bundle-format.md#assembling-a-bundle)); then export
   `LTEMBED_TEST_BUNDLE_DIR`.
8. `cargo test --lib tokenizer_parity -- --ignored`: the
   [tokenizer parity test](#tokenizer-parity), which
   plain `cargo test --lib` (step 3) skips as ignored.
9. `cargo test --test integration_tests`. The llama.cpp reference test embeds an
   8192-token input, which takes about 10 GB of memory
   ([architecture.md](./architecture.md#long-inputs-and-memory)); GitHub's standard Linux
   runners have 16 GB.

**`Python script tests`**: Python 3.13, `pip install pytest numpy`, then
`python3 -m pytest tests/ -q`. `torch` and `sentence-transformers` are not installed, so
`tests/test_bench_pytorch.py` and `tests/test_compare_q4f16_onnx_vs_pytorch.py` skip.

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

- missing `model.gguf`, `tokenizer.json` or `build-info.json` → `ModelLoad(MissingFile)`
- malformed `build-info.json` → `ModelLoad(Metadata)`
- unsupported `input_kind` → `ModelLoad(UnsupportedInputKind)`, and the previous model's
  `"pooling": "last_token"` → `ModelLoad(UnsupportedPooling)`
- `output_dimension` larger than raw → `ModelLoad(Config)`
- input over 8192 tokens → `InputTooLong { max: 8192 }`. This uses `assets/tokenizer.json`
  and skips if that file is missing.
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

The `build-info.json` range check for `max_length` (`1..=8192`) is covered by unit tests in
`src/engine/bundle.rs`.

**Tier 2** needs a real bundle in `LTEMBED_TEST_BUNDLE_DIR`:

| Test | Bundle files read | Checks |
|---|---|---|
| `test_golden_parity_at_every_trained_dimension` | all three | The 21 short golden inputs at widths 768, 512, 256 and 128: width, finite values, unit norm, cosine ≥ 0.985 per input and ≥ 0.993 on average against the golden truncated to that width, and each query ranks its relevant document first |
| `test_llama_cpp_reference_parity` | all three | All 25 golden inputs, up to 8192 tokens: the GGUF's sha256 is the one the reference was generated from; `tokenizer.json` gives the token ids the GGUF's own vocabulary gave upstream llama.cpp; the raw 768-wide vector has cosine ≥ 0.99999 and a norm within 1e-4 of the reference for this variant; and each input passes the per-input golden gate at 768 and 512 |
| `test_token_ids_match_python_tokenizers` | `tokenizer.json` only | `HFTokenizer` ids, masks and type ids equal the Python fixture exactly |
| `test_embed_batch_matches_single_embeds_and_repeats_exactly` | all three | `embed_batch` returns one vector per input, in order, bit-identical to embedding each alone, and repeating the call returns bit-identical vectors |
| `test_query_and_document_prompts_differ` | all three | The same text as a query and as a document gives different vectors |
| `test_default_config_output_is_normalized_and_512_wide` | all three | `EngineConfig::default()` output has 512 values and unit norm |
| `test_bundle_max_length_is_enforced_without_truncation` | all three | With `max_length` 64, an input of exactly 64 tokens embeds and one of 65 returns `InputTooLong { tokens: 65, max: 64 }` |
| `test_gguf_width_mismatch_returns_model_load_error` | all three | With `raw_embedding_dimension` 1024, loading fails with `ModelLoad(Runtime)` |
| `test_hidden_width_as_raw_dimension_returns_model_load_error` | all three | With `raw_embedding_dimension` 512, the hidden width, loading fails: the engine must return the projected 768-wide vector |
| `test_gguf_without_mean_pooling_returns_model_load_error` | all three | A copy of the GGUF with `<arch>.pooling_type` changed to 3 (last) fails to load with `ModelLoad(Runtime)` |
| `test_tokenizer_vocab_mismatch_returns_model_load_error` | `model.gguf`, `build-info.json` | With `assets/tokenizer.json` (30522 tokens), loading fails, naming both vocabulary sizes |
| `test_tokenizer_without_eos_returns_model_load_error` | all three | The bundle's tokenizer with `<eos>` removed from its template (same vocabulary size) fails the load-time tokenization check |

The GGUF copy in `test_gguf_without_mean_pooling_returns_model_load_error`, about 176 MB,
goes to Cargo's integration-test scratch directory (`CARGO_TARGET_TMPDIR`, `target/tmp/` by
default) rather than `/tmp`, and is removed afterwards, even when the test fails. No crafted
GGUF is committed. The tests that swap one bundle file symlink the others instead of
copying them.

### Skipping vs. failing without a bundle

By default a Tier 2 test **skips**: it returns early, passes, and prints `Skipping: …`. That
happens when `LTEMBED_TEST_BUNDLE_DIR` is unset, or when the directory lacks `model.gguf` or
`tokenizer.json`. The token-id test checks only `tokenizer.json`, so a directory holding
just that file is enough to run it. Local runs without a bundle therefore pass.

`build-info.json` is not part of that check. If the directory has `model.gguf` and
`tokenizer.json` but no `build-info.json`, the engine-backed tests do not skip. Each one
fails when it builds the engine, with `ModelLoad(MissingFile)`, whether or not
`LTEMBED_REQUIRE_TEST_BUNDLE` is set.

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

Under qemu emulation the model-backed tests are slow: the llama.cpp reference test alone
embeds about 13,500 tokens of long input.

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
prompt, pooling, tokenizer or width) moves it further. The gates sit between the two:
per-input cosine ≥ **0.985**, mean ≥ **0.993**, at every width.

Cosine of LTEmbed's output against the golden, over all 25 inputs at 768 dimensions, for
each GGUF of the pinned revision (x86-64-v3, flash attention on):

| GGUF | Minimum | Mean |
|---|---|---|
| `UD-Q4_K_XL` (release) | 0.98944 | 0.99471 |
| `UD-Q5_K_XL` | 0.9969 | 0.9985 |
| `UD-Q6_K_XL` | 0.9983 | 0.9992 |
| `Q8_0` | 0.9996 | 0.99985 |
| `BF16` | 0.99999 | 0.99999+ |

`BF16` agrees with the F32 reference on every input, including the 8192-token one, so
tokenization, prompts, pooling, projection and LTEmbed's attention settings are right and
the remaining gap is weight quantization, which shrinks with each larger quant. For the
release quant at the other widths: 512, minimum 0.98972 and mean 0.99486; 256, minimum
0.98998 and mean 0.99530; 128, minimum 0.99415 and mean 0.99685. On aarch64 the release
quant's vectors differ slightly (see the llama.cpp reference below); there the minimum at
768 is 0.98985 and the mean 0.99485, from the aarch64 reference vectors that LTEmbed must
match.

The previous model's gate was 0.99 per input. The 4-bit release GGUF cannot meet that: its
worst input, a Python code snippet, scores 0.9894. The gates hold for these 25 inputs, not
for every text: over 2000 short CN/EN sentences, 1% of the release quant's vectors fall below
0.985 ([benchmarking.md](./benchmarking.md#recorded-results)). Keeping 0.99 would fail correct output;
the gates were set just below the measured quantization error instead, and the llama.cpp
reference test below catches implementation errors with a far tighter gate.

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
| Prompt applied twice | 0.9985 (median 0.991) | mean gate; llama.cpp reference |
| `<eos>` missing | 0.9856 (median 0.973) | mean gate; load-time tokenizer check |

Every mistake also fails the llama.cpp reference test, which uses the correct prompt and
the GGUF's own tokenizer.

## llama.cpp reference: `tests/fixtures/llama_cpp_reference.json`

The golden can only show LTEmbed is within quantization error of the F32 model. This
fixture shows LTEmbed computes what llama.cpp itself computes on the same GGUF: upstream
llama.cpp's `llama-embedding` example (`examples/embedding/embedding.cpp`), built from the
commit the pinned static release compiles and linked against that release's own archives,
run on the release GGUF for every golden input. It runs the same kernels as LTEmbed but
none of LTEmbed's code: it tokenizes with the GGUF's vocabulary, not `tokenizer.json`, and
builds its own batch. The settings are LTEmbed's: mean pooling, non-causal attention, flash
attention on, `n_ctx = n_batch = n_ubatch = 8192`, no normalization.

The kernels differ between the variants (their vectors agree to a mean cosine of 0.99963,
and 0.99907 on the worst input), so the fixture holds one set of raw 768-wide vectors per
variant, plus the token ids the GGUF
gave, the GGUF identity, the llama.cpp commit and the tool's arguments. LTEmbed matches its
own variant's vectors to the tool's 7-decimal output: the gate is cosine ≥ 0.99999 and a
norm within 1e-4. Measured on x86-64-v3, the two agree to cosine 1 − 4·10⁻¹⁶.

`scripts/generate_llama_cpp_reference.py` writes one variant's section and keeps the
other's. `scripts/build_llama_cpp_reference_tool.sh` builds the tool from a llama.cpp
checkout at the release's commit and an extracted release (see its header for the
details):

```bash
COMMIT=cbe28bb3cb0758f0c566a9a3561766821aeb0748   # STATIC_LLAMA_LLAMA_CPP_COMMIT
git init .llama-artifacts/llama.cpp && git -C .llama-artifacts/llama.cpp fetch --depth 1 \
  https://github.com/ggml-org/llama.cpp "$COMMIT" && git -C .llama-artifacts/llama.cpp checkout FETCH_HEAD
python3 scripts/assemble_gguf_bundle.py .llama-artifacts/bundle

# x86-64-v3 host, x86_64-v3 release in .llama-artifacts/extracted-x86_64-v3:
scripts/build_llama_cpp_reference_tool.sh .llama-artifacts/llama.cpp \
  .llama-artifacts/extracted-x86_64-v3 .llama-artifacts/reftool-x86_64-v3
python3 scripts/generate_llama_cpp_reference.py --variant x86_64-v3 \
  --tool .llama-artifacts/reftool-x86_64-v3/llama-embedding --gguf .llama-artifacts/bundle/model.gguf

# aarch64: build in the linux/arm64 container, statically so that it also runs under qemu.
docker run --rm --platform linux/arm64 -v "$PWD:/work" -w /work rust:1.94.0 \
  scripts/build_llama_cpp_reference_tool.sh --static .llama-artifacts/llama.cpp \
  .llama-artifacts/extracted .llama-artifacts/reftool-aarch64
python3 scripts/generate_llama_cpp_reference.py --variant aarch64-graviton2 \
  --tool .llama-artifacts/reftool-aarch64/llama-embedding --gguf .llama-artifacts/bundle/model.gguf
```

With podman on an SELinux host, add `--security-opt label=disable` as in
[development.md](./development.md#container-builds). The generator checks the GGUF's
sha256, that the tool reports the pinned commit, that each input gives the token count the
golden recorded, and that both variants saw the same token ids. Use `--reset` when the
golden inputs change, then generate the other variant. Under qemu on an x86_64 host the
aarch64 run takes hours: the 4074- and 8192-token inputs dominate.

## Token-id fixture: `tests/fixtures/token_ids.json`

`scripts/generate_token_ids.py` encodes a fixed set of inputs with the Python `tokenizers`
library, singly and as one batch. It uses the `tokenizer.json` at the revision
`scripts/assemble_gguf_bundle.py` pins. The fixture records:

- `tokenizer`: repo, `revision` (`c8f89cc7…`) and the file's `sha256`
- `tokenizers_version` (`0.23.2`): the script refuses to run unless the installed Python
  `tokenizers` equals the Rust `tokenizers` version in `Cargo.lock`
- `padding`: `null`. EmbeddingGemma 2's tokenizer pads nothing, so the script requires each
  batch row to equal the single encoding and stores only that.
- `cases`: 15 inputs, as literal `text` or as a `repeat` spec (long inputs are never stored
  literally): prompted English and CJK text, emoji, mixed whitespace, byte fallback, the
  empty string, added-token text in several forms, a run of 100,000 spaces (3230 tokens),
  and 6009 tokens of repeated mixed Chinese and English.

If the bundle's `tokenizer.json` hash differs from the recorded one, the test fails with
instructions to regenerate, instead of a bare id mismatch.

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

`src/engine/tokenizer_parity.rs` checks that the bundle's two tokenizers give identical
token ids. One is `tokenizer.json`, run by the `tokenizers` crate, which the engine
tokenizes inference inputs with. The other is the GGUF's own vocabulary
(`tokenizer.ggml.*`), run by `llama_tokenize`, which is what upstream llama.cpp tokenizes
with. The load-time check compares the two on one text (`TOKENIZER_CHECK_TEXT`) and the
[llama.cpp reference test](#llamacpp-reference-testsfixturesllama_cpp_referencejson) on the
25 golden inputs; this test compares them on a fixed corpus of 25,401 texts:

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

On the pinned bundle all 76,203 encodings are identical. On x86-64-v3 this was measured
locally, and the test takes about 18 s in a debug build. CI runs it on both variants.

**What a pass shows.** The two tokenizers agree on this corpus, for the pinned
`tokenizer.json`, GGUF, `tokenizers` version and static llama.cpp release. It is not a
proof for every input. Neither is checked against the tokenizer the model was trained with;
for `tokenizer.json` against Python `tokenizers`, see the token-id fixture.

**How sensitive it is.** Run with the real GGUF and a deliberately altered copy of
`tokenizer.json`:

| Change to `tokenizer.json` | Encodings that differ |
|---|---|
| `<eos>` removed from the template | 76,203 (all) |
| NFC normalization added before the space replacement | 14,916 |
| `<mask>` no longer an added token | 2,712 |
| One BPE merge removed (`▁th` + `e`) | 0 |

Removing one merge changes nothing because the BPE reaches the same tokens through other
merges. Python `tokenizers` also gives unchanged ids for all 5268 Chinese and English
sentences after removing any one of nine merges tried.

**Running it.** The test needs the GGUF, so it is `#[ignore]`d and plain `cargo test --lib`
stays model-free:

```bash
LTEMBED_REQUIRE_TEST_BUNDLE=1 LTEMBED_TEST_BUNDLE_DIR="$PWD/gguf_bundle" \
  cargo test --lib tokenizer_parity -- --ignored
```

- **Skip or fail**: it follows the same rules as the Tier 2 integration tests
  ([Skipping vs. failing](#skipping-vs-failing-without-a-bundle)). It needs `model.gguf` and
  `tokenizer.json` and does not read `build-info.json`.
- **Corpus check**: the model-free unit test `test_corpus_reads_every_source`, part of
  `cargo test --lib`, checks that every source loads and that the `repeat` specs are
  expanded.
- **Expected warnings**: texts that start with `<bos>` or end with `<eos>` make llama.cpp
  print `check_double_bos_eos` warnings, about 560 per run. Both tokenizers add a second BOS
  or EOS there.

There is no fixture to regenerate: the test computes both sides on every run. If it fails
after a bump of `tokenizers`, the tokenizer revision or the static llama.cpp release, that is
a finding. It means the engine no longer tokenizes the way llama.cpp reads the GGUF.

## Python tests

`tests/test_*.py` cover the scripts in `scripts/`: the bundle assembler, the benchmark
orchestrator, report renderer, metadata writer, retrieval-case builder, the three reference
generators, embedding comparison, documentation reference check, the PyTorch runner and
the ONNX-era `compare_q4f16_onnx_vs_pytorch.py`. They do not build LTEmbed or need a bundle
or model weights. The orchestrator tests check the `cargo run` command line without running
it. `tests/test_assemble_gguf_bundle.py` and `tests/test_reference_generators.py` also check
that the committed fixtures, the generators, the bundle pins and the static release pins in
`ci.yml` still name the same files.
`tests/CN_EN_Data.csv` is the source data for the CN/EN retrieval evaluation.

`tests/test_pre_push_hook.py` covers `.githooks/pre-push` and the x86-64-v3 check in
`.github/scripts/x86-64-v3.sh`. It runs the hook in a scratch Git repository with stub
`uname`, `cargo` and `docker` commands and a stub CPU check, so on any machine it checks
which hosts run clippy natively and which use the container, without compiling anything.
The real CPU check runs against a fake dynamic loader and `/proc/cpuinfo`. It needs `bash`
and `git`.

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
