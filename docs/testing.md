# Testing

What each test suite checks, what it needs, and exactly what CI runs. Every Rust target
links the static llama.cpp archives, so all `cargo test` and `cargo clippy` commands need
aarch64 Linux with `STATIC_LLAMA_DIR` set. On other hosts, run them in the container from
[development.md](./development.md#non-aarch64-hosts).

## Suites

| Suite | Command | Needs beyond the build | Run by CI |
|---|---|---|---|
| Library unit tests (`src/**`) | `cargo test --lib` | `assets/tokenizer.json` (tracked) for tokenizer shape tests | Yes |
| Tokenizer reload memory test | `cargo test --test tokenizer_reload_tests` | — (writes its own BPE tokenizer to a temp file) | Yes |
| Benchmark support and binary unit tests | `cargo test --test benchmarking_support_tests --bin benchmark_ltembed` | — | Yes |
| Example compiles | `cargo check --example api_usage` | — | Yes |
| Example unit test | `cargo test --example api_usage` | — | **No** |
| Integration tests, Tier 1 | `cargo test --test integration_tests` | — | Yes |
| Integration tests, Tier 2 | same | A GGUF bundle in `LTEMBED_TEST_BUNDLE_DIR` | Yes (required) |
| Python script tests | `python3 -m pytest tests/ -q` | `pytest`, `numpy`; `torch` for two modules | Yes, without `torch` |
| Format | `cargo fmt --check` | Nothing (no build script) | Yes |
| Clippy | `cargo clippy --all-targets -- -D warnings` | — | Yes |

`tokenizer_reload_tests` is its own test binary because it installs a counting global
allocator. It checks that loading and dropping `HFTokenizer` repeatedly does not keep
growing memory, which guards the BPE-cache workaround in `src/traits/tokenizer.rs`.

## What CI runs

`.github/workflows/ci.yml` runs on every push and pull request to `main`, on
`ubuntu-24.04-arm`. It has three jobs.

**`Test`** (job env `LTEMBED_REQUIRE_TEST_BUNDLE=1`):

1. `.github/scripts/fetch-static-llama.sh`: download, verify and extract the pinned static
   llama.cpp release, then export `STATIC_LLAMA_DIR`.
2. `mkdir -p assets`
3. `cargo test --lib`
4. `cargo test --test tokenizer_reload_tests`
5. `cargo test --test benchmarking_support_tests --bin benchmark_ltembed`
6. `cargo check --example api_usage`
7. Assemble `ci_gguf_bundle/` (Q5_K_M at the pinned Hugging Face revision; see
   [bundle-format.md](./bundle-format.md#assembling-a-bundle-by-hand)) and export
   `LTEMBED_TEST_BUNDLE_DIR`.
8. `cargo test --test integration_tests`

**`Python script tests`**: Python 3.13, `pip install pytest numpy`, then
`python3 -m pytest tests/ -q`. `torch` is not installed, so `tests/test_bench_pytorch.py`
and `tests/test_compare_q4f16_onnx_vs_pytorch.py` skip.

**`Lint`**: fetch static llama, `cargo fmt --check`, then
`cargo clippy --all-targets -- -D warnings`.

CI does not run plain `cargo test`, doc tests, the example's unit test, or any benchmark.
Benchmarks run only by hand; see [benchmarking.md](./benchmarking.md).

## Integration tests: Tier 1 and Tier 2

`tests/integration_tests.rs` mixes two kinds of tests.

**Tier 1** never needs model weights. These tests build throwaway bundles in a temp
directory and check load-time validation:

- missing `model.gguf`, `tokenizer.json` or `build-info.json` → `ModelLoad(MissingFile)`
- malformed `build-info.json` → `ModelLoad(Metadata)`
- unsupported `input_kind` or `pooling` → `ModelLoad(UnsupportedInputKind / UnsupportedPooling)`
- `output_dimension` larger than raw → `ModelLoad(Config)`
- input over 8192 tokens → `InputTooLong { max: 8192 }`. This uses `assets/tokenizer.json`
  and skips if that file is missing.
- the skip and require logic itself (`bundle_or_skip`)

**Tier 2** needs a real bundle in `LTEMBED_TEST_BUNDLE_DIR`:

| Test | Bundle files read | Checks |
|---|---|---|
| `test_golden_parity_cosine_similarity` | `model.gguf`, `tokenizer.json`, `build-info.json` | Cosine > 0.99 against each golden vector |
| `test_token_ids_match_python_tokenizers` | `tokenizer.json` only | `HFTokenizer` ids, masks and type ids equal the Python fixture exactly |
| `test_embed_batch_consistency` | `model.gguf`, `tokenizer.json`, `build-info.json` | `embed_batch(..)[0] == embed(..)` |
| `test_output_is_l2_normalized` | `model.gguf`, `tokenizer.json`, `build-info.json` | Unit norm |
| `test_output_dimension_is_512` | `model.gguf`, `tokenizer.json`, `build-info.json` | Output length 512 |

The Tier 2 tests build the engine with a 512-d, L2-normalized `EngineConfig`, regardless of
the bundle's `output_embedding_dimension`.

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

Under qemu emulation the token-id test alone can take a few minutes, because one case
tokenizes a 1M-space input.

## Golden fixture: `tests/fixtures/test_fixtures.json`

This is the **immutable PyTorch FP32 reference**. It holds four inputs (three queries, one
document), each with its final 512-d, truncated and L2-normalized vector:

```json
{
  "model": "jinaai/jina-embeddings-v5-text-nano-retrieval",
  "raw_dim": 768,
  "dim": 512,
  "max_length": 8192,
  "fixtures": [{ "kind": "query", "text": "Hello, world!", "embedding": [/* 512 floats */] }]
}
```

- `text` is raw caller text. The prefix is applied by the generator and by the engine, never
  stored.
- `scripts/generate_fixtures.py` produces it with `transformers` + `torch`
  (`pip install transformers torch numpy`). It loads the model with
  `trust_remote_code=True` in float32 (the model's native dtype is bfloat16), then applies
  last-token pooling, truncation to 512 and L2 normalization.
- **Never regenerate it from GGUF or LTEmbed output.** The golden measures how far the
  quantized llama.cpp path drifts from the original model. Regenerating it from GGUF
  output would make that check meaningless.
- Regenerate it, from PyTorch, only when the reference itself changes: a different model,
  different `TEST_INPUTS`, or a change to pooling, prefixes or output dimension. The script
  downloads the model's latest Hugging Face revision; it does not pin one.
- If `dim` is not 512, the parity test skips rather than compare against the wrong
  baseline. `LTEMBED_REQUIRE_TEST_BUNDLE` does not affect this skip. The file once shipped
  with `dim: 0`, and parity was silently skipped until the llama.cpp migration.

## Token-id fixture: `tests/fixtures/token_ids.json`

`scripts/generate_token_ids.py` encodes a fixed set of inputs with the Python `tokenizers`
library, once singly and once as one padded batch. It uses the `tokenizer.json` at the
Hugging Face revision that CI pins. The fixture records:

- `tokenizer`: repo, `revision` (`ac5d898c…`) and the file's `sha256`
- `tokenizers_version` (`0.23.2`): the script refuses to run unless the installed Python
  `tokenizers` equals the Rust `tokenizers` version in `Cargo.lock`
- `padding`: right padding, `pad_id` `128004`, `pad_type_id` `0`
- `cases`: each input as literal `text` or as a `repeat` spec (the 1M-space input is never
  stored literally), with its single encoding and its row of the batch encoding. Batch
  rows store trailing padding as a count.

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

`REVISION` in `scripts/generate_token_ids.py` must equal `HF_REVISION` in
`.github/workflows/ci.yml`. Change them together.

## Python tests

`tests/test_*.py` cover the scripts in `scripts/`: the benchmark orchestrator, report
renderer, metadata writer, retrieval-case builder, token-id generator, embedding comparison,
the PyTorch runner and the ONNX-era `compare_q4f16_onnx_vs_pytorch.py`. They do not build
LTEmbed or need a bundle. The orchestrator tests check the `cargo run` command line without
running it.
`tests/CN_EN_Data.csv` is the source data for the CN/EN retrieval evaluation.
