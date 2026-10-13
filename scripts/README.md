# Scripts

Python tooling for fixtures, benchmarks and documentation checks, plus the Git hook
installer. How the pieces fit together is described in
[docs/benchmarking.md](../docs/benchmarking.md) and [docs/testing.md](../docs/testing.md).
The script tests in `tests/test_*.py` cover most of these files.

## Bundles

### `assemble_gguf_bundle.py`

Assembles a GGUF bundle (`model.gguf`, `tokenizer.json`, `build-info.json`, plus
`SHA256SUMS` and `provenance.json`) for EmbeddingGemma 2. It is the single place that pins
the bundle's sources: `GGUF_REPO` / `GGUF_REVISION` and one sha256 per quant in
`GGUF_FILES`, `TOKENIZER_REPO` / `TOKENIZER_REVISION` / `TOKENIZER_SHA256`, and the
`build-info.json` it writes. CI, the release workflow and the ARM64 benchmark all call it.
Python 3 standard library only.

```bash
python3 scripts/assemble_gguf_bundle.py gguf_bundle                  # UD-Q4_K_XL
python3 scripts/assemble_gguf_bundle.py gguf_bundle-q8 --quant Q8_0
python3 scripts/assemble_gguf_bundle.py gguf_bundle --gguf PATH --tokenizer PATH  # local copies
```

It downloads from `resolve/<revision>` only, rejects any file whose sha256 differs from its
pin, and leaves no partial bundle on failure. Details are in
[docs/bundle-format.md](../docs/bundle-format.md#assembling-a-bundle).

## Fixtures

### `generate_fixtures.py`

Regenerates `tests/fixtures/test_fixtures.json`, the immutable PyTorch F32 golden for
EmbeddingGemma 2, from the safetensors model `unsloth/embeddinggemma-2` at the pinned
`REVISION` (which must equal `TOKENIZER_REVISION` in `assemble_gguf_bundle.py`). Never
regenerate it from GGUF or LTEmbed output. Requires `sentence-transformers`, `transformers`,
`torch`, `torchvision`, `pillow` and `numpy`; the docstring has the exact command.

- runs the model's own SentenceTransformers pipeline in float32 on the CPU, text tower only
- prompts: the model's `query` and `document` prompts, applied by SentenceTransformers
- pooling: mean over every token, then the 512→768 projection, then L2 normalization
- stores the full 768-wide vector of each of 25 inputs (21 short, 4 of 89 to 8192 tokens);
  the tests derive 512, 256 and 128 by truncating and renormalizing

### `generate_token_ids.py`

Regenerates `tests/fixtures/token_ids.json`, the token-id parity fixture for `HFTokenizer`
(`pip install tokenizers==<version>` with the Rust `tokenizers` version from `Cargo.lock`,
which the script checks; no model weights). It downloads `tokenizer.json` at the pinned
`REVISION`, which must equal `TOKENIZER_REVISION` in `assemble_gguf_bundle.py`, and records
the single encodings, after checking that a batch encoding pads nothing, plus the file's
sha256. Rerun it after changing its inputs, bumping the revision, or bumping `tokenizers` in
`Cargo.lock`.

### `generate_llama_cpp_reference.py`

Regenerates one variant's section of `tests/fixtures/llama_cpp_reference.json`: the raw
768-wide vectors and token ids that upstream llama.cpp's own `llama-embedding` tool
computes for every golden input on the release GGUF. Python 3 standard library only; it
needs the tool, built by `build_llama_cpp_reference_tool.sh`, and the release GGUF. See
[docs/testing.md](../docs/testing.md#llamacpp-reference-testsfixturesllama_cpp_referencejson).

### `build_llama_cpp_reference_tool.sh`

Builds `llama-embedding` from a llama.cpp checkout at the commit the pinned static release
was built from, linked against that release's own archives, for
`generate_llama_cpp_reference.py`. It refuses a checkout at another commit or a release for
another architecture. `--static` links a fully static binary, for an aarch64 build made in
the `linux/arm64` container that must also run under qemu.

## Benchmarks

### `run_embedding_benchmarks.py`

Top-level orchestrator for the LTEmbed and PyTorch runners. It launches
`target/release/benchmark_ltembed` if that file exists, and otherwise
`cargo run --release --bin benchmark_ltembed`.

Important assumptions:

- `--bundle-dir` points at a GGUF bundle directory containing `model.gguf`, `tokenizer.json`, and `build-info.json`
- `--model-dir` points at a local Hugging Face snapshot of `unsloth/embeddinggemma-2` for the PyTorch runner (the files SentenceTransformers needs, at the pinned revision). The default, `assets/`, is not such a snapshot.
- `--cold-iters` runs the cold-start pass N times per scenario (fresh process each time) and aggregates the latency distribution Python-side
- `--golden-parity` re-embeds the texts from the immutable `tests/fixtures/test_fixtures.json` golden and reports per-item cosine similarity against the golden truncated to `--output-dimension` (`mode=golden_parity` CSV rows). It is skipped unless `--output-dimension` is one of the fixture's widths (768, 512, 256, 128). The golden file is never written
- `--emit-reference <path>` runs only the PyTorch retrieval pass, writes its embeddings to `<path>`, and exits. `--reference-path <path>` loads that file instead of running PyTorch, so only LTEmbed runs. The `benchmark-arm64` workflow uses this pair to run PyTorch once for all quants.
- `--output-dimension` and `--l2-normalize` describe LTEmbed post-processing explicitly
- correctness and golden-parity thresholds (`--correctness-threshold`, `--golden-parity-threshold`, both `0.98` by default) should allow for the drift of a quantized GGUF model from the PyTorch FP32 reference
- `--threads` is passed to PyTorch as `torch.set_num_threads(...)` and to `benchmark_ltembed --threads`, which sets llama.cpp `n_threads` and `n_threads_batch`; the CSV `threads` column records this requested runner thread count
- Historical note (ONNX Runtime era): before LTEmbed passed `--threads` to ONNX Runtime's `with_intra_threads(...)`, LTEmbed rows recorded the requested `--threads` value in the CSV while the engine actually ran with 1 intra-op thread; LTEmbed rows with `threads > 1` produced before that fix are mislabeled and must not be compared against later runs (relevant when using `compare_benchmarks.py` across runs)
- `--ltembed-cargo-features` is only used by the `cargo run` fallback. The crate defines no Cargo features, so leave it empty.

### `bench_pytorch.py`

PyTorch retrieval-eval reference runner (retrieval-only; other `--mode` values exit with an
error). It runs EmbeddingGemma 2's SentenceTransformers pipeline in float32 with the model's
own prompts, one input at a time, and refuses inputs over 8192 tokens instead of truncating
them. Requires the same packages as `generate_fixtures.py`.

### `benchmark_corpus.json`

Deterministic texts for the `single/medium`, `single/long`, and `batch/medium/8` benchmark
scenarios. Embedded into the Rust binary via `include_str!` and read by
`build_cn_en_retrieval_cases.py`, so both sides embed byte-identical inputs.

### `retrieval_eval_cases.json`

A small built-in retrieval-eval case (`mini-retrieval-v1`). It is the default
`--retrieval-eval-path` of `run_embedding_benchmarks.py`.

### `build_cn_en_retrieval_cases.py`

Generates the CN/EN cross-lingual retrieval-eval case from `tests/CN_EN_Data.csv` and, with
`--fixture-output`, the resolved latency fixture covering every benchmark scenario.

### `write_benchmark_metadata.py`

Writes a quant matrix job's `metadata.json` (model/bundle sizes and SHA, static llama
tag/SHA/contract version, runner + CPU flags, run parameters, scenario list) with proper
JSON number types. Consumed by `render_benchmark_report.py`.

### `render_benchmark_report.py`

Aggregates every quant's `metadata.json` + `benchmark-report.csv` into the cross-quant
`results.json` + `report.md` (normalized per quant × scenario × warm/cold records, parity
and retrieval summaries, and the recommended quant under the Lambda bundle-size budget).
The recommendation rules are in
[docs/benchmarking.md](../docs/benchmarking.md#how-the-report-recommends-a-quant).

### `compare_benchmarks.py`

Compares two or more `benchmark-report.csv` files. The first file is the baseline.

```bash
python3 scripts/compare_benchmarks.py baseline.csv candidate.csv
python3 scripts/compare_benchmarks.py --label before:a.csv after:b.csv --metric p95_ms
```

By default it compares `mean_ms` of `warm_latency` rows; see `--mode`, `--metric` and
`--impl`.

### `compare_embedding_outputs.py`

Compares one scenario's embeddings between an LTEmbed and a PyTorch JSON payload and
reports cosine and norms. It expects payloads with `results[].scenario` and
`results[].embeddings`, the output of the old `correctness` mode. Neither current runner
produces that shape: `benchmark_ltembed` lost its `correctness` mode in #149, and
`bench_pytorch.py` runs only retrieval.

### `compare_q4f16_onnx_vs_pytorch.py`

ONNX Runtime era. Compares a q4f16 ONNX model, run through `onnxruntime`, with the PyTorch
reference. It does not apply to GGUF bundles.

## Documentation

### `check_doc_references.py`

Fails if a relative link in a tracked Markdown file, or its `#anchor`, does not resolve, or
if an `AGENTS.md` names a repository path or a `file` § "Section" that does not exist. It
never fetches external URLs and needs only Git and the Python 3 standard library. CI runs it
in the `Lint` job.

```bash
python3 scripts/check_doc_references.py
```

What it checks, and how to let an `AGENTS.md` mention a name that is not a repository path,
is in [docs/testing.md](../docs/testing.md#documentation-references).

## Git hooks

### `install-git-hooks.sh`

Sets `core.hooksPath=.githooks` for this clone. See
[docs/development.md](../docs/development.md#git-hooks).
