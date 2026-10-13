# Releases

What a release contains, how it is produced, and which model quant it ships. Everything
here comes from `.github/workflows/release-bundles.yml`.

As of 2026-10-06, the GitHub repository has no tags and no published releases. The crate
version in `Cargo.toml` is `0.1.0`. This repository has no workflow that publishes to
crates.io.

## What a release tarball contains

One tarball per matrix entry. There is currently one entry, `UD-Q4_K_XL`:

```
ltembed-<version>-unsloth__embeddinggemma-2-GGUF-UD-Q4_K_XL-linux-arm64.tar.gz
ltembed-<version>-unsloth__embeddinggemma-2-GGUF-UD-Q4_K_XL-linux-arm64.tar.gz.sha256
└── ltembed-<version>-unsloth__embeddinggemma-2-GGUF-UD-Q4_K_XL-linux-arm64/
    ├── <source tree>      git archive of the commit being built
    └── gguf_bundle/
        ├── model.gguf         embeddinggemma-2-UD-Q4_K_XL.gguf
        ├── build-info.json
        ├── SHA256SUMS
        └── provenance.json
```

- `<version>` is `package.version` from `Cargo.toml`.
- `gguf_bundle/` is what `scripts/assemble_gguf_bundle.py` writes and follows
  [bundle-format.md](./bundle-format.md): the GGUF at its pinned Hugging Face revision,
  checked against its pinned sha256, which is the file CI tests on both architectures.
  `provenance.json` records each file's repository, revision, URL and sha256, the upstream
  model and its license, and the LTEmbed commit and workflow run.
- The workflow fails if the bundle holds any file but these four, or if the archive
  contains an `mmproj` (vision or audio projector) file or any `tokenizer.json`. The engine
  tokenizes with the GGUF's own vocabulary
  ([bundle-format.md](./bundle-format.md#tokenization)).
- The tarball contains **no compiled binary and no static llama.cpp archives**. To build
  from it, you still need aarch64 Linux or x86-64-v3 Linux and `STATIC_LLAMA_DIR`; see
  [development.md](./development.md).
- `linux-arm64` in the name dates from when aarch64 Linux was the only supported build
  platform. The tarball itself holds source code and model files, which build on either
  supported platform.
- Check the download and the bundle:

  ```bash
  sha256sum -c ltembed-*.tar.gz.sha256
  tar -xzf ltembed-*.tar.gz
  cd ltembed-*/gguf_bundle && sha256sum -c SHA256SUMS
  ```

The matrix also sets `output_dimension: 512` and `l2_normalize: true`. They are only
written to the run summary, with the GGUF's repository, revision, file and sha256. They are
not stored in the bundle; the caller chooses them through `EngineConfig`.

## How the workflow runs

| Trigger | Builds the tarball | Uploads a workflow artifact | Creates the GitHub release | Uploads a release asset |
|---|---|---|---|---|
| `push` of a tag matching `v*` | Yes | Yes | Yes, if missing (`--verify-tag --generate-notes`) | Yes |
| `release: published` | Yes | Yes | — (already exists) | Yes |
| `workflow_dispatch` | Yes | Yes | No | No |

- It runs on `ubuntu-24.04-arm` with `permissions: contents: write`.
- Workflow artifacts are kept for 7 days.
- Release assets are uploaded with `gh release upload --clobber`, so a re-run replaces an
  asset with the same name.
- If the tag is not `v<Cargo.toml version>`, the workflow only prints a warning and still
  publishes, with the `Cargo.toml` version in the file name.
- The workflow does **not** build the crate or run tests. It packages whatever commit the
  tag points at, so run CI on that commit first.
- `model.gguf` comes from the same pinned revision and hash as CI's bundle
  (`scripts/assemble_gguf_bundle.py`); no step downloads from a branch such as `main`.
  Changing the release's model file therefore means changing the pins in that script, which
  CI then tests.

### Cutting a release

The repository has no release script. Based on the workflow, the steps are:

1. Set `version` in `Cargo.toml` and merge that change to `main` with CI green.
2. Push a tag `v<version>` that points at that commit:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

3. When the `release-bundles` run finishes, check that the release has the tarball and its
   `.sha256` attached.

To try the packaging without publishing anything, run the workflow by hand
(`workflow_dispatch`) and download the workflow artifact.

## Why UD-Q4_K_XL

The release and CI bundles use `embeddinggemma-2-UD-Q4_K_XL.gguf`, Unsloth's dynamic 4-bit
quant, from [`unsloth/embeddinggemma-2-GGUF`](https://huggingface.co/unsloth/embeddinggemma-2-GGUF)
at revision `031f0d4b…`. It is the smallest pinned quant, it meets the size budget, and its
quantization error is within the gates the tests apply:

| GGUF | Bundle | Fits 230 MiB | Golden cosine, min / mean (768) |
|---|---|---|---|
| `UD-Q4_K_XL` | 175.7 MB (167.5 MiB) | yes | 0.98944 / 0.99471 |
| `UD-Q5_K_XL` | 210.1 MB (200.3 MiB) | yes | 0.9969 / 0.9985 |
| `UD-Q6_K_XL` | 248.8 MB (237.3 MiB) | no | 0.9983 / 0.9992 |
| `Q8_0` | 309.9 MB (295.5 MiB) | no | 0.9996 / 0.99985 |
| `BF16` | 558.0 MB (532.1 MiB) | no | ≥ 0.99999 |

- **Size**: the 230 MiB budget is the AWS Lambda unzipped package limit of 250 MiB minus a
  20 MiB allowance for a binary (see [AWS Lambda](#aws-lambda)). A bundle is its GGUF plus
  about 1.4 KB of `build-info.json`, `SHA256SUMS` and `provenance.json`. Since the bundle
  dropped the 32 MB `tokenizer.json`, `UD-Q5_K_XL` fits the budget too.
- **Why not `UD-Q5_K_XL`**: the test gates and the llama.cpp reference fixture are
  calibrated for `UD-Q4_K_XL`, and the benchmark report recommends the smallest bundle that
  passes its quality gate, which is still `UD-Q4_K_XL`. Shipping `UD-Q5_K_XL` would be a
  separate change: recalibrate the gates and regenerate the llama.cpp reference for both
  variants on it.
- **Parity**: cosine of LTEmbed's output against the PyTorch F32 golden over 25 inputs of
  up to 8192 tokens, measured on x86-64-v3 (aarch64: minimum 0.98985, mean 0.99485). At the
  default 512 dimensions the release quant scores a minimum of 0.98972 and a mean of
  0.99486. Its worst input is a Python code snippet. The test gates are 0.985 per input and 0.993 on average; see
  [testing.md](./testing.md#gates-and-how-they-were-set) for how they were set.
- **Tail**: over 2000 short Chinese and English sentences, 1% drift below 0.985, down to
  0.9614, while retrieval recall stays at the F32 model's level; `UD-Q5_K_XL`'s worst is
  0.9911 ([benchmarking.md](./benchmarking.md#recorded-results)).
- The previous model's gate of 0.99 per input is not met by this 4-bit quant on its worst
  input. Where the size budget does not apply, `Q8_0` is the closest to the F32 model;
  `scripts/assemble_gguf_bundle.py --quant Q8_0` builds that bundle.

The `benchmark-arm64` workflow re-checks the size budget and a quality gate for any set of
quants; see [benchmarking.md](./benchmarking.md#how-the-report-recommends-a-quant).

## AWS Lambda

The 250 MB Lambda limit is a **size budget** that drives the quant choice. It is not a
deployment target this repository supports:

- There is no Lambda handler, bootstrap binary, packaging script, layer or deploy workflow
  on `main`.
- The release tarball is source plus a model bundle, not a Lambda package.
- The static llama.cpp archives are built in an Amazon Linux 2023 image; the aarch64
  variant targets Graviton2 (Neoverse N1). Nothing in this repository builds, runs or tests
  LTEmbed on Lambda.

The ONNX Runtime-era Lambda notes are kept for reference in
[history/ort/](./history/ort/ort-rust-lambda-guidelines.md). They do not apply to `main`.
