# Releases

What a release contains, how it is produced, and which model quant it ships. Everything
here comes from `.github/workflows/release-bundles.yml`.

As of 2026-10-06, the GitHub repository has no tags and no published releases. The crate
version in `Cargo.toml` is `0.1.0`. This repository has no workflow that publishes to
crates.io.

## What a release tarball contains

One tarball per matrix entry. There is currently one entry, `UD-Q5_K_XL`:

```
ltembed-<version>-unsloth__embeddinggemma-2-GGUF-UD-Q5_K_XL-linux-arm64.tar.gz
ltembed-<version>-unsloth__embeddinggemma-2-GGUF-UD-Q5_K_XL-linux-arm64.tar.gz.sha256
└── ltembed-<version>-unsloth__embeddinggemma-2-GGUF-UD-Q5_K_XL-linux-arm64/
    ├── <source tree>      git archive of the commit being built
    └── gguf_bundle/
        ├── model.gguf         embeddinggemma-2-UD-Q5_K_XL.gguf
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

## Why UD-Q5_K_XL

The release and CI bundles use `embeddinggemma-2-UD-Q5_K_XL.gguf`, Unsloth's dynamic 5-bit
quant, from [`unsloth/embeddinggemma-2-GGUF`](https://huggingface.co/unsloth/embeddinggemma-2-GGUF)
at revision `031f0d4b…`. It is the most accurate pinned quant that meets the size budget:

| GGUF | Bundle | Fits 230 MiB | Golden cosine, min / mean (768) | CN/EN min (512) |
|---|---|---|---|---|
| `UD-Q4_K_XL` | 175.7 MB (167.5 MiB) | yes | 0.98944 / 0.99471 | 0.9614 |
| `UD-Q5_K_XL` | 210.1 MB (200.3 MiB) | yes | 0.99690 / 0.99849 | 0.9911 |
| `UD-Q6_K_XL` | 248.8 MB (237.3 MiB) | no | 0.9983 / 0.9992 | 0.9949 |
| `Q8_0` | 309.9 MB (295.5 MiB) | no | 0.9996 / 0.99985 | 0.9986 |
| `BF16` | 558.0 MB (532.1 MiB) | no | ≥ 0.99999 | 0.9999 |

- **Size**: the 230 MiB budget is the AWS Lambda unzipped package limit of 250 MiB minus a
  20 MiB allowance for a binary (see [AWS Lambda](#aws-lambda)). A bundle is its GGUF plus
  about 1.4 KB of `build-info.json`, `SHA256SUMS` and `provenance.json`. `UD-Q5_K_XL` has
  fitted since the bundle dropped the 32 MB `tokenizer.json`; `UD-Q6_K_XL` does not.
- **Parity**: cosine of LTEmbed's output against the PyTorch F32 golden over 25 inputs of
  up to 8192 tokens, measured on x86-64-v3 (aarch64: TODO(aarch64), from the aarch64
  llama.cpp reference). At the default 512 dimensions the release quant scores a minimum of
  0.99702 and a mean of 0.99852. Its worst input is a Rust code snippet. The test gate is
  0.995 for every input at every width; see
  [testing.md](./testing.md#gates-and-how-they-were-set) for how it was set.
- **Tail**: over 2000 short Chinese and English sentences at 512 dimensions, the lowest
  cosine is 0.9911, the 1st percentile 0.9962, and 10 texts (0.5%) fall below 0.995; recall
  stays at the F32 model's level ([benchmarking.md](./benchmarking.md#recorded-results)).
- **Why not `UD-Q4_K_XL`**, the release quant before: it is 32.8 MiB smaller and faster on
  short inputs (`single/zh` 18.4 ms against 24.6 ms at 6 threads on x86-64-v3; about the
  same at 4075 tokens, 5374 ms against 5288 ms), with the same recall on the CN/EN set. But 1% of those 2000 sentences fall
  below 0.985, down to 0.9614, so its vectors drift further from the model's on short
  text. The benchmark report still recommends it, because its quality gate is a mean
  cosine of 0.99 and it picks the smallest bundle that passes
  ([benchmarking.md](./benchmarking.md#how-the-report-recommends-a-quant)); that gate does
  not see the tail. `scripts/assemble_gguf_bundle.py --quant UD-Q4_K_XL` still builds it.
- Where the size budget does not apply, `Q8_0` is the closest to the F32 model;
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
