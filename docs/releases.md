# Releases

What a release contains, how it is produced, and which model quant it ships. Everything
here comes from `.github/workflows/release-bundles.yml`.

As of 2026-10-06, the GitHub repository has no tags and no published releases. The crate
version in `Cargo.toml` is `0.1.0`. This repository has no workflow that publishes to
crates.io.

## What a release tarball contains

One tarball per matrix entry. There is currently one entry, `Q5_K_M`:

```
ltembed-<version>-jinaai__jina-embeddings-v5-text-nano-retrieval-Q5_K_M-linux-arm64.tar.gz
└── ltembed-<version>-jinaai__jina-embeddings-v5-text-nano-retrieval-Q5_K_M-linux-arm64/
    ├── <source tree>      git archive of the commit being built
    └── gguf_bundle/
        ├── model.gguf
        ├── tokenizer.json
        ├── build-info.json
        └── SHA256SUMS
```

- `<version>` is `package.version` from `Cargo.toml`.
- `gguf_bundle/` follows [bundle-format.md](./bundle-format.md).
- The tarball contains **no compiled binary and no static llama.cpp archives**. To build
  from it, you still need aarch64 Linux or x86-64-v3 Linux and `STATIC_LLAMA_DIR`; see
  [development.md](./development.md).
- `linux-arm64` in the name dates from when aarch64 Linux was the only supported build
  platform. The tarball itself holds source code and model files, which build on either
  supported platform.
- Check the bundle after downloading:

  ```bash
  tar -xzf ltembed-*.tar.gz
  cd ltembed-*/gguf_bundle && sha256sum -c SHA256SUMS
  ```

  The tarball itself has no published checksum.

The matrix also sets `output_dimension: 512` and `l2_normalize: true`. They are only
written to the run summary. They are not stored in the bundle; the caller chooses them
through `EngineConfig`.

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
- `model.gguf` and `tokenizer.json` are downloaded from Hugging Face `resolve/main`, not
  from the revision CI pins. A release can therefore contain files that CI never tested.
  Compare `SHA256SUMS` with the reference hashes in
  [bundle-format.md](./bundle-format.md#modelgguf).

### Cutting a release

The repository has no release script. Based on the workflow, the steps are:

1. Set `version` in `Cargo.toml` and merge that change to `main` with CI green.
2. Push a tag `v<version>` that points at that commit:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

3. When the `release-bundles` run finishes, check that the release has the tarball
   attached.

To try the packaging without publishing anything, run the workflow by hand
(`workflow_dispatch`) and download the workflow artifact.

## Why Q5_K_M

The release and CI bundles use `v5-nano-retrieval-Q5_K_M.gguf`. The llama.cpp migration
spike (July 2026, see
[history/migrations/llama-cpp-spike-results.md](./history/migrations/llama-cpp-spike-results.md))
chose it as the smallest quant that met both targets:

- **Parity**: cosine ≥ 0.99 against the PyTorch FP32 golden fixture. Q5_K_M scored a mean of
  0.99720 and a minimum of 0.99663. Q4_K_M dropped to 0.98937.
- **Size**: within the AWS Lambda unzipped package limit of 250 MB. Q5_K_M (169 MB) with the
  17 MB tokenizer and a 7 MB stripped binary came to about 193 MB. Q8_0 (233 MB) came to
  about 257 MB.

The full parity table is in [benchmarking.md](./benchmarking.md#recorded-results). The
`benchmark-arm64` workflow re-checks the same two targets for any set of quants; see
[benchmarking.md](./benchmarking.md#how-the-report-recommends-a-quant).

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
