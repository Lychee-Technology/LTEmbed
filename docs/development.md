# Development

How to set up a build, run the same checks as CI, and work from a host that is not aarch64
Linux. For what the tests cover see [testing.md](./testing.md); for the design see
[architecture.md](./architecture.md).

## Requirements

| Need | Why | Source of truth |
|---|---|---|
| Rust `1.94.0` with `clippy` and `rustfmt` | Pinned so local `rustfmt`/`clippy` match CI | `rust-toolchain.toml` |
| `aarch64-unknown-linux-gnu` host, or a `linux/arm64` container | The static llama.cpp archives are aarch64 Linux objects | `build.rs` |
| Extracted static llama.cpp release in `STATIC_LLAMA_DIR` | `build.rs` links it; every `cargo build/test/clippy` needs it | `build.rs`, `.github/workflows/ci.yml` |
| A GGUF bundle | Only to run inference (Tier 2 tests, example, benchmarks) | [bundle-format.md](./bundle-format.md) |
| Python 3.13 with `pytest`, `numpy` | Script tests; CI uses 3.13 | `.github/workflows/ci.yml` |

Only cargo commands that compile the crate need the artifacts: `build`, `check`, `clippy`,
`test`, `run` and `doc` run `build.rs`, which panics without `STATIC_LLAMA_DIR`. Commands
that do not compile it, such as `cargo fmt`, `cargo metadata` and `cargo tree`, work on any
host.

## Static llama.cpp artifacts

`build.rs` requires `STATIC_LLAMA_DIR` to point at an **extracted, SHA-verified** release of
[`static-llama-cpp-rs-builder`](https://github.com/Lychee-Technology/static-llama-cpp-rs-builder).
It panics if `lib/libllama.a` or `bindings.rs` is missing there. It links `lib/libggml.a`,
`lib/libggml-cpu.a` and `lib/libggml-base.a` without checking them first, so a partial
extraction fails at link time instead. The full release is validated by the `SHA256SUMS`
check in the download steps below and, on hosts where it uses the container, by the
`pre-push` hook (see [Git hooks](#git-hooks)).

The release is pinned by three variables, set to the same values in
`.github/workflows/ci.yml` and `.github/workflows/benchmark-arm64.yml`:

| Variable | Value |
|---|---|
| `STATIC_LLAMA_REPO` | `Lychee-Technology/static-llama-cpp-rs-builder` |
| `STATIC_LLAMA_TAG` | `v0.1.151-1` |
| `STATIC_LLAMA_SHA256` | `48f3aa293824086d667d3938b40be37390d31275af20fd20a88c3880bfee2b90` |

The archives target `aarch64-unknown-linux-gnu` with
`-march=armv8.2-a+fp16+dotprod+rcpc -mtune=neoverse-n1` (Graviton2 / Neoverse N1), use
artifact contract `2`, and have no OpenMP dependency. A CPU without those ISA extensions
cannot run the result.

To download and verify the release into `.llama-artifacts/`, the location the `pre-push`
hook expects, follow the same steps as `.github/scripts/fetch-static-llama.sh`:

```bash
STATIC_LLAMA_REPO=Lychee-Technology/static-llama-cpp-rs-builder
STATIC_LLAMA_TAG=v0.1.151-1
STATIC_LLAMA_SHA256=48f3aa293824086d667d3938b40be37390d31275af20fd20a88c3880bfee2b90
TARBALL="static-llama-cpp-${STATIC_LLAMA_TAG}-aarch64-graviton2.tar.gz"

mkdir -p .llama-artifacts/dl .llama-artifacts/extracted
gh release download "$STATIC_LLAMA_TAG" --repo "$STATIC_LLAMA_REPO" \
  --pattern "$TARBALL" --pattern "$TARBALL.sha256" --dir .llama-artifacts/dl
(
  cd .llama-artifacts/dl
  echo "${STATIC_LLAMA_SHA256}  ${TARBALL}" | sha256sum -c -   # repo-pinned hash
  sha256sum -c "${TARBALL}.sha256"                             # release sidecar
)
tar -xzf ".llama-artifacts/dl/$TARBALL" -C .llama-artifacts/extracted
(cd .llama-artifacts/extracted && sha256sum -c SHA256SUMS)
grep '"artifact_contract_version"' .llama-artifacts/extracted/build-info.json   # must be "2"

export STATIC_LLAMA_DIR="$PWD/.llama-artifacts/extracted"
```

`.llama-artifacts/` is gitignored. The repo-pinned hash matters because the release's own
`.sha256` sidecar lives on the same mutable tag and proves only that the download was not
corrupted.

The rest of the pin is not in the workflows. `fetch-static-llama.sh` selects the
`*-aarch64-graviton2.tar.gz` asset, and both it and `.githooks/pre-push` reject any
artifact contract other than `2`. When bumping the release, update all three variables in
**both** workflows, and update the script and the hook if the new release changes the asset
name or the contract version.

## Building on aarch64 Linux

```bash
export STATIC_LLAMA_DIR=/abs/path/to/.llama-artifacts/extracted
cargo build
cargo test --lib
```

## Non-aarch64 hosts

On macOS or x86_64, run cargo inside a `linux/arm64` `rust:1.94.0` container. From the
repository root:

```bash
docker run --rm --platform linux/arm64 \
  -v "$PWD":/work \
  -v "$PWD/.llama-artifacts/extracted":/opt/static-llama \
  -v ltembed-rustup:/usr/local/rustup \
  -v ltembed-cargo-registry:/usr/local/cargo/registry \
  -w /work \
  -e STATIC_LLAMA_DIR=/opt/static-llama \
  -e CARGO_TARGET_DIR=/work/target-linux-arm64 \
  rust:1.94.0 cargo test --lib
```

- The image, mounts and environment are the ones the `pre-push` hook uses. The hook ends
  with `cargo clippy --all-targets -- -D warnings` instead of `cargo test --lib`; any cargo
  command can go in that position.
- The named volumes keep the toolchain and registry between runs, and
  `target-linux-arm64/` (gitignored) keeps container builds apart from the host `target/`.
- To run Tier 2 tests, put a bundle under the repo and pass its container path, for
  example `-e LTEMBED_TEST_BUNDLE_DIR=/work/gguf_bundle`.
- On an x86_64 host, `--platform linux/arm64` runs under qemu emulation, which is slow; the
  token-id parity test alone takes minutes.
- With Podman on an SELinux host, add `--security-opt label=disable` so the container can
  read the bind mounts.

## Git hooks

```bash
./scripts/install-git-hooks.sh   # sets core.hooksPath=.githooks for this clone
```

- `pre-commit` runs `cargo fmt --all --check` when staged files include `*.rs`,
  `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml` or `rustfmt.toml`.
- `pre-push` runs `cargo clippy --all-targets -- -D warnings`:
  - natively, if the host is Linux aarch64 and `STATIC_LLAMA_DIR` is set;
  - otherwise in the container above, using `.llama-artifacts/extracted/`. It fails fast if
    that directory lacks any of `lib/libllama.a`, `lib/libggml.a`, `lib/libggml-cpu.a`,
    `lib/libggml-base.a`, `bindings.rs` or `build-info.json`, if the contract version is not
    `2`, or if `docker info` fails.

  The hook does not verify SHA-256 sums; it trusts that the download step above did.

## Checks before pushing

The CI `Test` and `Lint` jobs, in order (details in [testing.md](./testing.md#what-ci-runs)):

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --lib
cargo test --test tokenizer_reload_tests
cargo test --test benchmarking_support_tests --bin benchmark_ltembed
cargo check --example api_usage
LTEMBED_REQUIRE_TEST_BUNDLE=1 LTEMBED_TEST_BUNDLE_DIR="$PWD/gguf_bundle" \
  cargo test --test integration_tests
python3 -m pytest tests/ -q
```

## End-to-end sanity check

With a bundle at `./gguf_bundle` (see
[bundle-format.md](./bundle-format.md#assembling-a-bundle-by-hand)):

```bash
cargo run --example api_usage   # searches ./gguf_bundle in the current dir and its ancestors
cargo run --release --bin benchmark_ltembed -- --mode retrieval --bundle-dir gguf_bundle \
  --retrieval-eval-path scripts/retrieval_eval_cases.json \
  --output-dimension 512 --l2-normalize true
```

The example prints the embedding count, the dimension (`512`) and the first five values.

## Repository layout

| Path | Contents |
|---|---|
| `src/engine/` | `EmbeddingEngine`, bundle parsing, config, inputs, postprocessing |
| `src/engine/llama/` | `LlamaBackend` and the FFI include |
| `src/traits/tokenizer.rs` | `HFTokenizer` |
| `src/error.rs` | `LTEmbedError`, `ModelLoadError`, `InferenceError` |
| `src/benchmarking.rs`, `src/bin/benchmark_ltembed.rs` | Benchmark scenarios and CLI ([benchmarking.md](./benchmarking.md)) |
| `examples/api_usage.rs` | Minimal API example |
| `tests/` | Rust integration tests, Python script tests, fixtures |
| `scripts/` | Python tooling ([scripts/README.md](../scripts/README.md)) |
| `assets/` | `config.json` and an old BERT `tokenizer.json`, used only by tokenizer unit tests. Not a bundle. |
| `.github/scripts/fetch-static-llama.sh` | CI download and verify of the static release |
| `.githooks/` | `pre-commit`, `pre-push` |

## Branches

| Branch | Contents |
|---|---|
| `main` | llama.cpp/GGUF (this documentation) |
| `ort` | Frozen ONNX Runtime backend (`OnnxEngine`, `ort_bundle`), last updated 2026-07-08 |
| `matrixmultiply` | Legacy pure-Rust BERT / `matrixmultiply` backend and NEON tuning experiments |

Documents written for the `ort` and `matrixmultiply` lines are under
[history/](./README.md#history).

## Coding standard

See [rust-coding-std.md](./rust-coding-std.md).
