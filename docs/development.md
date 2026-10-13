# Development

How to set up a build, run the same checks as CI, and work from a host that cannot run the
static llama.cpp archives natively. For what the tests cover see [testing.md](./testing.md);
for the design see [architecture.md](./architecture.md).

## Requirements

| Need | Why | Source of truth |
|---|---|---|
| Rust `1.94.0` with `clippy` and `rustfmt` | Pinned so local `rustfmt`/`clippy` match CI | `rust-toolchain.toml` |
| Linux on aarch64 (Graviton2 / Neoverse N1 or newer) or on an x86-64-v3 CPU; elsewhere a `linux/arm64` container | The static llama.cpp archives are Linux objects with a fixed CPU baseline per variant | `build.rs`, `.github/scripts/fetch-static-llama.sh` |
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
check in the download steps below, and the copy it builds against is checked by the
`pre-push` hook (see [Git hooks](#git-hooks)).

The release is pinned by these variables in `.github/workflows/ci.yml`.
`.github/workflows/benchmark-arm64.yml` runs only on ARM64 and sets the same values except
`STATIC_LLAMA_SHA256_X86_64`:

| Variable | Value |
|---|---|
| `STATIC_LLAMA_REPO` | `Lychee-Technology/static-llama-cpp-rs-builder` |
| `STATIC_LLAMA_TAG` | `v0.1.159-2` |
| `STATIC_LLAMA_LLAMA_CPP_COMMIT` | `cbe28bb3cb0758f0c566a9a3561766821aeb0748` (llama.cpp `b11577`, version `0.6.0-dev`) |
| `STATIC_LLAMA_SHA256_AARCH64` | `720b395930962c9c7a17e1f2828b41fc8fc188db075e44b8583fb402bcdee5a3` |
| `STATIC_LLAMA_SHA256_X86_64` | `12c16eba8a756025d8ae2721544cef7b13decaf1b04e04ef1371dfeead50113d` |

The release uses artifact contract `5`, has no OpenMP dependency, and ships one variant per
architecture. Both variants are built from the same llama.cpp commit and ship the same
`bindings.rs`:

| Variant (`target_profile`) | Asset | `target_triple` | Compiler flags | CPU requirement |
|---|---|---|---|---|
| `aarch64-graviton2` | `static-llama-cpp-v0.1.159-2-aarch64-graviton2.tar.gz` | `aarch64-unknown-linux-gnu` | `-march=armv8.2-a+fp16+dotprod+rcpc -mtune=neoverse-n1` | Graviton2 / Neoverse N1 or newer |
| `x86_64-v3` | `static-llama-cpp-v0.1.159-2-x86_64-v3-linux-gnu.tar.gz` | `x86_64-unknown-linux-gnu` | `-march=x86-64-v3` | x86-64-v3: AVX2, BMI1/2, F16C, FMA, LZCNT, MOVBE |

There is no runtime ISA dispatch: a CPU below the variant's baseline faults with `SIGILL`.
x86-64 CPUs below v3 are not supported. To check an x86_64 host, run
`/lib64/ld-linux-x86-64.so.2 --help`; it must list `x86-64-v3 (supported, searched)`.

To download and verify the release into `.llama-artifacts/`, follow the same steps as
`.github/scripts/fetch-static-llama.sh`. Pick the variant for the machine that will run
cargo: `aarch64-graviton2` on aarch64 Linux and inside the `linux/arm64` container,
`x86_64-v3` on an x86-64-v3 Linux host. The `pre-push` hook builds against
`STATIC_LLAMA_DIR` when it runs natively; its container path needs the `aarch64-graviton2`
variant in `.llama-artifacts/extracted/`.

```bash
STATIC_LLAMA_REPO=Lychee-Technology/static-llama-cpp-rs-builder
STATIC_LLAMA_TAG=v0.1.159-2
STATIC_LLAMA_LLAMA_CPP_COMMIT=cbe28bb3cb0758f0c566a9a3561766821aeb0748

# aarch64 Linux, or the linux/arm64 container (and the pre-push hook):
VARIANT=aarch64-graviton2 PROFILE=aarch64-graviton2 TRIPLE=aarch64-unknown-linux-gnu
SHA256=720b395930962c9c7a17e1f2828b41fc8fc188db075e44b8583fb402bcdee5a3
DEST=.llama-artifacts/extracted
# x86-64-v3 Linux instead:
# VARIANT=x86_64-v3-linux-gnu PROFILE=x86_64-v3 TRIPLE=x86_64-unknown-linux-gnu
# SHA256=12c16eba8a756025d8ae2721544cef7b13decaf1b04e04ef1371dfeead50113d
# DEST=.llama-artifacts/extracted-x86_64-v3

TARBALL="static-llama-cpp-${STATIC_LLAMA_TAG}-${VARIANT}.tar.gz"
mkdir -p .llama-artifacts/dl "$DEST"
gh release download "$STATIC_LLAMA_TAG" --repo "$STATIC_LLAMA_REPO" \
  --pattern "$TARBALL" --pattern "$TARBALL.sha256" --dir .llama-artifacts/dl
(
  cd .llama-artifacts/dl
  echo "${SHA256}  ${TARBALL}" | sha256sum -c -   # repo-pinned hash
  sha256sum -c "${TARBALL}.sha256"                # release sidecar
)
tar -xzf ".llama-artifacts/dl/$TARBALL" -C "$DEST"
(cd "$DEST" && sha256sum -c SHA256SUMS)
# Each line must print the expected value: 5, the profile, the triple, the commit.
jq -r '.artifact_contract_version, .target_profile, .target_triple, .llama_cpp.commit' \
  "$DEST/build-info.json"
echo "expected: 5 $PROFILE $TRIPLE $STATIC_LLAMA_LLAMA_CPP_COMMIT"

export STATIC_LLAMA_DIR="$PWD/$DEST"
```

`.llama-artifacts/` is gitignored. Extract each variant into its own empty directory, as
above; do not extract one over the other. The repo-pinned hash matters because the
release's own `.sha256` sidecar and `SHA256SUMS` live on the same mutable tag and prove
only that the download is complete and uncorrupted. The pinned values were checked against
both before they were committed.

The rest of the pin is not in the workflows. `fetch-static-llama.sh` maps `uname -m` to the
asset name, `target_profile` and `target_triple`, and on x86_64 refuses to continue unless
the CPU passes the x86-64-v3 check in `.github/scripts/x86-64-v3.sh`, which
`.githooks/pre-push` also uses. The script and the hook reject any artifact contract other
than `5`, and any `target_triple` other than that of the variant they expect. When bumping
the release, update the tag, the llama.cpp commit and both SHA-256 pins in `ci.yml`, and
the same variables in `benchmark-arm64.yml`. Update the script and the hook if the new
release changes an asset name, a variant or the contract version.

## Native builds

On aarch64 Linux, or on x86_64 Linux with an x86-64-v3 CPU, point `STATIC_LLAMA_DIR` at the
variant for that host, in the directory the download steps above extracted it to:

```bash
# aarch64 Linux:
export STATIC_LLAMA_DIR=/abs/path/to/.llama-artifacts/extracted
# x86-64-v3 Linux instead:
# export STATIC_LLAMA_DIR=/abs/path/to/.llama-artifacts/extracted-x86_64-v3
cargo build
cargo test --lib
```

`build.rs` does not check the variant. With the other architecture's archives, the build
fails at link time (`rust-lld: error: … is incompatible with elf64-x86-64` on x86_64).

## Container builds

On macOS, or on an x86_64 host whose CPU is below x86-64-v3, run cargo inside a
`linux/arm64` `rust:1.94.0` container with the `aarch64-graviton2` variant. The `pre-push`
hook uses this container whenever it cannot run clippy natively: on those hosts, and on any
host where `STATIC_LLAMA_DIR` is unset or empty (see [Git hooks](#git-hooks)). From the
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
  llama.cpp reference test embeds about 13,500 tokens of long input and takes far longer
  than natively. The 8192-token input needs about 10 GB of memory in the container too.
- With Podman on an SELinux host, add `--security-opt label=disable` so the container can
  read the bind mounts.

## Git hooks

```bash
./scripts/install-git-hooks.sh   # sets core.hooksPath=.githooks for this clone
```

- `pre-commit` runs `cargo fmt --all --check` when staged files include `*.rs`,
  `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml` or `rustfmt.toml`.
- `pre-push` runs `cargo clippy --all-targets -- -D warnings`:
  - natively, against `STATIC_LLAMA_DIR`, if that variable is set and the host is Linux
    aarch64 or Linux x86_64 whose CPU passes the same x86-64-v3 check as CI
    (`.github/scripts/x86-64-v3.sh`). Neither Docker nor `.llama-artifacts/extracted/` is
    needed. `STATIC_LLAMA_DIR` must hold the host's variant, with `target_triple`
    `aarch64-unknown-linux-gnu` or `x86_64-unknown-linux-gnu`; a wrong variant fails the
    push instead of falling back to the container;
  - otherwise in the container above, against `.llama-artifacts/extracted/`, which must
    hold the `aarch64-graviton2` variant (`target_triple` `aarch64-unknown-linux-gnu`): on
    macOS, on an x86_64 CPU below v3 (the hook prints which part of the check failed), and
    wherever `STATIC_LLAMA_DIR` is unset or empty. To use the container on a host that
    would run natively, push with `STATIC_LLAMA_DIR= git push`.

  Either way, the hook first fails fast if that release directory lacks any of
  `lib/libllama.a`, `lib/libggml.a`, `lib/libggml-cpu.a`, `lib/libggml-base.a`,
  `bindings.rs` or `build-info.json`, if its contract version is not `5`, or if its
  `target_triple` is not the one above; in the container case, also if `docker info` fails.
  It does not verify SHA-256 sums; it trusts that the download step above did.

  CI's `Lint` job runs clippy on aarch64, and a native run on x86_64 lints the x86_64
  target instead. The crate has no `cfg(target_arch)` code, but a lint can still differ
  between the targets where a type does, such as `c_char` (`i8` on x86_64, `u8` on
  aarch64).

## Checks before pushing

The commands of the CI `Lint`, `Test` and `Python script tests` jobs, in that order (details
in [testing.md](./testing.md#what-ci-runs)). `check_doc_references.py` checks Markdown links
and `AGENTS.md` paths and needs only Git and Python 3
([testing.md](./testing.md#documentation-references)).

```bash
python3 scripts/check_doc_references.py
cargo fmt --check
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo test --lib
cargo test --test tokenizer_reload_tests
cargo test --test benchmarking_support_tests --bin benchmark_ltembed
cargo check --example api_usage
LTEMBED_REQUIRE_TEST_BUNDLE=1 LTEMBED_TEST_BUNDLE_DIR="$PWD/gguf_bundle" \
  cargo test --lib tokenizer_parity -- --ignored
LTEMBED_REQUIRE_TEST_BUNDLE=1 LTEMBED_TEST_BUNDLE_DIR="$PWD/gguf_bundle" \
  cargo test --test integration_tests
python3 -m pytest tests/ -q
```

The tokenizer parity test and the model-backed integration tests need a bundle at
`./gguf_bundle`; build it with
`python3 scripts/assemble_gguf_bundle.py gguf_bundle`
([bundle-format.md](./bundle-format.md#assembling-a-bundle)), which downloads the same pinned
files CI uses. The integration tests also need about 10 GB of free memory for the
8192-token input.

## End-to-end sanity check

With a bundle at `./gguf_bundle` (see
[bundle-format.md](./bundle-format.md#assembling-a-bundle)):

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
| `.github/scripts/x86-64-v3.sh` | The x86-64-v3 CPU check, sourced by `fetch-static-llama.sh` and `pre-push` |
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
