# Development

How to set up a build, run the same checks as CI, and work from a host that cannot run the
static llama.cpp archives natively. For what the tests cover see [testing.md](./testing.md);
for the design see [architecture.md](./architecture.md).

## Requirements

| Need | Why | Source of truth |
|---|---|---|
| Rust `1.94.0` with `clippy` and `rustfmt` | Pinned so local `rustfmt`/`clippy` match CI | `rust-toolchain.toml` |
| Linux on aarch64 (Graviton2 / Neoverse N1 or newer) or on an x86-64-v3 CPU; on another ARM64 host, a `linux/arm64` container | The static llama.cpp archives are Linux objects with a fixed CPU baseline per variant | `build.rs`, `.github/scripts/fetch-static-llama.sh` |
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
| `STATIC_LLAMA_TAG` | `v0.1.160-1` |
| `STATIC_LLAMA_LLAMA_CPP_COMMIT` | `620d2225cc436ab0b001e3200e590e3b286ef9ab` (llama.cpp `v0.7.0`, version `0.7.0-dev`) |
| `STATIC_LLAMA_SHA256_AARCH64` | `b39f2150292ee6f2aca472fa43448f581ce488725327f9b1f64c409082be106e` |
| `STATIC_LLAMA_SHA256_X86_64` | `835ab81e49b76c16c61581c4230e5663a8562b42472d58052ba3470363f4f316` |

The release uses artifact contract `7`, has no OpenMP dependency, and ships one variant per
architecture. Both variants are built from the same llama.cpp commit and ship the same
`bindings.rs`:

| Variant (`target_profile`) | Asset | `target_triple` | Compiler flags | CPU requirement |
|---|---|---|---|---|
| `aarch64-graviton2` | `static-llama-cpp-v0.1.160-1-aarch64-graviton2.tar.gz` | `aarch64-unknown-linux-gnu` | `-march=armv8.2-a+fp16+dotprod+rcpc -mtune=neoverse-n1` | Graviton2 / Neoverse N1 or newer |
| `x86_64-v3` | `static-llama-cpp-v0.1.160-1-x86_64-v3-linux-gnu.tar.gz` | `x86_64-unknown-linux-gnu` | `-march=x86-64-v3` | x86-64-v3: AVX2, BMI1/2, F16C, FMA, LZCNT, MOVBE |

There is no runtime ISA dispatch: a CPU below the variant's baseline faults with `SIGILL`.
x86-64 CPUs below v3 are not supported. To check an x86_64 host, run
`/lib64/ld-linux-x86-64.so.2 --help`; it must list `x86-64-v3 (supported, searched)`.

To download and verify the release into `.llama-artifacts/`, follow the same steps as
`.github/scripts/fetch-static-llama.sh`. Pick the variant for the machine that will run
cargo: `aarch64-graviton2` on aarch64 Linux and inside the `linux/arm64` container,
`x86_64-v3` on an x86-64-v3 Linux host. The `pre-push` hook builds against
`STATIC_LLAMA_DIR` when it runs natively; its container path, on ARM64 hosts only, needs the
`aarch64-graviton2` variant in `.llama-artifacts/extracted/`.

```bash
STATIC_LLAMA_REPO=Lychee-Technology/static-llama-cpp-rs-builder
STATIC_LLAMA_TAG=v0.1.160-1
STATIC_LLAMA_LLAMA_CPP_COMMIT=620d2225cc436ab0b001e3200e590e3b286ef9ab

# aarch64 Linux, or the linux/arm64 container (and the pre-push hook):
VARIANT=aarch64-graviton2 PROFILE=aarch64-graviton2 TRIPLE=aarch64-unknown-linux-gnu
SHA256=b39f2150292ee6f2aca472fa43448f581ce488725327f9b1f64c409082be106e
DEST=.llama-artifacts/extracted
# x86-64-v3 Linux instead:
# VARIANT=x86_64-v3-linux-gnu PROFILE=x86_64-v3 TRIPLE=x86_64-unknown-linux-gnu
# SHA256=835ab81e49b76c16c61581c4230e5663a8562b42472d58052ba3470363f4f316
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
# Each line must print the expected value: 7, the profile, the triple, the commit.
jq -r '.artifact_contract_version, .target_profile, .target_triple, .llama_cpp.commit' \
  "$DEST/build-info.json"
echo "expected: 7 $PROFILE $TRIPLE $STATIC_LLAMA_LLAMA_CPP_COMMIT"

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
than `7`, and any `target_triple` other than that of the variant they expect. When bumping
the release, update the tag, the llama.cpp commit and both SHA-256 pins in `ci.yml`, and
the same variables in `benchmark-arm64.yml`; `tests/test_reference_generators.py` fails
while the two differ. Update the script and the hook if the new release changes an asset
name, a variant or the contract version; `tests/test_pre_push_hook.py` fails if the hook's
`expected_contract` differs from the script's `EXPECTED_CONTRACT`.
Update [licenses/llama.cpp/NOTICE.md](../licenses/llama.cpp/NOTICE.md) as well, and compare its
license copies with the new release's `LICENSES/`; `tests/test_reference_generators.py` fails
while the notice names another release or llama.cpp commit than `ci.yml`, or while a copy
differs from the sha256 the notice records. The same test fails until `LLAMA_CPP_COMMIT`,
`STATIC_RELEASE` and `VARIANTS` in `scripts/generate_llama_cpp_reference.py` match `ci.yml`
and both variants' sections of `tests/fixtures/llama_cpp_reference.json` name the new release:
rebuild the reference tool and regenerate both sections
([testing.md](./testing.md#llamacpp-reference-testsfixturesllama_cpp_referencejson)), then
run the tokenizer parity and integration tests on both variants. The engine tokenizes
without the backend's lock, and a data race can pass the concurrency test, so also check
the new llama.cpp commit against the argument at `tokenize_with_vocab` in
`src/engine/llama/mod.rs` that `llama_tokenize` only reads shared state
([architecture.md](./architecture.md#concurrency)). Stored
[embedding spaces](./architecture.md#embedding-space) do not name the release, so also
decide whether the new one needs a bump of `STRING_FORM_VERSION` in `src/engine/space.rs`.
It does if it changes how this model tokenizes or pools, which gives the same `model.gguf`
and settings other vectors (`src/engine/AGENTS.md` § "Embedding Space"). Compare the
regenerated sections of `tests/fixtures/llama_cpp_reference.json` with the old ones: if
every token id and vector is unchanged, as from `v0.1.159-3` to `v0.1.160-1`, no bump is
needed. Otherwise find the cause in the llama.cpp diff (tokenization is in its
`src/llama-vocab.cpp`, pooling in `src/llama-graph.cpp`) and bump if it is either. A change
in the kernels alone needs no bump while it moves each variant's vectors no further from
the old ones than the two variants are apart (`CPU_VARIANT_MIN_COSINE` and
`CPU_VARIANT_MAX_NORM_ERROR` in `tests/integration_tests.rs`, which
`test_llama_cpp_reference_variants_agree_within_one_space` checks between the variants).
Beyond that, bump as well.

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

On an ARM64 host that cannot run the archives itself, such as macOS on Apple silicon, run
cargo inside a `linux/arm64` `rust:1.94.0` container with the `aarch64-graviton2` variant.
The `pre-push` hook uses this container on ARM64 hosts where it cannot run clippy natively
(see [Git hooks](#git-hooks)). Use it only on an ARM64 CPU: on x86_64,
`--platform linux/arm64` runs every build and test under qemu emulation, many times slower
than natively. On x86_64, build against the `x86_64-v3` variant
([Native builds](#native-builds)) and run aarch64 work on ARM64 hardware, such as an ARM64
Linux machine or CI's `ubuntu-24.04-arm` runners. An x86_64 CPU below v3 has no supported
build. From the repository root:

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
  - otherwise, on an ARM64 host, in the container above, against
    `.llama-artifacts/extracted/`, which must hold the `aarch64-graviton2` variant
    (`target_triple` `aarch64-unknown-linux-gnu`): on macOS on Apple silicon, and on Linux
    aarch64 where `STATIC_LLAMA_DIR` is unset or empty. To use the container on Linux
    aarch64, push with `STATIC_LLAMA_DIR= git push`.

  On any other host the container would run under emulation, so the hook fails and says
  why clippy cannot run natively: on x86_64 Linux where `STATIC_LLAMA_DIR` is unset or
  empty (set it to the `x86_64-v3` release), on an x86_64 CPU below v3 (the hook prints
  which part of the check failed), and on an Intel Mac or any other CPU. Of these, only the
  first can lint locally; from the others, `git push --no-verify` skips the hook, and CI's
  `Lint` job still runs clippy.

  Before clippy runs, the hook fails fast if that release directory lacks any of
  `lib/libllama.a`, `lib/libggml.a`, `lib/libggml-cpu.a`, `lib/libggml-base.a`,
  `bindings.rs` or `build-info.json`, if its contract version is not `7`, or if its
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
cargo test --test benchmarking_support_tests --bin benchmark_ltembed
cargo check --example api_usage
LTEMBED_REQUIRE_TEST_BUNDLE=1 LTEMBED_TEST_BUNDLE_DIR="$PWD/gguf_bundle" \
  LTEMBED_TEST_REFERENCE_TOKENIZER="$PWD/reference-tokenizer.json" \
  cargo test --lib tokenizer_parity -- --ignored
LTEMBED_REQUIRE_TEST_BUNDLE=1 LTEMBED_TEST_BUNDLE_DIR="$PWD/gguf_bundle" \
  cargo test --test integration_tests
python3 -m pytest tests/ -q
```

The tokenizer parity tests and the model-backed integration tests need a bundle at
`./gguf_bundle`, and the parity tests also the model's `tokenizer.json` at
`./reference-tokenizer.json`, outside the bundle. Build both with
`python3 scripts/assemble_gguf_bundle.py gguf_bundle --reference-tokenizer reference-tokenizer.json`
([bundle-format.md](./bundle-format.md#assembling-a-bundle)), which downloads the same pinned
files CI uses. The integration tests, which embed two 8192-token inputs while other tests
run their own engines, peak at about 3.4 GiB of memory on x86-64-v3.

## End-to-end sanity check

With a bundle at `./gguf_bundle` (see
[bundle-format.md](./bundle-format.md#assembling-a-bundle)):

```bash
cargo run --example api_usage   # searches ./gguf_bundle in the current dir and its ancestors
cargo run --release --bin benchmark_ltembed -- --mode retrieval --bundle-dir gguf_bundle \
  --retrieval-eval-path scripts/retrieval_eval_cases.json \
  --output-dimension 512 --l2-normalize true
```

The example prints the engine's embedding space, the embedding count, the dimension (`512`)
and the first five values.

## Repository layout

| Path | Contents |
|---|---|
| `src/engine/` | `EmbeddingEngine`, bundle parsing, config, inputs, postprocessing |
| `src/engine/llama/` | `LlamaBackend` and the FFI include |
| `src/error.rs` | `LTEmbedError`, `ModelLoadError`, `InferenceError` |
| `src/benchmarking.rs`, `src/bin/benchmark_ltembed.rs` | Benchmark scenarios and CLI ([benchmarking.md](./benchmarking.md)) |
| `examples/api_usage.rs` | Minimal API example |
| `tests/` | Rust integration tests, Python script tests, fixtures |
| `scripts/` | Python tooling ([scripts/README.md](../scripts/README.md)) |
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
