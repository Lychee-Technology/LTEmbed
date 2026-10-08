#!/usr/bin/env bash
# Fetch, SHA-verify, and extract the prebuilt static llama.cpp archives that the crate
# links against, then export STATIC_LLAMA_DIR for subsequent workflow steps.
#
# The release ships one variant per architecture; the runner's `uname -m` selects it:
#   aarch64 → static-llama-cpp-<tag>-aarch64-graviton2.tar.gz, pinned by STATIC_LLAMA_SHA256_AARCH64
#   x86_64  → static-llama-cpp-<tag>-x86_64-v3-linux-gnu.tar.gz, pinned by STATIC_LLAMA_SHA256_X86_64
# The x86_64 archives execute x86-64-v3 instructions unconditionally (no runtime dispatch),
# so on x86_64 the runner's CPU must pass an x86-64-v3 check first.
#
# Requires: gh (with GH_TOKEN), jq, STATIC_LLAMA_REPO, STATIC_LLAMA_TAG,
# STATIC_LLAMA_LLAMA_CPP_COMMIT, the sha256 pin for the runner's architecture,
# GITHUB_WORKSPACE, GITHUB_ENV. Also exports STATIC_LLAMA_SHA256 (the pin that was checked).
set -euo pipefail

# The artifact contract the crate's FFI (src/engine/llama/) was written and tested against.
readonly EXPECTED_CONTRACT=4

: "${STATIC_LLAMA_REPO:?set STATIC_LLAMA_REPO}"
: "${STATIC_LLAMA_TAG:?set STATIC_LLAMA_TAG}"
: "${STATIC_LLAMA_LLAMA_CPP_COMMIT:?set STATIC_LLAMA_LLAMA_CPP_COMMIT to the llama.cpp commit the pinned release compiles}"
command -v jq >/dev/null || { echo "::error::jq is required"; exit 1; }

# Fail closed unless glibc and the kernel both report every x86-64-v3 feature. glibc lists
# x86-64-v3 as "supported" only if CPUID reports the features and XGETBV shows the OS
# enabled AVX state; /proc/cpuinfo is checked as well, as in the builder's own gate.
require_x86_64_v3() {
  local ldso=/lib64/ld-linux-x86-64.so.2 hwcaps flags missing=() f
  hwcaps="$("$ldso" --help 2>/dev/null || true)"
  if ! grep -Eq '^[[:space:]]*x86-64-v3 \(supported' <<<"$hwcaps"; then
    echo "::error::$ldso does not report x86-64-v3 as supported; the x86_64-v3 static llama.cpp archives cannot run on this CPU"
    exit 1
  fi
  flags=" $(grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2) "
  # x86-64-v2 + v3 feature set. Linux reports SSE3 as `pni` and LZCNT as `abm`.
  for f in cx16 lahf_lm popcnt pni sse4_1 sse4_2 ssse3 avx avx2 bmi1 bmi2 f16c fma abm movbe xsave; do
    [[ "$flags" == *" $f "* ]] || missing+=("$f")
  done
  if [ "${#missing[@]}" -ne 0 ]; then
    echo "::error::/proc/cpuinfo lacks x86-64-v3 features: ${missing[*]}"
    exit 1
  fi
  echo "x86-64-v3 capability: PASS ($(grep -m1 '^model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//'))"
}

arch="$(uname -m)"
case "$arch" in
  aarch64)
    variant=aarch64-graviton2
    profile=aarch64-graviton2
    triple=aarch64-unknown-linux-gnu
    pinned_sha256="${STATIC_LLAMA_SHA256_AARCH64:?set STATIC_LLAMA_SHA256_AARCH64 to the repo-pinned tarball sha256}"
    ;;
  x86_64)
    variant=x86_64-v3-linux-gnu
    profile=x86_64-v3
    triple=x86_64-unknown-linux-gnu
    pinned_sha256="${STATIC_LLAMA_SHA256_X86_64:?set STATIC_LLAMA_SHA256_X86_64 to the repo-pinned tarball sha256}"
    require_x86_64_v3
    ;;
  *)
    echo "::error::no static llama.cpp variant for architecture '$arch' (supported: aarch64, x86_64)"
    exit 1
    ;;
esac
tarball="static-llama-cpp-${STATIC_LLAMA_TAG}-${variant}.tar.gz"

dest="${GITHUB_WORKSPACE}/artifacts/llama"
rm -rf "$dest"
mkdir -p "$dest"

gh release download "$STATIC_LLAMA_TAG" \
  --repo "$STATIC_LLAMA_REPO" \
  --pattern "$tarball" \
  --pattern "$tarball.sha256" \
  --dir "$dest"

cd "$dest"
# Pin the artifact to a repo-controlled expected sha256, not just the release's own sidecar
# checksum (which lives on the same mutable tag and only proves transfer integrity).
echo "${pinned_sha256}  ${tarball}" | sha256sum -c -
sha256sum -c "${tarball}.sha256"

mkdir -p extracted
tar -xzf "$tarball" -C extracted
(cd extracted && sha256sum -c SHA256SUMS)

# Fail closed on a release built against a different artifact contract, for another
# variant, or from a llama.cpp commit other than the one the crate was validated against.
check_field() {
  local field="$1" expected="$2" actual
  actual="$(jq -r "$field" extracted/build-info.json)"
  if [ "$actual" != "$expected" ]; then
    echo "::error::static-llama build-info.json $field is '$actual' (expected '$expected')"
    exit 1
  fi
}
check_field .artifact_contract_version "$EXPECTED_CONTRACT"
check_field .target_profile "$profile"
check_field .architecture "$arch"
check_field .target_triple "$triple"
check_field .llama_cpp.commit "$STATIC_LLAMA_LLAMA_CPP_COMMIT"

{
  echo "STATIC_LLAMA_DIR=${dest}/extracted"
  echo "STATIC_LLAMA_SHA256=${pinned_sha256}"
} >>"$GITHUB_ENV"
echo "Fetched static llama.cpp ${STATIC_LLAMA_TAG} (${profile}, contract v${EXPECTED_CONTRACT}) → ${dest}/extracted"
