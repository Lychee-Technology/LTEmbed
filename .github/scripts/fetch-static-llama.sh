#!/usr/bin/env bash
# Fetch, SHA-verify, and extract the prebuilt static llama.cpp archives that the crate
# links against, then export STATIC_LLAMA_DIR for subsequent workflow steps.
#
# Requires: gh (with GH_TOKEN), STATIC_LLAMA_REPO, STATIC_LLAMA_TAG, GITHUB_WORKSPACE,
# GITHUB_ENV. Runs on the aarch64 Linux CI runners.
set -euo pipefail

dest="${GITHUB_WORKSPACE}/artifacts/llama"
rm -rf "$dest"
mkdir -p "$dest"

gh release download "$STATIC_LLAMA_TAG" \
  --repo "$STATIC_LLAMA_REPO" \
  --pattern 'static-llama-cpp-*-aarch64-graviton2.tar.gz' \
  --pattern 'static-llama-cpp-*-aarch64-graviton2.tar.gz.sha256' \
  --dir "$dest"

cd "$dest"
sha256sum -c ./*.tar.gz.sha256

mkdir -p extracted
tar -xzf ./*.tar.gz -C extracted
(cd extracted && sha256sum -c SHA256SUMS)

# Sanity-check the contract version the crate was written against.
contract="$(grep -o '"artifact_contract_version": *"[0-9]*"' extracted/build-info.json | grep -o '[0-9]*')"
if [ "$contract" != "2" ]; then
  echo "::error::unexpected static-llama artifact_contract_version '$contract' (expected 2)"
  exit 1
fi

echo "STATIC_LLAMA_DIR=${dest}/extracted" >>"$GITHUB_ENV"
echo "Fetched static llama.cpp artifacts (contract v${contract}) → ${dest}/extracted"
