#!/usr/bin/env bash
# build.sh — Cross-compile LTEmbed for AWS Lambda (ARM64/Graviton) from macOS
#
# Prerequisites:
#   rustup target add aarch64-unknown-linux-gnu
#   brew tap messense/macos-cross-toolchains
#   brew install aarch64-unknown-linux-gnu
#
# Usage:
#   chmod +x build.sh && ./build.sh

set -euo pipefail

TARGET="aarch64-unknown-linux-gnu"
OUTPUT_DIR="dist"

echo "==> Cross-compiling for $TARGET ..."
export RUSTFLAGS="-C target-cpu=neoverse-n1"
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-unknown-linux-gnu-gcc \
  cargo build --release --target "$TARGET"

echo "==> Packaging Lambda deployment zip ..."
mkdir -p "$OUTPUT_DIR"
cp "target/$TARGET/release/bootstrap" "$OUTPUT_DIR/bootstrap"
cp -r assets "$OUTPUT_DIR/assets"
# Remove large model file from zip if deploying via S3 layer instead
# rm -f "$OUTPUT_DIR/assets/model.safetensors"

(cd "$OUTPUT_DIR" && zip -r "ltembed-lambda.zip" bootstrap assets/)

echo "==> Done: $OUTPUT_DIR/ltembed-lambda.zip"
echo "    Upload to AWS Lambda. Runtime: provided.al2023. Architecture: arm64. Handler: bootstrap."
