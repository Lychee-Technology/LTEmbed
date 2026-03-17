#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null
chmod +x "$repo_root/.githooks/pre-commit" "$repo_root/.githooks/pre-push"
git -C "$repo_root" config --local core.hooksPath .githooks

echo "Installed repository Git hooks."
echo "hooksPath=$(git -C "$repo_root" config --local core.hooksPath)"
