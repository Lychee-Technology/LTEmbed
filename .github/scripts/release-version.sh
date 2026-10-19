#!/usr/bin/env bash
# Print the version a release is made for, package.version in Cargo.toml, and fail if the
# release's tag is not v<version>. release-bundles.yml runs it before anything is downloaded
# or published, and names the tarball with the version it prints.
#
# Usage: release-version.sh [CARGO_TOML]   (default: Cargo.toml in the working directory)
#
# RELEASE_TAG is the tag of the push or published release; empty or unset (workflow_dispatch)
# skips the check. The error goes to stderr, because stdout is the version the caller reads.
#
# Requires: python3 >= 3.11 (tomllib).
set -euo pipefail

cargo_toml="${1:-Cargo.toml}"
version="$(python3 - "$cargo_toml" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as f:
    print(tomllib.load(f)["package"]["version"])
PY
)"

if [ -n "${RELEASE_TAG:-}" ] && [ "${RELEASE_TAG}" != "v${version}" ]; then
  echo "::error::release tag ${RELEASE_TAG} does not match Cargo.toml version ${version}; tag the release v${version}, or set the version to the tag's" >&2
  exit 1
fi
echo "${version}"
