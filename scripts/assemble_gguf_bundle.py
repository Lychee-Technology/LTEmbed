#!/usr/bin/env python3
"""
Assemble an EmbeddingGemma 2 GGUF bundle from pinned Hugging Face revisions.

This file is the single home of the bundle's source pins: the GGUF repository, revision,
file name and sha256 of every quant LTEmbed accepts. It also pins the repository, revision
and sha256 of the model's tokenizer.json, which the bundle does not contain: the engine
tokenizes with the GGUF's own vocabulary, and the tokenizer tests compare that vocabulary
with this tokenizer.json (src/engine/tokenizer_parity.rs). CI (`.github/workflows/ci.yml`),
the release workflow (`release-bundles.yml`) and the ARM64 benchmark workflow
(`benchmark-arm64.yml`) all build their bundles with it, so the files a release ships are the
files CI tested.

Every download names a commit, never a branch, and is checked against its pinned sha256
before anything else uses it. The bundle directory ends up holding exactly:

    model.gguf       the quant's GGUF (text model only; no mmproj projector)
    build-info.json  the metadata LTEmbed's loader checks (docs/bundle-format.md)
    SHA256SUMS       sha256 of the two files above
    provenance.json  where each file came from, and the LTEmbed commit that assembled it

`--reference-tokenizer PATH` also writes the pinned tokenizer.json to PATH, outside the
bundle, for the tokenizer tests.

Standard library only, so it runs on a bare CI runner.

Usage:
    python3 scripts/assemble_gguf_bundle.py DEST                    # default quant
    python3 scripts/assemble_gguf_bundle.py DEST --quant Q8_0
    python3 scripts/assemble_gguf_bundle.py DEST --gguf PATH        # local copy of the GGUF
    python3 scripts/assemble_gguf_bundle.py DEST --reference-tokenizer REF.json \
        [--tokenizer PATH]                                          # plus the test reference

DEST must not exist or must be empty. Local copies are checked against the same pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


GGUF_REPO = "unsloth/embeddinggemma-2-GGUF"
GGUF_REVISION = "031f0d4b35536f69ab3509d4893c923264fcf253"
# Text-model GGUFs at GGUF_REVISION, with the sha256 Hugging Face records for each LFS
# object. F16 is left out on purpose: the model card does not support float16, and nothing
# here has measured it. The repository's mmproj-*.gguf files are the vision/audio
# projectors; a text bundle never contains one.
GGUF_FILES = {
    "UD-Q4_K_XL": ("embeddinggemma-2-UD-Q4_K_XL.gguf",
                   "ea905fd08e8061db77a0031cbb5096d459d7cfd2bdd0ad2ba66e483f16719493"),
    "UD-Q5_K_XL": ("embeddinggemma-2-UD-Q5_K_XL.gguf",
                   "a9d7a3b7eb421113f05e964e10882676f10fff83811b7876d5bb983db7c7ac41"),
    "UD-Q6_K_XL": ("embeddinggemma-2-UD-Q6_K_XL.gguf",
                   "dc84f042bc3ffbe6122b3b38b2fb299d66f5f37b276c9d458db85d4a4f1e6ba3"),
    "Q8_0": ("embeddinggemma-2-Q8_0.gguf",
             "6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc"),
    "BF16": ("embeddinggemma-2-BF16.gguf",
             "f315cbbb30dd487e44d501c8902abe88808755e43753a96beed1964f0a48aa4f"),
}
DEFAULT_QUANT = "UD-Q5_K_XL"

# Unsloth's safetensors copy of the model: the GGUF repository has no tokenizer.json. Its
# revision is also the golden fixture's model revision (scripts/generate_fixtures.py).
TOKENIZER_REPO = "unsloth/embeddinggemma-2"
TOKENIZER_REVISION = "c8f89cc7ecb4ba5122cd15fdd487422b2b487daf"
TOKENIZER_SHA256 = "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"

UPSTREAM_MODEL = "google/embeddinggemma-2"
# The `license` in the model card front matter of both repositories at the pinned revisions.
MODEL_LICENSE = "apache-2.0"

# The values below must equal the constants in src/engine/mod.rs; the bundle loader rejects
# any other prompt, pooling or width (src/engine/bundle.rs).
TARGET_ID = GGUF_REPO
QUERY_PREFIX = "task: search result | query: "
DOCUMENT_PREFIX = "title: none | text: "
RAW_EMBEDDING_DIMENSION = 768
MAX_LENGTH = 8192

BUNDLE_FILES = ["model.gguf", "build-info.json"]
OUTPUT_FILES = sorted(BUNDLE_FILES + ["SHA256SUMS", "provenance.json"])
CHUNK = 1 << 20


def resolve_url(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def build_info(quant: str) -> dict:
    """The bundle's build-info.json, exactly as docs/bundle-format.md specifies it."""
    if quant not in GGUF_FILES:
        raise ValueError(f"unknown quant {quant!r}; pinned quants: {', '.join(GGUF_FILES)}")
    return {
        "target_id": TARGET_ID,
        "model_metadata": {
            "model_format": "gguf",
            "quant": quant,
            "pooling": "mean",
            "input_kind": "retrieval",
            "query_prefix": QUERY_PREFIX,
            "document_prefix": DOCUMENT_PREFIX,
            "raw_embedding_dimension": RAW_EMBEDDING_DIMENSION,
            "output_embedding_dimension": RAW_EMBEDDING_DIMENSION,
            "max_length": MAX_LENGTH,
        },
    }


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ltembed-assemble-gguf-bundle"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out, CHUNK)


def place(source: Path | None, url: str, destination: Path, expected_sha256: str) -> None:
    """Copy `source`, or download `url`, to `destination`, and require `expected_sha256`.
    A mismatching file is deleted before the error is raised."""
    if source is None:
        print(f"downloading {url}", file=sys.stderr)
        fetch(url, destination)
    else:
        shutil.copyfile(source, destination)
    actual = sha256_of(destination)
    if actual != expected_sha256:
        destination.unlink()
        origin = url if source is None else str(source)
        raise ValueError(
            f"{destination.name} from {origin} has sha256 {actual}, expected the pinned "
            f"{expected_sha256}"
        )


def git_commit() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def workflow_run() -> str | None:
    server, repository, run_id = (
        os.environ.get(name) for name in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID")
    )
    if server and repository and run_id:
        return f"{server}/{repository}/actions/runs/{run_id}"
    return None


def provenance(quant: str, hashes: dict[str, str]) -> dict:
    gguf_file, _ = GGUF_FILES[quant]
    return {
        "schema_version": 1,
        "assembled_by": "scripts/assemble_gguf_bundle.py",
        "ltembed_commit": git_commit(),
        "workflow_run": workflow_run(),
        "model": {
            "upstream": UPSTREAM_MODEL,
            "license": MODEL_LICENSE,
            "quant": quant,
        },
        "files": {
            "model.gguf": {
                "repo": GGUF_REPO,
                "revision": GGUF_REVISION,
                "file": gguf_file,
                "url": resolve_url(GGUF_REPO, GGUF_REVISION, gguf_file),
                "sha256": hashes["model.gguf"],
            },
            "build-info.json": {
                "generated": True,
                "sha256": hashes["build-info.json"],
            },
        },
    }


def assemble(
    destination: Path,
    quant: str = DEFAULT_QUANT,
    gguf: Path | None = None,
) -> dict[str, str]:
    """Build the bundle in `destination` and return the sha256 of each bundle file.

    The files are assembled in a temporary sibling directory and moved into place only after
    every check passes, so a failed run leaves `destination` as it found it.
    """
    gguf_file, gguf_sha256 = GGUF_FILES.get(quant, (None, None))
    if gguf_file is None:
        raise ValueError(f"unknown quant {quant!r}; pinned quants: {', '.join(GGUF_FILES)}")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"{destination} is not empty")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".assemble-") as scratch:
        work = Path(scratch)
        place(gguf, resolve_url(GGUF_REPO, GGUF_REVISION, gguf_file), work / "model.gguf",
              gguf_sha256)
        (work / "build-info.json").write_text(
            json.dumps(build_info(quant), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        hashes = {name: sha256_of(work / name) for name in BUNDLE_FILES}
        (work / "SHA256SUMS").write_text(
            "".join(f"{hashes[name]}  {name}\n" for name in BUNDLE_FILES), encoding="utf-8"
        )
        (work / "provenance.json").write_text(
            json.dumps(provenance(quant, hashes), indent=2) + "\n", encoding="utf-8"
        )
        present = sorted(path.name for path in work.iterdir())
        if present != OUTPUT_FILES:
            raise RuntimeError(f"bundle holds {present}, expected exactly {OUTPUT_FILES}")

        destination.mkdir(exist_ok=True)
        for name in OUTPUT_FILES:
            (work / name).replace(destination / name)
    return hashes


def fetch_reference_tokenizer(destination: Path, source: Path | None = None) -> None:
    """Write the pinned tokenizer.json to `destination`, replacing any file there only once
    the new one matches TOKENIZER_SHA256.

    It is not a bundle file: it is the reference the tokenizer tests compare the GGUF's own
    vocabulary with (src/engine/tokenizer_parity.rs).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".reference-") as scratch:
        part = Path(scratch) / "tokenizer.json"
        place(source, resolve_url(TOKENIZER_REPO, TOKENIZER_REVISION, "tokenizer.json"), part,
              TOKENIZER_SHA256)
        part.replace(destination)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("destination", type=Path, help="bundle directory (absent or empty)")
    parser.add_argument("--quant", default=DEFAULT_QUANT, choices=sorted(GGUF_FILES))
    parser.add_argument("--gguf", type=Path, help="local copy of the quant's GGUF")
    parser.add_argument("--reference-tokenizer", type=Path, metavar="PATH",
                        help="also write the pinned tokenizer.json, the tokenizer tests' "
                             "reference, to PATH (outside the bundle)")
    parser.add_argument("--tokenizer", type=Path,
                        help="local copy of that tokenizer.json (needs --reference-tokenizer)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.tokenizer is not None and args.reference_tokenizer is None:
            raise ValueError("--tokenizer is a local copy of the reference tokenizer.json; "
                             "it needs --reference-tokenizer")
        reference = args.reference_tokenizer
        if reference is not None and reference.resolve().is_relative_to(
                args.destination.resolve()):
            raise ValueError(f"the reference tokenizer {reference} must be outside the bundle "
                             f"{args.destination}: a bundle has no tokenizer.json")
        hashes = assemble(args.destination, args.quant, args.gguf)
        if reference is not None:
            fetch_reference_tokenizer(reference, args.tokenizer)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for name in BUNDLE_FILES:
        print(f"{hashes[name]}  {args.destination / name}")
    if reference is not None:
        print(f"{TOKENIZER_SHA256}  {reference}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
