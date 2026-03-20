#!/usr/bin/env python3
"""Convert assets/ (e5-small-v2 HuggingFace model) to assets/model.gguf.

Prerequisites:
  - git submodule `vendor/llama.cpp` must be initialized (see issue #60)
  - Python packages: torch, transformers, sentencepiece (pip install ...)

Usage:
  python scripts/convert_to_gguf.py [--outtype {f32,f16,q8_0}]

The output is written to assets/model.gguf.
"""
import argparse
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
ASSETS = REPO_ROOT / "assets"
LLAMA = REPO_ROOT / "vendor" / "llama.cpp"
CONVERTER = LLAMA / "convert_hf_to_gguf.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outtype",
        choices=["f32", "f16", "q8_0"],
        default="f32",
        help="Output tensor type (default: f32 for accuracy parity testing)",
    )
    args = parser.parse_args()

    if not CONVERTER.exists():
        sys.exit(
            f"ERROR: {CONVERTER} not found.\n"
            "Run: git submodule update --init vendor/llama.cpp"
        )

    if not (ASSETS / "config.json").exists():
        sys.exit(f"ERROR: {ASSETS}/config.json not found — model assets missing.")

    out_path = ASSETS / "model.gguf"
    cmd = [
        sys.executable,
        str(CONVERTER),
        str(ASSETS),
        "--outfile", str(out_path),
        "--outtype", args.outtype,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"\nDone. Written to: {out_path}")
    print("You can now use --features ggml-backend builds and the backend comparison bench.")


if __name__ == "__main__":
    main()
