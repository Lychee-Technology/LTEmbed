#!/usr/bin/env python3
"""
Generate the quantized reference: upstream llama.cpp's own embedding tool on the release GGUF.

The golden fixture (scripts/generate_fixtures.py) is the F32 PyTorch model. LTEmbed runs a
4-bit GGUF, so it can only approach that reference to within quantization error. This script
records what upstream llama.cpp itself computes for the same GGUF and inputs, so that the
tests can tell the two apart: LTEmbed must reproduce this reference almost exactly (any gap
is an LTEmbed tokenization, prefix, pooling or FFI error), while the gap between this
reference and the F32 golden is the quantization error.

The tool is `llama-embedding` (examples/embedding/embedding.cpp) from llama.cpp at the commit
the static release was built from, linked against that release's own static archives
(libllama.a, libggml*.a). It therefore runs the same tokenizer (the GGUF's vocabulary) and
kernels as LTEmbed but none of LTEmbed's code: it tokenizes and builds its batch itself.
Kernels differ between the x86_64-v3 and aarch64-graviton2 variants, so the fixture holds one
set of vectors per variant. scripts/build_llama_cpp_reference_tool.sh builds the tool;
docs/testing.md § "llama.cpp reference" shows how, for each variant.

Each input runs alone in its own process, as one sequence, with the context settings LTEmbed
uses: mean pooling, non-causal attention, flash attention on, n_ctx = n_batch = n_ubatch =
8192, and no normalization (the raw 768-wide projected vector).

Requirements: Python 3 only (no third-party packages).

Usage:
    python3 scripts/generate_llama_cpp_reference.py --variant x86_64-v3 \\
        --tool /path/to/llama-embedding --gguf /path/to/embeddinggemma-2-UD-Q4_K_XL.gguf

Output:
    tests/fixtures/llama_cpp_reference.json (the variant's section is replaced, others kept)
"""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


GOLDEN_PATH = Path("tests/fixtures/test_fixtures.json")
OUTPUT_PATH = Path("tests/fixtures/llama_cpp_reference.json")
LLAMA_CPP_COMMIT = "cbe28bb3cb0758f0c566a9a3561766821aeb0748"
STATIC_RELEASE = "v0.1.159-2"
GGUF = {
    "repo": "unsloth/embeddinggemma-2-GGUF",
    "revision": "031f0d4b35536f69ab3509d4893c923264fcf253",
    "file": "embeddinggemma-2-UD-Q4_K_XL.gguf",
    "sha256": "ea905fd08e8061db77a0031cbb5096d459d7cfd2bdd0ad2ba66e483f16719493",
}
# SHA-256 of each variant's release archive, as pinned in .github/workflows/ci.yml
# (tests/test_reference_generators.py checks that they agree).
VARIANTS = {
    "x86_64-v3": "12c16eba8a756025d8ae2721544cef7b13decaf1b04e04ef1371dfeead50113d",
    "aarch64-graviton2": "720b395930962c9c7a17e1f2828b41fc8fc188db075e44b8583fb402bcdee5a3",
}
MAX_LENGTH = 8192
# Never occurs in an input, so the tool treats each file as one prompt rather than one per line.
SEPARATOR = "<#LTEMBED-NO-SEPARATOR#>"
TOOL_ARGS = [
    "--no-escape", "--embd-separator", SEPARATOR,
    "--pooling", "mean", "--attention", "non-causal", "--flash-attn", "on",
    "--embd-normalize", "-1", "--embd-output-format", "json", "--verbose-prompt",
    "-c", str(MAX_LENGTH), "-b", str(MAX_LENGTH), "-ub", str(MAX_LENGTH),
]
TOKEN_LINE = re.compile(r"^ *(-?\d+) -> '", re.MULTILINE)
JSON_START = '{\n  "object": "list"'


def parse_tool_output(stdout: str) -> tuple:
    """The token ids that `--verbose-prompt` prints and the one embedding of the JSON output."""
    start = stdout.index(JSON_START)
    tokens = [int(token) for token in TOKEN_LINE.findall(stdout[:start])]
    data = json.loads(stdout[start:])["data"]
    if len(data) != 1:
        raise ValueError(f"expected one embedding, got {len(data)}")
    return tokens, data[0]["embedding"]


def float32_list(values) -> list:
    """The values the tool printed. It prints 7 decimals (an absolute error of at most 5e-8 on
    raw values of order 1 to 100); 9 significant digits keep every digit it printed."""
    return [float(f"{float(value):.9g}") for value in values]


def format_json(value, level: int = 0) -> str:
    """Indented JSON that keeps each list of numbers on one line."""
    indent = "  " * (level + 1)
    if isinstance(value, dict) and value:
        items = [f"{indent}{json.dumps(key)}: {format_json(item, level + 1)}"
                 for key, item in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + "  " * level + "}"
    if isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        items = [indent + format_json(item, level + 1) for item in value]
        return "[\n" + ",\n".join(items) + "\n" + "  " * level + "]"
    return json.dumps(value, ensure_ascii=False)


def run_tool(tool: Path, gguf: Path, text: str, threads: int) -> tuple:
    # `-f` drops one trailing newline, so an input ending in one would be changed.
    if text.endswith("\n"):
        raise ValueError("an input that ends in a newline cannot be passed through -f")
    with tempfile.TemporaryDirectory() as tmp:
        prompt = Path(tmp) / "prompt.txt"
        prompt.write_text(text, encoding="utf-8")
        result = subprocess.run(
            [str(tool), "-m", str(gguf), "-f", str(prompt), "-t", str(threads), *TOOL_ARGS],
            capture_output=True, text=True, encoding="utf-8", check=False,
        )
    if result.returncode != 0:
        raise SystemExit(f"{tool} failed ({result.returncode}):\n{result.stderr[-2000:]}")
    return parse_tool_output(result.stdout)


def tool_commit(tool: Path) -> str:
    version = subprocess.run([str(tool), "--version"], capture_output=True, text=True,
                             check=False)
    match = re.search(r"commit ([0-9a-f]+)", version.stdout + version.stderr)
    if not match or not LLAMA_CPP_COMMIT.startswith(match.group(1)):
        raise SystemExit(f"{tool} is not built from llama.cpp {LLAMA_CPP_COMMIT}: "
                         f"{(version.stdout + version.stderr).strip()}")
    return match.group(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--tool", required=True, type=Path,
                        help="llama-embedding linked against the variant's release archives")
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reset", action="store_true",
                        help="drop the other variants' sections (after the inputs changed)")
    args = parser.parse_args()

    gguf_sha256 = hashlib.sha256(args.gguf.read_bytes()).hexdigest()
    if gguf_sha256 != GGUF["sha256"]:
        raise SystemExit(f"{args.gguf} has sha256 {gguf_sha256}, expected {GGUF['sha256']}")
    commit = tool_commit(args.tool)

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    existing = (json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
                if OUTPUT_PATH.exists() else {})
    known_tokens = {case["name"]: case["token_ids"] for case in existing.get("inputs", [])}

    inputs, embeddings = [], []
    for fixture in golden["fixtures"]:
        text = golden["prompts"][fixture["kind"]] + fixture["text"]
        tokens, embedding = run_tool(args.tool, args.gguf, text, args.threads)
        name = fixture["name"]
        if len(embedding) != golden["raw_dim"]:
            raise SystemExit(f"{name}: {len(embedding)} values, expected {golden['raw_dim']}")
        if len(tokens) != fixture["tokens"]:
            raise SystemExit(f"{name}: the GGUF tokenizes it to {len(tokens)} tokens, the "
                             f"PyTorch reference to {fixture['tokens']}")
        # Tokenization does not depend on the variant: every variant must agree.
        if name in known_tokens and known_tokens[name] != tokens:
            raise SystemExit(f"{name}: token ids differ from the other variants' run")
        inputs.append({"name": name, "token_ids": tokens})
        embeddings.append({"name": name, "embedding": float32_list(embedding)})
        print(f"  OK: {name} ({len(tokens)} tokens)")

    variants = {} if args.reset else existing.get("variants", {})
    names = [entry["name"] for entry in inputs]
    for variant, section in variants.items():
        if variant != args.variant and [e["name"] for e in section["embeddings"]] != names:
            raise SystemExit(f"the {variant} section was generated from other inputs: rerun "
                             f"with --reset, then once more for {variant}")
    variants[args.variant] = {
        "static_release": STATIC_RELEASE,
        "archive_sha256": VARIANTS[args.variant],
        "embeddings": embeddings,
    }
    output = {
        "generator": "scripts/generate_llama_cpp_reference.py",
        "inputs_from": str(GOLDEN_PATH),
        "gguf": GGUF,
        "llama_cpp": {
            "commit": LLAMA_CPP_COMMIT,
            "tool": "llama-embedding (examples/embedding/embedding.cpp)",
            "tool_commit": commit,
            "arguments": TOOL_ARGS,
        },
        "raw_dim": golden["raw_dim"],
        "inputs": inputs,
        "variants": dict(sorted(variants.items())),
    }
    OUTPUT_PATH.write_text(format_json(output) + "\n", encoding="utf-8")
    print(f"\nSaved {len(inputs)} {args.variant} embeddings → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
