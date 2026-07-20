#!/usr/bin/env python3
"""write_benchmark_metadata.py — emit a quant matrix job's metadata.json.

Replaces the inline heredoc in ``benchmark-arm64.yml`` so the metadata contract is
unit-testable and numeric fields are real JSON numbers. Each record describes one
quant's benchmark run: the model file and bundle it measured, the static llama.cpp
artifacts it linked, the runner it executed on, and the run parameters.

The ``bundle_size_bytes`` field sums every file in the bundle directory
(model.gguf + tokenizer.json + build-info.json) — the exact contents that would ship
in a Lambda package, which is what the aggregate report's size constraint measures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
DEFAULT_SCENARIOS = "single/zh,single/en,single/medium,single/long,batch/medium/8"
RUNNER_LABELS_ENV = "BENCHMARK_RUNNER_LABELS"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_size_bytes(bundle_dir: Path) -> int:
    return sum(entry.stat().st_size for entry in sorted(bundle_dir.iterdir()) if entry.is_file())


def static_llama_contract_version(static_llama_dir: Path) -> int:
    build_info = json.loads((static_llama_dir / "build-info.json").read_text(encoding="utf-8"))
    return int(build_info["artifact_contract_version"])


def cpu_info(cpuinfo_path: Path = Path("/proc/cpuinfo")) -> tuple[str, list[str]]:
    """(cpu_model, cpu_flags) from /proc/cpuinfo; aarch64 uses ``Features``, x86 ``flags``."""
    model = platform.processor() or "unknown"
    flags: list[str] = []
    if not cpuinfo_path.exists():
        return model, flags
    for line in cpuinfo_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key in {"model name", "Processor", "Hardware"} and value and model in ("", "unknown"):
            model = value
        if key in {"Features", "flags"} and value and not flags:
            flags = value.split()
    return model, flags


def git_sha() -> str:
    return (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )


def build_metadata(args: argparse.Namespace) -> dict[str, object]:
    model_path = args.bundle_dir / "model.gguf"
    cpu_model, cpu_flags = cpu_info(args.cpuinfo_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "llama.cpp",
        "quant": args.quant,
        "model_id": args.model_id,
        "model_file": args.model_file,
        "model_sha256": sha256_file(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "bundle_size_bytes": bundle_size_bytes(args.bundle_dir),
        "static_llama_tag": args.static_llama_tag,
        "static_llama_sha256": args.static_llama_sha256,
        "static_llama_contract_version": static_llama_contract_version(args.static_llama_dir),
        "runner_labels": args.runner_labels,
        "cpu_model": cpu_model,
        "cpu_flags": cpu_flags,
        "threads": args.threads,
        "warmup_iters": args.warmup_iters,
        "timed_iters": args.timed_iters,
        "cold_iters": args.cold_iters,
        "output_dimension": args.output_dimension,
        "l2_normalize": args.l2_normalize,
        "scenarios": [name for name in args.scenarios.split(",") if name],
        "git_sha": args.git_sha or git_sha(),
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def non_negative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-file", required=True, help="GGUF release asset filename.")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--static-llama-dir", type=Path, required=True)
    parser.add_argument("--static-llama-tag", required=True)
    parser.add_argument("--static-llama-sha256", required=True)
    parser.add_argument("--runner-labels", default=os.environ.get(RUNNER_LABELS_ENV, ""))
    parser.add_argument("--threads", type=positive_int, required=True)
    parser.add_argument("--warmup-iters", type=non_negative_int, required=True)
    parser.add_argument("--timed-iters", type=positive_int, required=True)
    parser.add_argument("--cold-iters", type=positive_int, required=True)
    parser.add_argument("--output-dimension", type=positive_int, required=True)
    parser.add_argument("--l2-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS, help="Comma-separated scenario names.")
    parser.add_argument("--git-sha", default="", help="Repo commit; defaults to `git rev-parse HEAD`.")
    parser.add_argument("--cpuinfo-path", type=Path, default=Path("/proc/cpuinfo"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = build_metadata(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
