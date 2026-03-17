#!/usr/bin/env python3
"""
Orchestrate LTEmbed, Candle, and PyTorch embedding benchmarks.

This script runs a unified benchmark suite and writes a normalized CSV report
covering cold-start latency, warm latency, and correctness comparisons against
PyTorch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "intfloat/e5-small-v2"
DEFAULT_MODEL_SOURCE = "huggingface"
DEFAULT_CORRECTNESS_THRESHOLD = 0.999
RUNNER_LABELS_ENV = "BENCHMARK_RUNNER_LABELS"

SHORT_TEXT = "query: Hello, world!"
MEDIUM_TEXT = (
    "query: What is the impact of large language models on software engineering "
    "productivity?"
)
LONG_TEXT = "passage: " + "The quick brown fox jumps over the lazy dog. " * 30

CSV_FIELDNAMES = [
    "run_id",
    "timestamp_utc",
    "host_os",
    "host_arch",
    "cpu_model",
    "runner_labels",
    "model_id",
    "model_source",
    "implementation",
    "implementation_version",
    "git_sha",
    "scenario",
    "mode",
    "batch_size",
    "text_profile",
    "threads",
    "warmup_iters",
    "timed_iters",
    "mean_ms",
    "median_ms",
    "p95_ms",
    "p99_ms",
    "min_ms",
    "max_ms",
    "cosine_similarity_vs_pytorch",
    "status",
    "notes",
]


class Scenario(dict):
    @property
    def name(self) -> str:
        return self["name"]

    @property
    def batch_size(self) -> int:
        return self["batch_size"]

    @property
    def text_profile(self) -> str:
        return self["text_profile"]

    @property
    def texts(self) -> tuple[str, ...]:
        return self["texts"]


SCENARIOS = [
    Scenario(name="single/short", batch_size=1, text_profile="short", texts=(SHORT_TEXT,)),
    Scenario(name="single/medium", batch_size=1, text_profile="medium", texts=(MEDIUM_TEXT,)),
    Scenario(name="single/long", batch_size=1, text_profile="long", texts=(LONG_TEXT,)),
    Scenario(name="batch/medium/1", batch_size=1, text_profile="medium", texts=(MEDIUM_TEXT,)),
    Scenario(name="batch/medium/4", batch_size=4, text_profile="medium", texts=(MEDIUM_TEXT,) * 4),
    Scenario(name="batch/medium/8", batch_size=8, text_profile="medium", texts=(MEDIUM_TEXT,) * 8),
    Scenario(
        name="batch/mixed/8",
        batch_size=8,
        text_profile="mixed",
        texts=(
            SHORT_TEXT,
            MEDIUM_TEXT,
            LONG_TEXT,
            SHORT_TEXT,
            MEDIUM_TEXT,
            LONG_TEXT,
            SHORT_TEXT,
            MEDIUM_TEXT,
        ),
    ),
    Scenario(
        name="batch/medium/16",
        batch_size=16,
        text_profile="medium",
        texts=(MEDIUM_TEXT,) * 16,
    ),
]
SCENARIO_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}


def extract_cargo_lock_version(lock_text: str, package_name: str) -> str:
    current_name: str | None = None
    for raw_line in lock_text.splitlines():
        line = raw_line.strip()
        if line.startswith("name = "):
            current_name = line.split('"', 2)[1]
        elif current_name == package_name and line.startswith("version = "):
            return line.split('"', 2)[1]
    raise ValueError(f"package {package_name!r} not found in Cargo.lock")


def write_csv_report(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})


def build_correctness_row(
    base_fields: dict[str, str],
    cosine_similarity: float,
    threshold: float,
) -> dict[str, str]:
    row = {field: "" for field in CSV_FIELDNAMES}
    row.update(base_fields)
    row["cosine_similarity_vs_pytorch"] = f"{cosine_similarity:.6f}"
    row["status"] = "pass" if cosine_similarity >= threshold else "fail"
    return row


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


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            if key in {"model name", "Processor", "Hardware"} and value:
                return value
    return platform.processor() or "unknown"


def runner_labels() -> str:
    return os.environ.get(RUNNER_LABELS_ENV, "")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def host_metadata() -> dict[str, str]:
    return {
        "host_os": platform.system().lower(),
        "host_arch": platform.machine().lower(),
        "cpu_model": cpu_model(),
        "runner_labels": runner_labels(),
    }


def cargo_lock_text() -> str:
    return (ROOT / "Cargo.lock").read_text(encoding="utf-8")


def rust_version() -> str:
    return (
        subprocess.run(
            ["rustc", "--version"],
            check=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )


def python_version() -> str:
    return platform.python_version()


def cosine_similarity(lhs: list[float], rhs: list[float]) -> float:
    if len(lhs) != len(rhs):
        raise ValueError("embedding lengths differ")
    dot = sum(a * b for a, b in zip(lhs, rhs))
    lhs_norm = sum(a * a for a in lhs) ** 0.5
    rhs_norm = sum(b * b for b in rhs) ** 0.5
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        raise ValueError("zero vector encountered")
    return dot / (lhs_norm * rhs_norm)


def scenario_from_name(name: str) -> Scenario:
    try:
        return SCENARIO_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown scenario: {name}") from exc


def cargo_run_prefix(features: str | None) -> list[str]:
    command = ["cargo", "run", "--quiet", "--release"]
    if features:
        command.extend(["--features", features])
    return command


def ltembed_warm_command(args: argparse.Namespace) -> list[str]:
    command = cargo_run_prefix(getattr(args, "ltembed_cargo_features", None))
    command.extend([
        "--bin",
        "benchmark_ltembed",
        "--",
        "--mode",
        "warm",
        "--model-dir",
        str(args.model_dir),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--threads",
        str(args.threads),
    ])
    if getattr(args, "scenario", None):
        command.extend(["--scenario", str(args.scenario)])
    return command


def ltembed_cold_command(args: argparse.Namespace, scenario_name: str) -> list[str]:
    return cargo_run_prefix(getattr(args, "ltembed_cargo_features", None)) + [
        "--bin",
        "benchmark_ltembed",
        "--",
        "--mode",
        "cold",
        "--scenario",
        scenario_name,
        "--model-dir",
        str(args.model_dir),
        "--threads",
        str(args.threads),
    ]


def ltembed_correctness_command(args: argparse.Namespace) -> list[str]:
    return cargo_run_prefix(getattr(args, "ltembed_cargo_features", None)) + [
        "--bin",
        "benchmark_ltembed",
        "--",
        "--mode",
        "correctness",
        "--model-dir",
        str(args.model_dir),
        "--threads",
        str(args.threads),
    ]


def candle_warm_command(args: argparse.Namespace) -> list[str]:
    return [
        "cargo",
        "run",
        "--quiet",
        "--release",
        "--example",
        "benchmark_candle",
        "--",
        "--mode",
        "warm",
        "--model-dir",
        str(args.model_dir),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--threads",
        str(args.threads),
    ]


def candle_cold_command(args: argparse.Namespace, scenario_name: str) -> list[str]:
    return [
        "cargo",
        "run",
        "--quiet",
        "--release",
        "--example",
        "benchmark_candle",
        "--",
        "--mode",
        "cold",
        "--scenario",
        scenario_name,
        "--model-dir",
        str(args.model_dir),
        "--threads",
        str(args.threads),
    ]


def candle_correctness_command(args: argparse.Namespace) -> list[str]:
    return [
        "cargo",
        "run",
        "--quiet",
        "--release",
        "--example",
        "benchmark_candle",
        "--",
        "--mode",
        "correctness",
        "--model-dir",
        str(args.model_dir),
        "--threads",
        str(args.threads),
    ]


def pytorch_warm_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "bench_pytorch.py"),
        "--mode",
        "warm",
        "--model-name-or-path",
        str(args.model_dir),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--threads",
        str(args.threads),
    ]


def pytorch_cold_command(args: argparse.Namespace, scenario_name: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "bench_pytorch.py"),
        "--mode",
        "cold",
        "--scenario",
        scenario_name,
        "--model-name-or-path",
        str(args.model_dir),
        "--threads",
        str(args.threads),
    ]


def pytorch_correctness_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "bench_pytorch.py"),
        "--mode",
        "correctness",
        "--model-name-or-path",
        str(args.model_dir),
        "--threads",
        str(args.threads),
    ]


RUNNERS = {
    "ltembed": {
        "warm": ltembed_warm_command,
        "cold": ltembed_cold_command,
        "correctness": ltembed_correctness_command,
        "version": lambda: git_sha(),
    },
    "candle": {
        "warm": candle_warm_command,
        "cold": candle_cold_command,
        "correctness": candle_correctness_command,
        "version": lambda: extract_cargo_lock_version(cargo_lock_text(), "candle-transformers"),
    },
    "pytorch": {
        "warm": pytorch_warm_command,
        "cold": pytorch_cold_command,
        "correctness": pytorch_correctness_command,
        "version": lambda: "",
    },
}


def resolved_implementation_version(implementation: str, payload: dict[str, Any]) -> str:
    if implementation in {"ltembed", "candle"}:
        return RUNNERS[implementation]["version"]()
    return str(payload.get("implementation_version", ""))


def resolved_notes(implementation: str, payload: dict[str, Any]) -> str:
    if implementation == "ltembed":
        backend = str(payload.get("backend", "")).strip()
        if backend:
            return f"dense_backend={backend}"
    return ""


def run_json_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"failed to parse JSON from {' '.join(command)}:\n{completed.stdout}\n{completed.stderr}"
        ) from exc


def base_row_fields(
    *,
    run_id: str,
    timestamp_utc: str,
    model_id: str,
    model_source: str,
    implementation: str,
    implementation_version: str,
    git_revision: str,
    scenario: Scenario,
    mode: str,
    threads: int,
    warmup_iters: int,
    timed_iters: int,
    host: dict[str, str],
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "host_os": host["host_os"],
        "host_arch": host["host_arch"],
        "cpu_model": host["cpu_model"],
        "runner_labels": host["runner_labels"],
        "model_id": model_id,
        "model_source": model_source,
        "implementation": implementation,
        "implementation_version": implementation_version,
        "git_sha": git_revision,
        "scenario": scenario.name,
        "mode": mode,
        "batch_size": str(scenario.batch_size),
        "text_profile": scenario.text_profile,
        "threads": str(threads),
        "warmup_iters": str(warmup_iters),
        "timed_iters": str(timed_iters),
        "mean_ms": "",
        "median_ms": "",
        "p95_ms": "",
        "p99_ms": "",
        "min_ms": "",
        "max_ms": "",
        "cosine_similarity_vs_pytorch": "",
        "status": "pass",
        "notes": "",
    }


def stats_row_from_runner(
    *,
    base_fields: dict[str, str],
    stats: dict[str, Any],
) -> dict[str, str]:
    row = {field: "" for field in CSV_FIELDNAMES}
    row.update(base_fields)
    row["mean_ms"] = f"{float(stats['mean_ms']):.6f}"
    row["median_ms"] = f"{float(stats['median_ms']):.6f}"
    row["p95_ms"] = f"{float(stats['p95_ms']):.6f}"
    row["p99_ms"] = f"{float(stats['p99_ms']):.6f}"
    row["min_ms"] = f"{float(stats['min_ms']):.6f}"
    row["max_ms"] = f"{float(stats['max_ms']):.6f}"
    return row


def collect_warm_rows(
    *,
    args: argparse.Namespace,
    run_id: str,
    timestamp_utc: str,
    host: dict[str, str],
    git_revision: str,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    results: dict[str, dict[str, Any]] = {}
    for implementation, runner in RUNNERS.items():
        payload = run_json_command(runner["warm"](args))
        results[implementation] = payload
        version = resolved_implementation_version(implementation, payload)
        for entry in payload["results"]:
            scenario = scenario_from_name(entry["scenario"])
            base_fields = base_row_fields(
                run_id=run_id,
                timestamp_utc=timestamp_utc,
                model_id=args.model_id,
                model_source=args.model_source,
                implementation=implementation,
                implementation_version=version,
                git_revision=git_revision,
                scenario=scenario,
                mode="warm_latency",
                threads=args.threads,
                warmup_iters=args.warmup,
                timed_iters=args.iters,
                host=host,
            )
            row = stats_row_from_runner(base_fields=base_fields, stats=entry["stats"])
            row["notes"] = resolved_notes(implementation, payload)
            rows.append(row)
    return rows, results


def collect_cold_rows(
    *,
    args: argparse.Namespace,
    run_id: str,
    timestamp_utc: str,
    host: dict[str, str],
    git_revision: str,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    results: dict[str, dict[str, Any]] = {implementation: {} for implementation in RUNNERS}
    for scenario in SCENARIOS:
        for implementation, runner in RUNNERS.items():
            payload = run_json_command(runner["cold"](args, scenario.name))
            results[implementation][scenario.name] = payload
            version = resolved_implementation_version(implementation, payload)
            base_fields = base_row_fields(
                run_id=run_id,
                timestamp_utc=timestamp_utc,
                model_id=args.model_id,
                model_source=args.model_source,
                implementation=implementation,
                implementation_version=version,
                git_revision=git_revision,
                scenario=scenario,
                mode="cold_start",
                threads=args.threads,
                warmup_iters=0,
                timed_iters=1,
                host=host,
            )
            row = stats_row_from_runner(base_fields=base_fields, stats=payload["stats"])
            row["notes"] = resolved_notes(implementation, payload)
            rows.append(row)
    return rows, results


def collect_correctness_rows(
    *,
    args: argparse.Namespace,
    run_id: str,
    timestamp_utc: str,
    host: dict[str, str],
    git_revision: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    payloads: dict[str, Any] = {}
    for implementation, runner in RUNNERS.items():
        payloads[implementation] = run_json_command(runner["correctness"](args))

    reference = payloads["pytorch"]
    for implementation, payload in payloads.items():
        version = resolved_implementation_version(implementation, payload)
        for entry in payload["results"]:
            scenario = scenario_from_name(entry["scenario"])
            base_fields = base_row_fields(
                run_id=run_id,
                timestamp_utc=timestamp_utc,
                model_id=args.model_id,
                model_source=args.model_source,
                implementation=implementation,
                implementation_version=version,
                git_revision=git_revision,
                scenario=scenario,
                mode="correctness",
                threads=args.threads,
                warmup_iters=0,
                timed_iters=0,
                host=host,
            )
            if implementation == "pytorch":
                row = build_correctness_row(base_fields=base_fields, cosine_similarity=1.0, threshold=1.0)
            else:
                reference_entry = next(
                    item for item in reference["results"] if item["scenario"] == entry["scenario"]
                )
                similarities = [
                    cosine_similarity(lhs, rhs)
                    for lhs, rhs in zip(entry["embeddings"], reference_entry["embeddings"], strict=True)
                ]
                average_similarity = sum(similarities) / len(similarities)
                row = build_correctness_row(
                    base_fields=base_fields,
                    cosine_similarity=average_similarity,
                    threshold=args.correctness_threshold,
                )
            row["notes"] = resolved_notes(implementation, payload)
            rows.append(row)
    return rows, payloads


def summary_lines(
    *,
    args: argparse.Namespace,
    git_revision: str,
    warm_payloads: dict[str, dict[str, Any]],
    cold_payloads: dict[str, dict[str, Any]] | None,
    correctness_payloads: dict[str, Any] | None,
) -> list[str]:
    lines = [
        f"run_id={args.run_id}",
        f"git_sha={git_revision}",
        f"model_id={args.model_id}",
        f"model_source={args.model_source}",
        f"python_version={python_version()}",
        f"rust_version={rust_version()}",
    ]
    for implementation in ("ltembed", "candle", "pytorch"):
        payload = warm_payloads.get(implementation, {})
        version = resolved_implementation_version(implementation, payload)
        if implementation == "pytorch":
            torch_version = resolved_implementation_version(implementation, payload)
            transformers_version = payload.get("transformers_version", "")
            lines.append(f"pytorch_version={torch_version}")
            lines.append(f"transformers_version={transformers_version}")
        else:
            lines.append(f"{implementation}_version={version}")
    ltembed_backend = warm_payloads.get("ltembed", {}).get("backend", "")
    if ltembed_backend:
        lines.append(f"ltembed_dense_backend={ltembed_backend}")
    if cold_payloads is not None:
        lines.append("cold_start=enabled")
    if correctness_payloads is not None:
        lines.append("correctness=enabled")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "assets",
        help="Local model directory containing e5-small-v2 assets.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-source", default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--ltembed-cargo-features",
        default="",
        help="Optional cargo feature list to enable for LTEmbed runs.",
    )
    parser.add_argument("--include-cold-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-correctness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--correctness-threshold", type=float, default=DEFAULT_CORRECTNESS_THRESHOLD)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "artifacts" / "benchmark-report.csv",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=ROOT / "artifacts" / "benchmark-summary.txt",
    )
    parser.add_argument("--run-id", default=f"bench-{uuid.uuid4().hex[:12]}")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = utc_now()
    git_revision = git_sha()
    host = host_metadata()
    rows: list[dict[str, str]] = []

    warm_rows, warm_payloads = collect_warm_rows(
        args=args,
        run_id=args.run_id,
        timestamp_utc=timestamp,
        host=host,
        git_revision=git_revision,
    )
    rows.extend(warm_rows)

    cold_payloads = None
    if args.include_cold_start:
        cold_rows, cold_payloads = collect_cold_rows(
            args=args,
            run_id=args.run_id,
            timestamp_utc=timestamp,
            host=host,
            git_revision=git_revision,
        )
        rows.extend(cold_rows)

    correctness_payloads = None
    if args.include_correctness:
        correctness_rows, correctness_payloads = collect_correctness_rows(
            args=args,
            run_id=args.run_id,
            timestamp_utc=timestamp,
            host=host,
            git_revision=git_revision,
        )
        rows.extend(correctness_rows)

    write_csv_report(rows, args.output_csv)

    lines = summary_lines(
        args=args,
        git_revision=git_revision,
        warm_payloads=warm_payloads,
        cold_payloads=cold_payloads,
        correctness_payloads=correctness_payloads,
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
