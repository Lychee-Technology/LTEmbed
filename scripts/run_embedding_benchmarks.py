#!/usr/bin/env python3
"""
Orchestrate LTEmbed and PyTorch embedding benchmarks.

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
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "jinaai/jina-embeddings-v5-text-nano-retrieval"
DEFAULT_MODEL_SOURCE = "huggingface"
DEFAULT_CORRECTNESS_THRESHOLD = 0.98
DEFAULT_RETRIEVAL_EVAL_PATH = ROOT / "scripts" / "retrieval_eval_cases.json"
RUNNER_LABELS_ENV = "BENCHMARK_RUNNER_LABELS"

SHORT_TEXT = {"kind": "query", "text": "Hello, world!"}
MEDIUM_TEXT = {
    "kind": "query",
    "text": "What is the impact of large language models on software engineering productivity?",
}
LONG_TEXT = {
    "kind": "document",
    "text": "The quick brown fox jumps over the lazy dog. " * 30,
}

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
    "query_count",
    "recall_at_1",
    "recall_at_3",
    "mrr_at_3",
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
    def texts(self) -> tuple[dict[str, str], ...]:
        return self["texts"]


class RunContext:
    """Run-level fields shared by all row builders."""

    def __init__(
        self,
        *,
        run_id: str,
        timestamp_utc: str,
        model_id: str,
        model_source: str,
        git_revision: str,
        host: dict[str, str],
    ):
        self.run_id = run_id
        self.timestamp_utc = timestamp_utc
        self.model_id = model_id
        self.model_source = model_source
        self.git_revision = git_revision
        self.host = host


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


def load_retrieval_eval_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = list(payload["cases"]) if "cases" in payload else [payload]
    for case in cases:
        if not isinstance(case.get("name"), str):
            raise ValueError("retrieval eval case missing 'name'")
    if not cases:
        raise ValueError("no retrieval eval cases found")
    return cases


def cargo_run_prefix(cargo_features: str = "") -> list[str]:
    command = ["cargo"]
    command.extend(["run", "--quiet", "--release"])
    if cargo_features:
        command.extend(["--features", cargo_features])
    return command


def build_benchmark_command(
    implementation: str,
    mode: str,
    args: argparse.Namespace,
    scenario_name: str = "",
) -> list[str]:
    if implementation == "ltembed":
        command = cargo_run_prefix(getattr(args, "ltembed_cargo_features", ""))
        command.extend(
            ["--bin", "benchmark_ltembed", "--", "--mode", mode]
        )
        command.extend(
            [
                "--ort-bundle-dir",
                str(args.ort_bundle_dir),
                "--output-dimension",
                str(args.output_dimension),
                "--l2-normalize",
                "true" if args.l2_normalize else "false",
                "--threads",
                str(args.threads),
            ]
        )
        if mode == "warm":
            command.extend(
                ["--warmup", str(args.warmup), "--iters", str(args.iters)]
            )
            if getattr(args, "scenario", None):
                command.extend(["--scenario", str(args.scenario)])
        elif mode == "cold":
            command.extend(["--scenario", scenario_name])
        elif mode == "retrieval":
            command.extend(
                ["--retrieval-eval-path", str(args.retrieval_eval_path)]
            )
        return command

    command = [
        sys.executable,
        str(ROOT / "scripts" / "bench_pytorch.py"),
        "--mode",
        mode,
        "--model-name-or-path",
        str(args.model_dir),
        "--output-dimension",
        str(args.output_dimension),
        "--l2-normalize",
        "true" if args.l2_normalize else "false",
        "--threads",
        str(args.threads),
    ]
    if mode == "warm":
        command.extend(
            ["--warmup", str(args.warmup), "--iters", str(args.iters)]
        )
    elif mode == "cold":
        command.extend(["--scenario", scenario_name])
    elif mode == "retrieval":
        command.extend(
            ["--retrieval-eval-path", str(args.retrieval_eval_path)]
        )
    return command


RUNNERS = {
    "ltembed": {
        "warm": lambda args: build_benchmark_command("ltembed", "warm", args),
        "cold": lambda args, scenario_name: build_benchmark_command("ltembed", "cold", args, scenario_name),
        "correctness": lambda args: build_benchmark_command("ltembed", "correctness", args),
        "retrieval": lambda args: build_benchmark_command("ltembed", "retrieval", args),
        "version": lambda: git_sha(),
    },
    "pytorch": {
        "warm": lambda args: build_benchmark_command("pytorch", "warm", args),
        "cold": lambda args, scenario_name: build_benchmark_command("pytorch", "cold", args, scenario_name),
        "correctness": lambda args: build_benchmark_command("pytorch", "correctness", args),
        "retrieval": lambda args: build_benchmark_command("pytorch", "retrieval", args),
        "version": lambda: "",
    },
}


def resolved_implementation_version(implementation: str, payload: dict[str, Any]) -> str:
    if implementation == "ltembed":
        return RUNNERS[implementation]["version"]()
    return str(payload.get("implementation_version", ""))



def log_progress(label: str, state: str, elapsed_seconds: float | None = None) -> None:
    timestamp = utc_now()
    suffix = ""
    if elapsed_seconds is not None:
        suffix = f" ({elapsed_seconds:.1f}s)"
    print(f"[{timestamp}] {state} {label}{suffix}", file=sys.stderr, flush=True)


def run_json_command(command: list[str], label: str) -> dict[str, Any]:
    started_at = time.perf_counter()
    log_progress(label, "START")
    try:
        completed = subprocess.run(
            command,
            check=True,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        # Surface Cargo build errors and binary stderr that would otherwise be
        # swallowed by stdout capture.
        print(
            f"\n--- command failed (exit {exc.returncode}): {' '.join(exc.cmd)}\n"
            f"--- stdout ---\n{exc.stdout}",
            file=sys.stderr,
        )
        raise
    log_progress(label, "DONE", time.perf_counter() - started_at)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"failed to parse JSON from {' '.join(command)}:\n{completed.stdout}"
        ) from exc


def base_row_fields(
    *,
    ctx: RunContext,
    implementation: str,
    implementation_version: str,
    scenario: Scenario,
    mode: str,
    threads: int,
    warmup_iters: int,
    timed_iters: int,
) -> dict[str, str]:
    return {
        "run_id": ctx.run_id,
        "timestamp_utc": ctx.timestamp_utc,
        "host_os": ctx.host["host_os"],
        "host_arch": ctx.host["host_arch"],
        "cpu_model": ctx.host["cpu_model"],
        "runner_labels": ctx.host["runner_labels"],
        "model_id": ctx.model_id,
        "model_source": ctx.model_source,
        "implementation": implementation,
        "implementation_version": implementation_version,
        "git_sha": ctx.git_revision,
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


def retrieval_eval_row_from_metrics(
    *,
    base_fields: dict[str, str],
    metrics: dict[str, Any],
) -> dict[str, str]:
    row = {field: "" for field in CSV_FIELDNAMES}
    row.update(base_fields)
    row["query_count"] = str(int(metrics["query_count"]))
    row["recall_at_1"] = f"{float(metrics['recall_at_1']):.6f}"
    row["recall_at_3"] = f"{float(metrics['recall_at_3']):.6f}"
    row["mrr_at_3"] = f"{float(metrics['mrr_at_3']):.6f}"
    return row


def collect_warm_rows(
    *,
    args: argparse.Namespace,
    ctx: RunContext,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    results: dict[str, dict[str, Any]] = {}
    for implementation, runner in RUNNERS.items():
        payload = run_json_command(runner["warm"](args), f"{implementation} warm")
        results[implementation] = payload
        version = resolved_implementation_version(implementation, payload)
        for entry in payload["results"]:
            scenario = scenario_from_name(entry["scenario"])
            base_fields = base_row_fields(
                ctx=ctx,
                implementation=implementation,
                implementation_version=version,
                scenario=scenario,
                mode="warm_latency",
                threads=args.threads,
                warmup_iters=args.warmup,
                timed_iters=args.iters,
            )
            row = stats_row_from_runner(base_fields=base_fields, stats=entry["stats"])

            rows.append(row)
    return rows, results


def collect_cold_rows(
    *,
    args: argparse.Namespace,
    ctx: RunContext,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    results: dict[str, dict[str, Any]] = {implementation: {} for implementation in RUNNERS}
    for scenario in SCENARIOS:
        for implementation, runner in RUNNERS.items():
            payload = run_json_command(
                runner["cold"](args, scenario.name),
                f"{implementation} cold {scenario.name}",
            )
            results[implementation][scenario.name] = payload
            version = resolved_implementation_version(implementation, payload)
            base_fields = base_row_fields(
                ctx=ctx,
                implementation=implementation,
                implementation_version=version,
                scenario=scenario,
                mode="cold_start",
                threads=args.threads,
                warmup_iters=0,
                timed_iters=1,
            )
            row = stats_row_from_runner(base_fields=base_fields, stats=payload["stats"])

            rows.append(row)
    return rows, results


def collect_correctness_rows(
    *,
    args: argparse.Namespace,
    ctx: RunContext,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    payloads: dict[str, Any] = {}
    for implementation, runner in RUNNERS.items():
        payloads[implementation] = run_json_command(
            runner["correctness"](args),
            f"{implementation} correctness",
        )

    reference = payloads["pytorch"]
    for implementation, payload in payloads.items():
        version = resolved_implementation_version(implementation, payload)
        for entry in payload["results"]:
            scenario = scenario_from_name(entry["scenario"])
            base_fields = base_row_fields(
                ctx=ctx,
                implementation=implementation,
                implementation_version=version,
                scenario=scenario,
                mode="correctness",
                threads=args.threads,
                warmup_iters=0,
                timed_iters=0,
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

            rows.append(row)
    return rows, payloads


def compute_retrieval_metrics(
    retrieval_case: dict[str, Any],
    *,
    query_embeddings: dict[str, list[float]],
    document_embeddings: dict[str, list[float]],
) -> dict[str, Any]:
    query_results = []
    for query in retrieval_case["queries"]:
        query_id = str(query["id"])
        query_emb = query_embeddings[query_id]
        relevant_ids = set(str(doc_id) for doc_id in query["relevant_document_ids"])
        if len(relevant_ids) == 0:
            continue

        similarities = [
            (doc_id, cosine_similarity(query_emb, doc_emb))
            for doc_id, doc_emb in document_embeddings.items()
        ]
        similarities.sort(key=lambda item: item[1], reverse=True)

        rank = None
        for i, (doc_id, _) in enumerate(similarities):
            if doc_id in relevant_ids:
                rank = i + 1
                break

        query_results.append(
            {
                "query_id": query_id,
                "relevant_document_ids": sorted(relevant_ids),
                "rank_of_first_relevant": rank,
                "relevant_at_1": 1.0 if rank is not None and rank <= 1 else 0.0,
                "relevant_at_3": 1.0 if rank is not None and rank <= 3 else 0.0,
                "reciprocal_rank": 1.0 / rank if rank is not None and rank <= 3 else 0.0,
            }
        )

    query_count = len(query_results)
    return {
        "query_count": query_count,
        "recall_at_1": sum(q["relevant_at_1"] for q in query_results) / query_count if query_count > 0 else 0.0,
        "recall_at_3": sum(q["relevant_at_3"] for q in query_results) / query_count if query_count > 0 else 0.0,
        "mrr_at_3": sum(q["reciprocal_rank"] for q in query_results) / query_count if query_count > 0 else 0.0,
    }


def collect_retrieval_eval_rows(
    *,
    args: argparse.Namespace,
    ctx: RunContext,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    retrieval_cases = load_retrieval_eval_cases(args.retrieval_eval_path)
    cases_by_name = {str(case["name"]): case for case in retrieval_cases}
    rows: list[dict[str, str]] = []
    payloads: dict[str, Any] = {}

    for implementation, runner in RUNNERS.items():
        payload = run_json_command(
            runner["retrieval"](args),
            f"{implementation} retrieval",
        )
        payloads[implementation] = payload
        version = resolved_implementation_version(implementation, payload)
        for result in payload["results"]:
            case = cases_by_name[str(result["dataset_name"])]
            metrics = compute_retrieval_metrics(
                case,
                query_embeddings={str(item["id"]): item["embedding"] for item in result["queries"]},
                document_embeddings={str(item["id"]): item["embedding"] for item in result["documents"]},
            )
            scenario = Scenario(
                name=str(case["name"]),
                batch_size=len(case["documents"]),
                text_profile="retrieval_eval",
                texts=(),
            )
            base_fields = base_row_fields(
                ctx=ctx,
                implementation=implementation,
                implementation_version=version,
                scenario=scenario,
                mode="retrieval_eval",
                threads=args.threads,
                warmup_iters=0,
                timed_iters=0,
            )
            row = retrieval_eval_row_from_metrics(base_fields=base_fields, metrics=metrics)

            rows.append(row)
    return rows, payloads


def summary_lines(
    *,
    args: argparse.Namespace,
    git_revision: str,
    warm_payloads: dict[str, dict[str, Any]],
    cold_payloads: dict[str, dict[str, Any]] | None,
    correctness_payloads: dict[str, Any] | None,
    retrieval_payloads: dict[str, Any] | None = None,
) -> list[str]:
    lines = [
        f"run_id={args.run_id}",
        f"git_sha={git_revision}",
        f"model_id={args.model_id}",
        f"model_source={args.model_source}",
        f"python_version={python_version()}",
        f"rust_version={rust_version()}",
    ]
    for implementation in ("ltembed", "pytorch"):
        payload = warm_payloads.get(implementation, {})
        version = resolved_implementation_version(implementation, payload)
        if implementation == "pytorch":
            torch_version = resolved_implementation_version(implementation, payload)
            transformers_version = payload.get("transformers_version", "")
            lines.append(f"pytorch_version={torch_version}")
            lines.append(f"transformers_version={transformers_version}")
        else:
            lines.append(f"{implementation}_version={version}")
    if cold_payloads is not None:
        lines.append("cold_start=enabled")
    if correctness_payloads is not None:
        lines.append("correctness=enabled")
    if retrieval_payloads is not None:
        lines.append("retrieval_eval=enabled")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "assets",
        help="Local model directory for the PyTorch reference runner.",
    )
    parser.add_argument(
        "--ort-bundle-dir",
        type=Path,
        default=ROOT / "ort_bundle",
        help="Local LTEmbed ORT bundle directory containing model.ort, tokenizer.json, build-info.json, and libonnxruntime.so.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-source", default=DEFAULT_MODEL_SOURCE)
    parser.add_argument(
        "--scenario",
        help="Optional single scenario name to run instead of the full suite.",
    )
    parser.add_argument("--output-dimension", type=int, default=512)
    parser.add_argument("--l2-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--retrieval-eval-path",
        type=Path,
        default=DEFAULT_RETRIEVAL_EVAL_PATH,
    )
    parser.add_argument(
        "--ltembed-cargo-features",
        default="",
        help="Optional cargo features to pass through to LTEmbed benchmark builds.",
    )
    parser.add_argument("--include-cold-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-correctness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-retrieval-eval", action=argparse.BooleanOptionalAction, default=True)
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
    return _run(args=args, timestamp=timestamp, git_revision=git_revision, host=host, rows=rows)


def _run(
    *,
    args: argparse.Namespace,
    timestamp: str,
    git_revision: str,
    host: dict[str, str],
    rows: list[dict[str, str]],
) -> int:
    ctx = RunContext(
        run_id=args.run_id,
        timestamp_utc=timestamp,
        model_id=args.model_id,
        model_source=args.model_source,
        git_revision=git_revision,
        host=host,
    )

    warm_rows, warm_payloads = collect_warm_rows(args=args, ctx=ctx)
    rows.extend(warm_rows)

    cold_payloads = None
    if args.include_cold_start:
        cold_rows, cold_payloads = collect_cold_rows(args=args, ctx=ctx)
        rows.extend(cold_rows)

    correctness_payloads = None
    if args.include_correctness:
        correctness_rows, correctness_payloads = collect_correctness_rows(args=args, ctx=ctx)
        rows.extend(correctness_rows)

    retrieval_payloads = None
    if args.include_retrieval_eval:
        retrieval_rows, retrieval_payloads = collect_retrieval_eval_rows(args=args, ctx=ctx)
        rows.extend(retrieval_rows)

    write_csv_report(rows, args.output_csv)

    lines = summary_lines(
        args=args,
        git_revision=git_revision,
        warm_payloads=warm_payloads,
        cold_payloads=cold_payloads,
        correctness_payloads=correctness_payloads,
        retrieval_payloads=retrieval_payloads,
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
