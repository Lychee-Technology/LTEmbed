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
    "both_at_3",
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
    Scenario(name="single/short", batch_size=1, text_profile="short"),
    Scenario(name="single/medium", batch_size=1, text_profile="medium"),
    Scenario(name="single/long", batch_size=1, text_profile="long"),
    Scenario(name="batch/medium/8", batch_size=8, text_profile="medium"),
    Scenario(name="batch/mixed/8", batch_size=8, text_profile="mixed"),
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


def _l2_normalize_rows(matrix: Any) -> Any:
    """Row-wise L2 normalization so a dot product equals cosine similarity."""
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def scenario_from_name(name: str) -> Scenario:
    try:
        return SCENARIO_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown scenario: {name}") from exc


def load_corpus_texts(jsonl_path: Path) -> list[str]:
    """Read a JSONL corpus (e.g. jane-austen), returning non-empty texts sorted by length.

    Sorting by (token_count, position) is fully deterministic, so every quant job and both
    runners select byte-identical chunks from the same file.
    """
    texts: list[tuple[int, int, str]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        text = (record.get("text") or "").strip()
        if not text:
            continue
        token_count = int(record.get("token_count", len(text.split())))
        position = int(record.get("position", len(texts)))
        texts.append((token_count, position, text))
    if not texts:
        raise ValueError(f"no usable 'text' records in {jsonl_path}")
    texts.sort(key=lambda item: (item[0], item[1]))
    return [text for _, _, text in texts]


def resolve_fixture(jsonl_path: Path, scenarios: list[Scenario]) -> dict[str, Any]:
    """Select per-scenario texts from a corpus, keyed by scenario name.

    For each scenario we emit exactly ``batch_size`` inputs, chosen by ``text_profile``:
    short/medium/long pull from the short/median/long ends of the length distribution, and
    batches draw *distinct* chunks so latency and cosine-vs-FP32 see real variety.
    """
    corpus = load_corpus_texts(jsonl_path)
    n = len(corpus)
    short_text = corpus[0]
    long_text = corpus[-1]

    def medium_at(index: int) -> str:
        # Distinct chunks stepping outward from the median.
        return corpus[(n // 2 + index) % n]

    def spread_at(index: int, count: int) -> str:
        # Evenly spread across the whole distribution for mixed batches.
        return corpus[(n * (index + 1) // (count + 1)) % n]

    resolved: dict[str, list[dict[str, str]]] = {}
    for scenario in scenarios:
        batch_size = scenario.batch_size
        profile = scenario.text_profile
        items: list[dict[str, str]] = []
        if profile == "short":
            items = [{"kind": "query", "text": short_text} for _ in range(batch_size)]
        elif profile == "long":
            items = [{"kind": "document", "text": long_text} for _ in range(batch_size)]
        elif profile == "mixed":
            base = [
                {"kind": "query", "text": short_text},
                {"kind": "query", "text": medium_at(0)},
                {"kind": "document", "text": long_text},
            ]
            for i in range(batch_size):
                if i < len(base):
                    items.append(base[i])
                else:
                    kind = "query" if i % 2 == 0 else "document"
                    items.append({"kind": kind, "text": spread_at(i, batch_size)})
        else:  # "medium" and any other profile default to medium chunks
            items = [{"kind": "query", "text": medium_at(k)} for k in range(batch_size)]
        resolved[scenario.name] = items
    return {"source": str(jsonl_path), "scenarios": resolved}


def load_retrieval_eval_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = list(payload["cases"]) if "cases" in payload else [payload]
    for case in cases:
        if not isinstance(case.get("name"), str):
            raise ValueError("retrieval eval case missing 'name'")
    if not cases:
        raise ValueError("no retrieval eval cases found")
    return cases


def _prebuilt_ltembed_binary() -> Path | None:
    """The compiled release binary if it exists, else None (fall back to ``cargo run``)."""
    candidate = ROOT / "target" / "release" / "benchmark_ltembed"
    return candidate if candidate.exists() else None


def ltembed_launch_prefix(cargo_features: str = "") -> list[str]:
    """Command prefix (up to but excluding ``--mode``) that launches benchmark_ltembed.

    Prefers the prebuilt release binary so cargo's per-launch freshness check and the
    multi-minute release compile stay out of the timed harness. Falls back to
    ``cargo run --release`` for local ad-hoc use, honoring optional cargo features.
    """
    prebuilt = _prebuilt_ltembed_binary()
    if prebuilt is not None:
        return [str(prebuilt)]
    command = ["cargo", "run", "--quiet", "--release"]
    if cargo_features:
        command.extend(["--features", cargo_features])
    command.extend(["--bin", "benchmark_ltembed", "--"])
    return command


def _append_shared_benchmark_args(
    command: list[str],
    mode: str,
    args: argparse.Namespace,
    scenario_name: str,
) -> list[str]:
    """Append the arguments both runners share, keyed by mode."""
    command.extend(
        [
            "--output-dimension",
            str(args.output_dimension),
            "--l2-normalize",
            "true" if args.l2_normalize else "false",
            "--threads",
            str(args.threads),
        ]
    )
    if mode == "warm":
        command.extend(["--warmup", str(args.warmup), "--iters", str(args.iters)])
        if getattr(args, "scenario", None):
            command.extend(["--scenario", str(args.scenario)])
    elif mode == "cold":
        command.extend(["--scenario", scenario_name])
    elif mode == "retrieval":
        command.extend(["--retrieval-eval-path", str(args.retrieval_eval_path)])
    fixture_path = getattr(args, "resolved_fixture_path", None)
    if fixture_path:
        command.extend(["--fixture-path", str(fixture_path)])
    return command


def build_benchmark_command(
    implementation: str,
    mode: str,
    args: argparse.Namespace,
    scenario_name: str = "",
) -> list[str]:
    if implementation == "ltembed":
        command = ltembed_launch_prefix(getattr(args, "ltembed_cargo_features", ""))
        command.extend(["--mode", mode, "--bundle-dir", str(args.bundle_dir)])
    else:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "bench_pytorch.py"),
            "--mode",
            mode,
            "--model-name-or-path",
            str(args.model_dir),
        ]
    return _append_shared_benchmark_args(command, mode, args, scenario_name)


def resolved_implementation_version(
    implementation: str,
    payload: dict[str, Any],
    git_revision: str | None = None,
) -> str:
    if implementation == "ltembed":
        return git_revision if git_revision is not None else git_sha()
    return str(payload.get("implementation_version", ""))


def gather_payload(
    implementation: str,
    mode: str,
    args: argparse.Namespace,
    *,
    reference: dict[str, Any] | None,
    scenario_name: str = "",
) -> dict[str, Any]:
    """Return an implementation's payload for a mode, sourced from the reference when possible.

    In reference-consume mode the quant-independent PyTorch payloads (correctness, retrieval)
    are loaded from the reference JSON instead of launching PyTorch; everything else runs as a
    subprocess.
    """
    if implementation == "pytorch" and reference is not None and mode in reference:
        return reference[mode]
    label = f"{implementation} {mode}" + (f" {scenario_name}" if scenario_name else "")
    return run_json_command(build_benchmark_command(implementation, mode, args, scenario_name), label)


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
        # rows never override notes; the CSV column is always empty
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
    row["both_at_3"] = f"{float(metrics['both_at_3']):.6f}"
    row["mrr_at_3"] = f"{float(metrics['mrr_at_3']):.6f}"
    return row


def collect_warm_rows(
    *,
    args: argparse.Namespace,
    ctx: RunContext,
    implementations: list[str],
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    results: dict[str, dict[str, Any]] = {}
    for implementation in implementations:
        payload = run_json_command(
            build_benchmark_command(implementation, "warm", args), f"{implementation} warm"
        )
        results[implementation] = payload
        version = resolved_implementation_version(implementation, payload, ctx.git_revision)
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
    implementations: list[str],
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    results: dict[str, dict[str, Any]] = {implementation: {} for implementation in implementations}
    for scenario in SCENARIOS:
        for implementation in implementations:
            payload = run_json_command(
                build_benchmark_command(implementation, "cold", args, scenario.name),
                f"{implementation} cold {scenario.name}",
            )
            results[implementation][scenario.name] = payload
            version = resolved_implementation_version(implementation, payload, ctx.git_revision)
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
    implementations: list[str],
    reference: dict[str, Any] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    payloads: dict[str, Any] = {}
    for implementation in implementations:
        payloads[implementation] = gather_payload(
            implementation, "correctness", args, reference=reference
        )

    reference_payload = payloads["pytorch"]
    for implementation, payload in payloads.items():
        version = resolved_implementation_version(implementation, payload, ctx.git_revision)
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
                    item for item in reference_payload["results"] if item["scenario"] == entry["scenario"]
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
    """Rank documents per query and score against each query's *set* of relevant ids.

    A query may have multiple relevant documents (the CN/EN cross-lingual case marks
    both the self-language document and its translation as relevant). Metrics:

    - ``both_at_3``: fraction of queries where *all* relevant documents are in top-3
      — the "同时得到中英" success rate.
    - ``recall_at_1`` / ``recall_at_3``: mean fraction of relevant documents found in
      the top-1 / top-3 (0.5 when only one of two is found, 1.0 when both).
    - ``mrr_at_3``: mean reciprocal rank of the first relevant document within top-3.

    Ranking is vectorized with numpy (cosine == dot after L2 normalization); a stable
    argsort preserves document insertion order on ties.
    """
    import numpy as np

    empty = {"query_count": 0, "recall_at_1": 0.0, "recall_at_3": 0.0, "both_at_3": 0.0, "mrr_at_3": 0.0}
    doc_ids = list(document_embeddings.keys())
    if not doc_ids:
        return empty

    doc_matrix = _l2_normalize_rows(np.asarray([document_embeddings[d] for d in doc_ids], dtype=np.float64))
    corpus_ids = set(doc_ids)

    query_vectors: list[list[float]] = []
    query_relevant: list[set[str]] = []
    for query in retrieval_case["queries"]:
        relevant_ids = {str(doc_id) for doc_id in query["relevant_document_ids"]} & corpus_ids
        if not relevant_ids:
            continue
        query_vectors.append(query_embeddings[str(query["id"])])
        query_relevant.append(relevant_ids)

    query_count = len(query_vectors)
    if query_count == 0:
        return empty

    query_matrix = _l2_normalize_rows(np.asarray(query_vectors, dtype=np.float64))
    similarities = query_matrix @ doc_matrix.T  # (num_queries, num_docs)
    top_k = min(3, len(doc_ids))
    top_indices = np.argsort(-similarities, axis=1, kind="stable")[:, :top_k]

    recall_at_1 = recall_at_3 = both_at_3 = mrr_at_3 = 0.0
    for row, relevant in zip(top_indices, query_relevant):
        ranked_ids = [doc_ids[index] for index in row]
        n_relevant = len(relevant)
        found_at_1 = 1 if ranked_ids and ranked_ids[0] in relevant else 0
        found_at_3 = sum(1 for doc_id in ranked_ids if doc_id in relevant)
        recall_at_1 += found_at_1 / n_relevant
        recall_at_3 += found_at_3 / n_relevant
        both_at_3 += 1.0 if found_at_3 == n_relevant else 0.0
        for rank, doc_id in enumerate(ranked_ids, start=1):
            if doc_id in relevant:
                mrr_at_3 += 1.0 / rank
                break

    return {
        "query_count": query_count,
        "recall_at_1": recall_at_1 / query_count,
        "recall_at_3": recall_at_3 / query_count,
        "both_at_3": both_at_3 / query_count,
        "mrr_at_3": mrr_at_3 / query_count,
    }


def collect_retrieval_eval_rows(
    *,
    args: argparse.Namespace,
    ctx: RunContext,
    implementations: list[str],
    reference: dict[str, Any] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    retrieval_cases = load_retrieval_eval_cases(args.retrieval_eval_path)
    cases_by_name = {str(case["name"]): case for case in retrieval_cases}
    rows: list[dict[str, str]] = []
    payloads: dict[str, Any] = {}

    for implementation in implementations:
        payload = gather_payload(implementation, "retrieval", args, reference=reference)
        payloads[implementation] = payload
        version = resolved_implementation_version(implementation, payload, ctx.git_revision)
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
    reference: dict[str, Any] | None = None,
) -> list[str]:
    lines = [
        f"run_id={args.run_id}",
        f"git_sha={git_revision}",
        f"model_id={args.model_id}",
        f"model_source={args.model_source}",
        f"python_version={python_version()}",
        f"rust_version={rust_version()}",
    ]
    lines.append(
        f"ltembed_version={resolved_implementation_version('ltembed', warm_payloads.get('ltembed', {}), git_revision)}"
    )
    # In reference-consume mode PyTorch never runs a latency pass, so pull its versions from
    # the loaded reference (correctness payload) instead of the warm payloads.
    pytorch_payload = warm_payloads.get("pytorch")
    if pytorch_payload is None and reference is not None:
        pytorch_payload = reference.get("correctness")
    pytorch_payload = pytorch_payload or {}
    lines.append(f"pytorch_version={pytorch_payload.get('implementation_version', '')}")
    lines.append(f"transformers_version={pytorch_payload.get('transformers_version', '')}")
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
        "--bundle-dir",
        type=Path,
        default=ROOT / "gguf_bundle",
        help="Local LTEmbed GGUF bundle directory containing model.gguf, tokenizer.json, and build-info.json.",
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
        "--fixture-path",
        type=Path,
        default=None,
        help=(
            "Optional JSONL corpus (e.g. jane-austen). When set, per-scenario texts are "
            "selected from it and fed identically to both runners so the cosine comparison "
            "reflects real prose. Omit to use the built-in synthetic texts."
        ),
    )
    parser.add_argument(
        "--ltembed-cargo-features",
        default="",
        help="Optional cargo features to pass through to LTEmbed benchmark builds.",
    )
    parser.add_argument(
        "--emit-reference",
        type=Path,
        default=None,
        help=(
            "Run ONLY the PyTorch runner (correctness + retrieval) and write an embeddings-only "
            "reference JSON to this path, then exit. The PyTorch reference is quant-independent, "
            "so it is produced once per workflow and shared with every quant job."
        ),
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=None,
        help=(
            "Path to a reference JSON produced by --emit-reference. When set, warm/cold run "
            "ltembed only and the correctness/retrieval PyTorch baseline is loaded from the "
            "reference instead of launching PyTorch."
        ),
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


def resolve_fixture_if_present(args: argparse.Namespace) -> None:
    """Resolve per-scenario texts from a corpus once and point ``resolved_fixture_path`` at them.

    ``resolve_fixture`` is deterministic, so the reference producer and every quant job select
    byte-identical inputs from the same corpus (required for a valid cosine comparison).
    """
    if getattr(args, "fixture_path", None):
        resolved = resolve_fixture(args.fixture_path, SCENARIOS)
        resolved_path = args.output_csv.parent / "resolved_fixture.json"
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8")
        args.resolved_fixture_path = resolved_path


def main() -> int:
    args = parse_args()
    timestamp = utc_now()
    git_revision = git_sha()
    host = host_metadata()
    if getattr(args, "emit_reference", None) is not None:
        return _emit_reference(args=args)
    rows: list[dict[str, str]] = []
    return _run(args=args, timestamp=timestamp, git_revision=git_revision, host=host, rows=rows)


def _emit_reference(*, args: argparse.Namespace) -> int:
    """Run only the PyTorch runner (correctness + retrieval) and write an embeddings-only reference."""
    resolve_fixture_if_present(args)
    reference = {
        "correctness": run_json_command(
            build_benchmark_command("pytorch", "correctness", args), "pytorch correctness"
        ),
        "retrieval": run_json_command(
            build_benchmark_command("pytorch", "retrieval", args), "pytorch retrieval"
        ),
    }
    args.emit_reference.parent.mkdir(parents=True, exist_ok=True)
    args.emit_reference.write_text(json.dumps(reference), encoding="utf-8")
    print(f"wrote PyTorch reference (correctness + retrieval) to {args.emit_reference}")
    return 0


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

    resolve_fixture_if_present(args)

    # Reference-consume mode: PyTorch (a quant-independent baseline) never runs here. Latency
    # passes are ltembed-only; correctness/retrieval load the PyTorch baseline from the reference.
    reference: dict[str, Any] | None = None
    if getattr(args, "reference_path", None) is not None:
        reference = json.loads(Path(args.reference_path).read_text(encoding="utf-8"))
    latency_impls = ["ltembed"] if reference is not None else ["ltembed", "pytorch"]
    embedding_impls = ["ltembed", "pytorch"]

    warm_rows, warm_payloads = collect_warm_rows(args=args, ctx=ctx, implementations=latency_impls)
    rows.extend(warm_rows)

    cold_payloads = None
    if args.include_cold_start:
        cold_rows, cold_payloads = collect_cold_rows(args=args, ctx=ctx, implementations=latency_impls)
        rows.extend(cold_rows)

    correctness_payloads = None
    if args.include_correctness:
        correctness_rows, correctness_payloads = collect_correctness_rows(
            args=args, ctx=ctx, implementations=embedding_impls, reference=reference
        )
        rows.extend(correctness_rows)

    retrieval_payloads = None
    if args.include_retrieval_eval:
        retrieval_rows, retrieval_payloads = collect_retrieval_eval_rows(
            args=args, ctx=ctx, implementations=embedding_impls, reference=reference
        )
        rows.extend(retrieval_rows)

    write_csv_report(rows, args.output_csv)

    lines = summary_lines(
        args=args,
        git_revision=git_revision,
        warm_payloads=warm_payloads,
        cold_payloads=cold_payloads,
        correctness_payloads=correctness_payloads,
        retrieval_payloads=retrieval_payloads,
        reference=reference,
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
