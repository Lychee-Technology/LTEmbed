#!/usr/bin/env python3
"""render_benchmark_report.py — aggregate per-quant benchmark artifacts into one report.

    render_benchmark_report.py <input_dir> [output_dir]

Recursively finds every quant's ``metadata.json`` under ``<input_dir>`` (as produced by the
benchmark matrix jobs via ``write_benchmark_metadata.py``) and reads the sibling
``benchmark-report.csv``. Writes:

    <output_dir>/report.md      cross-quant Markdown comparison (size/latency/parity) with
                                the recommended quant under the Lambda bundle-size budget
    <output_dir>/results.json   {schema_version, records[], quants[], recommendation}

``records`` holds one normalized entry per (quant × scenario × warm/cold) carrying the full
latency distribution (min/mean/median==p50/p95/p99/max) plus the run's metadata.
``quants`` holds one summary per quant: sizes, Lambda fit, parity vs the immutable golden
fixture and vs the workflow's dynamic FP32 reference, and retrieval quality.

A quant is recommendable only when its mean cosine vs FP32 stays at or above the quality
gate AND its bundle (model.gguf + tokenizer.json + build-info.json) fits the uncompressed
AWS Lambda package limit.

If ``$GITHUB_STEP_SUMMARY`` is set, the report is also appended there so it renders in the
Actions run UI. ``<output_dir>`` defaults to the current directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# Mean cosine (vs PyTorch FP32) a quant must keep to be considered "quality-preserving".
QUALITY_GATE = 0.99

# Lambda size contract: the deployed package is the GGUF bundle (model.gguf +
# tokenizer.json + build-info.json — metadata.json's bundle_size_bytes) plus the
# bootstrap binary. AWS caps the unzipped deployment package at 250 MiB, so the
# bundle's effective budget reserves an allowance for the binary. This reproduces the
# tradeoff #150 requires: a ~233 MB Q8_0 GGUF is over budget once the tokenizer and
# binary ride along, while a ~169 MB Q5_K_M fits comfortably.
LAMBDA_PACKAGE_LIMIT_BYTES = 250 * 1024 * 1024
LAMBDA_BINARY_ALLOWANCE_BYTES = 20 * 1024 * 1024
LAMBDA_BUDGET_BYTES = LAMBDA_PACKAGE_LIMIT_BYTES - LAMBDA_BINARY_ALLOWANCE_BYTES

LATENCY_KEYS = ("min", "mean", "median", "p95", "p99", "max")
PHASE_BY_MODE = {"warm_latency": "warm", "cold_start": "cold"}
PHASE_ORDER = {"warm": 0, "cold": 1}

# Metadata denormalized onto every latency record (issue #150 per-record metadata).
RECORD_METADATA_KEYS = [
    "backend",
    "model_id",
    "model_file",
    "model_sha256",
    "model_size_bytes",
    "bundle_size_bytes",
    "static_llama_tag",
    "static_llama_sha256",
    "static_llama_contract_version",
    "runner_labels",
    "cpu_model",
    "cpu_flags",
    "git_sha",
    "output_dimension",
    "l2_normalize",
    "cold_iters",
]


def _float(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _int(raw: Any) -> int | None:
    value = _float(raw)
    return int(value) if value is not None else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(kind: str, value: Any) -> str:
    if value is None:
        return "—"
    if kind == "ms":
        return f"{float(value):.3f}"
    if kind == "cos":
        return f"{float(value):.6f}"
    if kind == "mb":
        return f"{float(value):.1f}"
    if kind == "fit":
        return "✔ fits" if value else "✖ exceeds"
    return str(value)


def _mb(size_bytes: int | None) -> float | None:
    return float(size_bytes) / (1024 * 1024) if size_bytes is not None else None


def _floats(rows: list[dict], impl: str, mode: str, column: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        if row.get("implementation") != impl or row.get("mode") != mode:
            continue
        value = _float(row.get(column))
        if value is not None:
            out.append(value)
    return out


def _normalized_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Metadata with pre-#150 field names (gguf_*) mapped to the current model_* names."""
    normalized = dict(metadata)
    normalized.setdefault("model_file", metadata.get("gguf_file"))
    normalized.setdefault("model_sha256", metadata.get("gguf_sha256"))
    normalized.setdefault("model_size_bytes", metadata.get("gguf_size_bytes"))
    return normalized


def build_latency_records(metadata: dict[str, Any], csv_rows: list[dict]) -> list[dict[str, Any]]:
    """One normalized record per (quant × scenario × warm/cold), latency-only."""
    metadata = _normalized_metadata(metadata)
    records: list[dict[str, Any]] = []
    for row in csv_rows:
        phase = PHASE_BY_MODE.get(row.get("mode", ""))
        if phase is None or row.get("implementation") != "ltembed":
            continue
        record: dict[str, Any] = {
            "quant": metadata.get("quant"),
            "scenario": row.get("scenario"),
            "phase": phase,
            "batch_size": _int(row.get("batch_size")),
            "text_profile": row.get("text_profile") or None,
            "threads": _int(row.get("threads")),
            "warmup_iters": _int(row.get("warmup_iters")),
            "timed_iters": _int(row.get("timed_iters")),
            # median == p50 (linear-interpolation percentiles, same as the Rust runner)
            "latency_ms": {key: _float(row.get(f"{key}_ms")) for key in LATENCY_KEYS},
        }
        record.update({key: metadata.get(key) for key in RECORD_METADATA_KEYS})
        records.append(record)
    records.sort(key=lambda r: (r.get("scenario") or "", PHASE_ORDER.get(r.get("phase"), 9)))
    return records


def _parity_summary(cosines: list[float], *, with_gate: bool = False) -> dict[str, Any] | None:
    if not cosines:
        return None
    summary: dict[str, Any] = {
        "mean_cosine": _mean(cosines),
        "min_cosine": min(cosines),
        "count": len(cosines),
    }
    if with_gate:
        summary["pass"] = summary["mean_cosine"] >= QUALITY_GATE
    return summary


def summarize_quant(metadata: dict[str, Any], csv_rows: list[dict]) -> dict[str, Any]:
    """Reduce one quant's CSV rows + metadata to a single comparison record."""
    metadata = _normalized_metadata(metadata)
    model_size = metadata.get("model_size_bytes")
    bundle_size = metadata.get("bundle_size_bytes", model_size)

    golden = _parity_summary(
        _floats(csv_rows, "ltembed", "golden_parity", "cosine_similarity_vs_pytorch"),
        with_gate=True,
    )
    dynamic = _parity_summary(
        _floats(csv_rows, "ltembed", "correctness", "cosine_similarity_vs_pytorch")
    )

    retrieval = None
    recall_at_3 = _mean(_floats(csv_rows, "ltembed", "retrieval_eval", "recall_at_3"))
    if recall_at_3 is not None:
        retrieval = {
            "recall_at_1": _mean(_floats(csv_rows, "ltembed", "retrieval_eval", "recall_at_1")),
            "recall_at_3": recall_at_3,
            "both_at_3": _mean(_floats(csv_rows, "ltembed", "retrieval_eval", "both_at_3")),
            "mrr_at_3": _mean(_floats(csv_rows, "ltembed", "retrieval_eval", "mrr_at_3")),
        }

    return {
        "quant": metadata.get("quant") or "?",
        "model_file": metadata.get("model_file"),
        "model_size_bytes": model_size,
        "bundle_size_bytes": bundle_size,
        "model_size_mb": _mb(model_size),
        "bundle_size_mb": _mb(bundle_size),
        "lambda_fit": (bundle_size <= LAMBDA_BUDGET_BYTES) if bundle_size is not None else None,
        "warm_mean_ms": _mean(_floats(csv_rows, "ltembed", "warm_latency", "mean_ms")),
        "cold_mean_ms": _mean(_floats(csv_rows, "ltembed", "cold_start", "mean_ms")),
        "golden_parity": golden,
        "dynamic_parity": dynamic,
        "retrieval": retrieval,
        "scenarios": metadata.get("scenarios"),
        "model_id": metadata.get("model_id"),
        "runner_labels": metadata.get("runner_labels"),
    }


def gate_cosine(quant: dict[str, Any]) -> float | None:
    """The cosine the quality gate is applied to: the immutable golden when present,
    else the dynamic FP32 reference parity."""
    for key in ("golden_parity", "dynamic_parity"):
        parity = quant.get(key)
        if parity and parity.get("mean_cosine") is not None:
            return parity["mean_cosine"]
    return None


def recommend(quants: list[dict[str, Any]]) -> tuple[str | None, str]:
    """Smallest bundle that passes the parity gate AND fits the Lambda budget."""
    budget_mb = _fmt("mb", _mb(LAMBDA_BUDGET_BYTES))
    scored = [q for q in quants if gate_cosine(q) is not None]
    gated = [q for q in scored if gate_cosine(q) >= QUALITY_GATE]
    fitting = [q for q in gated if q.get("lambda_fit")]
    if fitting:
        best = min(
            fitting,
            key=lambda q: (
                q.get("bundle_size_bytes") or float("inf"),
                q.get("warm_mean_ms") or float("inf"),
            ),
        )
        reason = (
            f"smallest bundle within the {budget_mb} MB Lambda budget whose mean cosine vs "
            f"FP32 stays ≥ {QUALITY_GATE:.2f} (mean_cosine={_fmt('cos', gate_cosine(best))}, "
            f"bundle={_fmt('mb', best['bundle_size_mb'])} MB)"
        )
        return best["quant"], reason
    if gated:
        names = ", ".join(f"`{q['quant']}`" for q in gated)
        return None, (
            f"{names} pass the {QUALITY_GATE:.2f} parity gate but exceed the "
            f"{budget_mb} MB Lambda bundle budget"
        )
    if scored:
        # No quant clears the gate: make no recommendation. Reporting the "least bad" quant
        # here would invite shipping a quant that fails the approved parity bar.
        best = max(scored, key=lambda q: gate_cosine(q))
        return None, (
            f"no quant met the {QUALITY_GATE:.2f} mean cosine gate "
            f"(best was `{best['quant']}` at mean_cosine={_fmt('cos', gate_cosine(best))})"
        )
    return None, "no cosine data available to rank quants"


def parse_expected_quants(raw: str | None) -> list[str]:
    """Accepts the workflow's quant list as either a JSON array or a CSV string."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in raw.split(",") if item.strip()]


def missing_quants(quants: list[dict[str, Any]], expected: list[str]) -> list[str]:
    present = {q.get("quant") for q in quants}
    return [name for name in expected if name not in present]


def latency_coverage_notes(
    quants: list[dict[str, Any]], records: list[dict[str, Any]]
) -> list[str]:
    """Per-quant notes when latency records don't cover every configured scenario/phase.

    Single-scenario or no-cold-start dispatches are legitimate smoke runs; the report
    labels them as partial instead of presenting them as full coverage.
    """
    notes: list[str] = []
    for quant in quants:
        name = quant.get("quant")
        expected = quant.get("scenarios") or []
        covered = {
            (r.get("scenario"), r.get("phase")) for r in records if r.get("quant") == name
        }
        missing_warm = [s for s in expected if (s, "warm") not in covered]
        if missing_warm:
            notes.append(f"`{name}`: no warm records for {', '.join(missing_warm)}")
        if expected and not any(phase == "cold" for _, phase in covered):
            notes.append(f"`{name}`: no cold-start records")
    return notes


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cells) + " |" for cells in rows)
    return "\n".join(lines) + "\n"


def _size_table(quants: list[dict[str, Any]]) -> str:
    rows = [
        [
            f"`{q['quant']}`",
            _fmt("mb", q.get("model_size_mb")),
            _fmt("mb", q.get("bundle_size_mb")),
            _fmt("fit", q.get("lambda_fit")),
        ]
        for q in quants
    ]
    budget_mb = _fmt("mb", _mb(LAMBDA_BUDGET_BYTES))
    return _table(["quant", "model MB", "bundle MB", f"≤ {budget_mb} MB (Lambda)"], rows)


def _parity_table(quants: list[dict[str, Any]]) -> str:
    def cell(q: dict[str, Any], key: str, field: str) -> str:
        parity = q.get(key)
        return _fmt("cos", parity.get(field)) if parity else "—"

    rows = [
        [
            f"`{q['quant']}`",
            cell(q, "golden_parity", "mean_cosine"),
            cell(q, "golden_parity", "min_cosine"),
            cell(q, "dynamic_parity", "mean_cosine"),
            cell(q, "dynamic_parity", "min_cosine"),
        ]
        for q in quants
    ]
    return _table(
        ["quant", "golden mean cos", "golden min cos", "dynamic mean cos", "dynamic min cos"],
        rows,
    )


def _retrieval_table(quants: list[dict[str, Any]]) -> str:
    def cell(q: dict[str, Any], field: str) -> str:
        retrieval = q.get("retrieval")
        return _fmt("cos", retrieval.get(field)) if retrieval else "—"

    rows = [
        [
            f"`{q['quant']}`",
            cell(q, "recall_at_1"),
            cell(q, "recall_at_3"),
            cell(q, "both_at_3"),
            cell(q, "mrr_at_3"),
        ]
        for q in quants
    ]
    return _table(["quant", "recall@1", "recall@3", "CN/EN both@3", "mrr@3"], rows)


def _latency_table(records: list[dict[str, Any]]) -> str:
    ordered = sorted(
        records,
        key=lambda r: (
            r.get("scenario") or "",
            PHASE_ORDER.get(r.get("phase"), 9),
            r.get("bundle_size_bytes") or 0,
            r.get("quant") or "",
        ),
    )
    rows = []
    for record in ordered:
        latency = record.get("latency_ms", {})
        rows.append(
            [
                f"`{record.get('quant')}`",
                str(record.get("scenario")),
                str(record.get("phase")),
                _fmt("ms", latency.get("mean")),
                _fmt("ms", latency.get("median")),
                _fmt("ms", latency.get("p95")),
                _fmt("ms", latency.get("p99")),
                _fmt("ms", latency.get("min")),
                _fmt("ms", latency.get("max")),
            ]
        )
    return _table(
        ["quant", "scenario", "phase", "mean", "p50", "p95", "p99", "min", "max"], rows
    )


def build_report(
    quants: list[dict[str, Any]],
    records: list[dict[str, Any]],
    recommendation: tuple[str | None, str],
    absent_quants: list[str] | None = None,
) -> str:
    if not quants:
        return "# Benchmark comparison\n\nNo quant results were found.\n"
    recommended, reason = recommendation
    model_id = next((q.get("model_id") for q in quants if q.get("model_id")), "?")
    runner = next((q.get("runner_labels") for q in quants if q.get("runner_labels")), "?")
    budget_mb = _fmt("mb", _mb(LAMBDA_BUDGET_BYTES))
    limit_mb = _fmt("mb", _mb(LAMBDA_PACKAGE_LIMIT_BYTES))
    allowance_mb = _fmt("mb", _mb(LAMBDA_BINARY_ALLOWANCE_BYTES))

    parts = [
        "# Benchmark comparison — GGUF quants vs PyTorch FP32",
        "",
        f"Model: `{model_id}` · Runner: `{runner}` · Quality gate: mean cosine ≥ "
        f"{QUALITY_GATE:.2f} · Lambda bundle budget: {budget_mb} MB "
        f"({limit_mb} MB package limit − {allowance_mb} MB binary allowance)",
        "",
    ]
    if absent_quants:
        names = ", ".join(f"`{name}`" for name in absent_quants)
        parts.append(
            f"> ⚠️ **INCOMPLETE RUN** — no results for {names} (matrix job failed or "
            "was skipped). No recommendation is made from a partial matrix."
        )
        parts.append("")
    coverage = latency_coverage_notes(quants, records)
    if coverage:
        parts.append("> ⚠️ **Partial latency coverage** — " + "; ".join(coverage) + ".")
        parts.append("")
    parts += [
        "## Size & Lambda fit",
        "",
        _size_table(quants),
        "## Parity vs FP32",
        "",
        _parity_table(quants),
        "## Retrieval quality",
        "",
        _retrieval_table(quants),
        "## Latency (ms, per scenario)",
        "",
        _latency_table(records),
    ]
    oversized = [q["quant"] for q in quants if q.get("lambda_fit") is False]
    if recommended:
        parts.append(f"**Recommended quant: `{recommended}`** — {reason}.")
    else:
        parts.append(f"**No recommendation:** {reason}.")
    if oversized:
        names = ", ".join(f"`{name}`" for name in oversized)
        parts.append("")
        parts.append(f"_Over budget: {names} — bundle exceeds the {budget_mb} MB budget "
                     f"({limit_mb} MB Lambda package limit − {allowance_mb} MB binary "
                     "allowance), excluded from recommendation regardless of parity._")
    parts.append("")
    parts.append(
        "_Golden cosine compares each quant against the immutable PyTorch/F32 fixtures "
        "(`tests/fixtures/test_fixtures.json`); dynamic cosine against the workflow's fresh "
        "FP32 reference on the CN/EN sentences. `batch/*` scenarios time one whole-batch "
        "embed call per iteration; median == p50._"
    )
    parts.append("")
    return "\n".join(parts)


def collect_results(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(quants, records) aggregated from every metadata.json + sibling CSV under input_dir."""
    quants: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for metadata_path in sorted(input_dir.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        csv_path = metadata_path.parent / "benchmark-report.csv"
        csv_rows: list[dict] = []
        if csv_path.is_file():
            with csv_path.open(newline="", encoding="utf-8") as fh:
                csv_rows = list(csv.DictReader(fh))
        quants.append(summarize_quant(metadata, csv_rows))
        records.extend(build_latency_records(metadata, csv_rows))
    quants.sort(key=lambda q: (q.get("bundle_size_bytes") or 0, q.get("quant") or ""))
    return quants, records


def build_results_payload(
    quants: list[dict[str, Any]],
    records: list[dict[str, Any]],
    recommendation: tuple[str | None, str],
    expected: list[str] | None = None,
    absent_quants: list[str] | None = None,
) -> dict[str, Any]:
    recommended, reason = recommendation
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model_id": next((q.get("model_id") for q in quants if q.get("model_id")), None),
        "quality_gate": QUALITY_GATE,
        "lambda_package_limit_bytes": LAMBDA_PACKAGE_LIMIT_BYTES,
        "lambda_binary_allowance_bytes": LAMBDA_BINARY_ALLOWANCE_BYTES,
        "lambda_budget_bytes": LAMBDA_BUDGET_BYTES,
        "expected_quants": expected or [],
        "missing_quants": absent_quants or [],
        "complete": not absent_quants,
        "records": records,
        "quants": quants,
        "recommendation": {"quant": recommended, "reason": reason},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("."))
    parser.add_argument(
        "--expected-quants",
        default="",
        help=(
            "Quant list this run was dispatched with (JSON array or CSV). When any are "
            "missing from the artifacts the report is marked incomplete and no "
            "recommendation is made."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    quants, records = collect_results(args.input_dir)
    expected = parse_expected_quants(args.expected_quants)
    absent = missing_quants(quants, expected)
    if absent:
        recommendation = (
            None,
            "incomplete quant matrix — no results for "
            + ", ".join(f"`{name}`" for name in absent),
        )
    else:
        recommendation = recommend(quants)
    report = build_report(quants, records, recommendation, absent_quants=absent)
    payload = build_results_payload(
        quants, records, recommendation, expected=expected, absent_quants=absent
    )

    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(report)

    print(f"rendered {len(quants)} quant result(s) to {output_dir / 'report.md'}")
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
