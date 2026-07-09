#!/usr/bin/env python3
"""render_benchmark_report.py — aggregate per-quant benchmark artifacts into one report.

    render_benchmark_report.py <input_dir> [output_dir]

Recursively finds every quant's ``metadata.json`` under ``<input_dir>`` (as produced by the
benchmark matrix jobs) and reads the sibling ``benchmark-report.csv``. It renders a side-by-side
Markdown comparison — one row per GGUF quant — covering llama.cpp warm/cold latency, cosine
similarity of each quant's embeddings against the PyTorch FP32 reference, retrieval quality, GGUF
size, and speedup over PyTorch. A recommended quant is chosen (smallest GGUF that keeps the
mean cosine at or above the quality gate).

Writes:

    <output_dir>/report.md      the Markdown report
    <output_dir>/results.json   the combined per-quant aggregates

If ``$GITHUB_STEP_SUMMARY`` is set, the report is also appended there so it renders in the Actions
run UI. ``<output_dir>`` defaults to the current directory.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

# Mean cosine (vs PyTorch FP32) a quant must keep to be considered "quality-preserving".
QUALITY_GATE = 0.98

# (json_key, column header, kind) — kind drives formatting in _fmt.
COLUMNS = [
    ("quant", "quant", "str"),
    ("size_mb", "GGUF MB", "mb"),
    ("warm_ms", "warm ms", "ms"),
    ("cold_ms", "cold ms", "ms"),
    ("mean_cosine", "mean cos vs FP32", "cos"),
    ("min_cosine", "min cos vs FP32", "cos"),
    ("both_at_3", "CN/EN both@3", "cos"),
    ("recall_at_3", "recall@3", "cos"),
    ("mrr_at_3", "mrr@3", "cos"),
]


def _fmt(kind: str, value: Any) -> str:
    if value is None:
        return "—"
    if kind == "ms":
        return f"{float(value):.3f}"
    if kind == "cos":
        return f"{float(value):.6f}"
    if kind == "mb":
        return f"{float(value):.1f}"
    if kind == "x":
        return f"{float(value):.2f}×"
    return str(value)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _floats(rows: list[dict], impl: str, mode: str, column: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        if row.get("implementation") != impl or row.get("mode") != mode:
            continue
        raw = row.get(column, "")
        if raw == "" or raw is None:
            continue
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def summarize_quant(metadata: dict[str, Any], csv_rows: list[dict]) -> dict[str, Any]:
    """Reduce one quant's CSV rows + metadata to a single comparison record."""
    quant = metadata.get("quant") or "?"
    size_bytes = metadata.get("gguf_size_bytes")
    size_mb = (float(size_bytes) / (1024 * 1024)) if size_bytes else None

    warm = _mean(_floats(csv_rows, "ltembed", "warm_latency", "mean_ms"))
    cold = _mean(_floats(csv_rows, "ltembed", "cold_start", "mean_ms"))
    cosines = _floats(csv_rows, "ltembed", "correctness", "cosine_similarity_vs_pytorch")
    mean_cosine = _mean(cosines)
    min_cosine = min(cosines) if cosines else None
    recall_at_1 = _mean(_floats(csv_rows, "ltembed", "retrieval_eval", "recall_at_1"))
    recall_at_3 = _mean(_floats(csv_rows, "ltembed", "retrieval_eval", "recall_at_3"))
    both_at_3 = _mean(_floats(csv_rows, "ltembed", "retrieval_eval", "both_at_3"))
    mrr_at_3 = _mean(_floats(csv_rows, "ltembed", "retrieval_eval", "mrr_at_3"))

    return {
        "quant": quant,
        "gguf_file": metadata.get("gguf_file"),
        "gguf_size_bytes": size_bytes,
        "size_mb": size_mb,
        "warm_ms": warm,
        "cold_ms": cold,
        "mean_cosine": mean_cosine,
        "min_cosine": min_cosine,
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "both_at_3": both_at_3,
        "mrr_at_3": mrr_at_3,
        "model_id": metadata.get("model_id"),
        "runner_labels": metadata.get("runner_labels"),
    }


def recommend(results: list[dict[str, Any]]) -> tuple[str | None, str]:
    """Pick the smallest GGUF that keeps mean cosine >= the quality gate."""
    gated = [
        r for r in results if r.get("mean_cosine") is not None and r["mean_cosine"] >= QUALITY_GATE
    ]
    if gated:
        best = min(
            gated,
            key=lambda r: (r.get("gguf_size_bytes") or float("inf"), r.get("warm_ms") or float("inf")),
        )
        reason = (
            f"smallest GGUF whose mean cosine vs FP32 stays ≥ {QUALITY_GATE:.2f} "
            f"(mean_cosine={_fmt('cos', best['mean_cosine'])}, size={_fmt('mb', best['size_mb'])} MB)"
        )
        return best["quant"], reason
    scored = [r for r in results if r.get("mean_cosine") is not None]
    if scored:
        best = max(scored, key=lambda r: r["mean_cosine"])
        reason = (
            f"no quant met the {QUALITY_GATE:.2f} mean cosine gate; highest mean cosine vs FP32 "
            f"(mean_cosine={_fmt('cos', best['mean_cosine'])})"
        )
        return best["quant"], reason
    return None, "no cosine data available to rank quants"


def render_table(results: list[dict[str, Any]]) -> str:
    headers = [header for _, header, _ in COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in results:
        cells = [_fmt(kind, row.get(key)) for key, _, kind in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_report(results: list[dict[str, Any]]) -> str:
    if not results:
        return "# Benchmark comparison\n\nNo quant results were found.\n"
    recommended, reason = recommend(results)
    model_id = next((r.get("model_id") for r in results if r.get("model_id")), "?")
    runner = next((r.get("runner_labels") for r in results if r.get("runner_labels")), "?")

    parts = [
        "# Benchmark comparison — GGUF quants vs PyTorch FP32",
        "",
        f"Model: `{model_id}` · Runner: `{runner}` · Quality gate: mean cosine ≥ {QUALITY_GATE:.2f}",
        "",
        render_table(results),
        "",
    ]
    if recommended:
        parts.append(f"**Recommended quant: `{recommended}`** — {reason}.")
    else:
        parts.append(f"**No recommendation:** {reason}.")
    parts.append("")
    parts.append(
        "_Cosine columns compare each quant's embeddings against the full-precision PyTorch "
        "(FP32) model on the same CN/EN sentences; higher is closer to the reference._"
    )
    parts.append("")
    return "\n".join(parts)


def collect_results(input_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for metadata_path in sorted(input_dir.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        csv_path = metadata_path.parent / "benchmark-report.csv"
        csv_rows: list[dict] = []
        if csv_path.is_file():
            with csv_path.open(newline="", encoding="utf-8") as fh:
                csv_rows = list(csv.DictReader(fh))
        results.append(summarize_quant(metadata, csv_rows))
    results.sort(key=lambda r: (r.get("gguf_size_bytes") or 0, r.get("quant") or ""))
    return results


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        sys.exit("Usage: render_benchmark_report.py <input_dir> [output_dir]")
    input_dir = Path(argv[0])
    output_dir = Path(argv[1]) if len(argv) > 1 else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = collect_results(input_dir)
    report = build_report(results)

    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(report)

    print(f"rendered {len(results)} quant result(s) to {output_dir / 'report.md'}")
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
