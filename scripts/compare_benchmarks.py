#!/usr/bin/env python3
"""
compare_benchmarks.py — compare two or more benchmark CSV reports.

Usage:
    python scripts/compare_benchmarks.py baseline.csv candidate.csv [candidate2.csv ...]
    python scripts/compare_benchmarks.py --label main:main.csv neon:neon.csv

Each CSV is labelled by its filename stem unless overridden with --label name:file.
Rows are matched on: model_id, implementation, scenario, mode, batch_size,
text_profile, threads, warmup_iters, timed_iters.
Only warm_latency rows are compared by default (override with --mode).
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

KEY_COLS = [
    "model_id",
    "implementation",
    "scenario",
    "mode",
    "batch_size",
    "text_profile",
    "threads",
    "warmup_iters",
    "timed_iters",
]
METRIC_COLS = ["mean_ms", "median_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"]


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def row_key(row: dict) -> tuple:
    return tuple(row.get(c, "") for c in KEY_COLS)


def index_rows(rows: list[dict]) -> dict[tuple, dict]:
    idx = {}
    for row in rows:
        k = row_key(row)
        if k in idx:
            # last write wins (shouldn't happen in well-formed CSVs)
            pass
        idx[k] = row
    return idx


def fmt_delta(delta: float, pct: float) -> str:
    sign = "+" if delta >= 0 else ""
    flag = ""
    if pct > 2.0:
        flag = " ❌"
    elif pct < -2.0:
        flag = " ✅"
    else:
        flag = " ≈"
    return f"{sign}{delta:.2f}ms ({sign}{pct:.1f}%){flag}"


def row_label(row: dict, varying_cols: list[str]) -> str:
    """Build a display label from scenario + any other varying key columns."""
    parts = [row.get("scenario", "?")]
    for col in varying_cols:
        if col != "scenario":
            parts.append(f"{col}={row.get(col, '?')}")
    return "  ".join(parts)


def compare(
    labels: list[str],
    indexes: list[dict[tuple, dict]],
    mode_filter: str | None,
    metric: str,
    impl_filter: str | None = None,
) -> None:
    baseline_label = labels[0]
    baseline_idx = indexes[0]

    # Collect all keys present in baseline (optionally filtered by mode)
    keys = [
        k
        for k in baseline_idx
        if mode_filter is None or baseline_idx[k].get("mode") == mode_filter
    ]
    if impl_filter:
        impl_col = KEY_COLS.index("implementation")
        keys = [k for k in keys if k[impl_col] == impl_filter]
    keys.sort(key=lambda k: (k[KEY_COLS.index("scenario")], k))

    if not keys:
        filter_desc = f" with mode={mode_filter!r}" if mode_filter else ""
        print(f"No rows found in baseline{filter_desc}.")
        return

    # Detect which key columns vary across rows (to build informative labels)
    varying_cols = []
    for i, col in enumerate(KEY_COLS):
        if col == "mode":
            continue  # already filtered
        vals = {k[i] for k in keys}
        if len(vals) > 1:
            varying_cols.append(col)
    if not varying_cols:
        varying_cols = ["scenario"]

    # Determine label column width
    baseline_rows = [baseline_idx[k] for k in keys]
    col_w = max(len(row_label(r, varying_cols)) for r in baseline_rows) + 2
    col_w = max(col_w, 30)

    metric_w = 28
    header = f"{'Row':<{col_w}}"
    header += f"  {baseline_label:>{metric_w}}"
    for lbl in labels[1:]:
        header += f"  {lbl:>{metric_w}}"
    print(header)
    print("-" * len(header))

    missing: list[tuple[str, tuple]] = []

    for k in keys:
        base_row = baseline_idx[k]
        try:
            base_val = float(base_row[metric])
        except (ValueError, KeyError):
            continue

        label = row_label(base_row, varying_cols)
        line = f"{label:<{col_w}}  {base_val:>{metric_w-3}.2f} ms "

        for lbl, idx in zip(labels[1:], indexes[1:]):
            if k not in idx:
                missing.append((lbl, k))
                line += f"  {'(missing)':>{metric_w}}"
                continue
            try:
                cand_val = float(idx[k][metric])
            except (ValueError, KeyError):
                line += f"  {'(n/a)':>{metric_w}}"
                continue
            delta = cand_val - base_val
            pct = (delta / base_val) * 100 if base_val != 0 else 0.0
            line += f"  {fmt_delta(delta, pct):>{metric_w}}"

        print(line)

    if missing:
        print()
        print("Missing in candidate:")
        for lbl, k in missing:
            print(f"  [{lbl}] scenario={k[KEY_COLS.index('scenario')]!r}")

    # Summary: geometric mean of |pct| per candidate
    print()
    print("Summary (geometric mean |Δ%| vs baseline):")
    for lbl, idx in zip(labels[1:], indexes[1:]):
        import math

        pcts = []
        for k in keys:
            if k not in idx or k not in baseline_idx:
                continue
            try:
                bv = float(baseline_idx[k][metric])
                cv = float(idx[k][metric])
            except (ValueError, KeyError):
                continue
            if bv != 0:
                pcts.append((cv - bv) / bv * 100)
        if pcts:
            mean_pct = sum(pcts) / len(pcts)
            geo = math.exp(sum(math.log(abs(p) + 1e-9) for p in pcts) / len(pcts))
            sign = "+" if mean_pct >= 0 else ""
            print(f"  {lbl}: arithmetic mean {sign}{mean_pct:.2f}%  geo-mean |Δ%| {geo:.2f}%")


def parse_label_file(token: str) -> tuple[str, str]:
    """Parse 'label:path' or just 'path' (label defaults to stem)."""
    if ":" in token:
        # handle Windows paths like C:\foo — colon after single char is a drive letter
        parts = token.split(":", 1)
        if len(parts[0]) == 1 and parts[0].isalpha():
            # looks like a Windows drive letter, treat whole thing as path
            return Path(token).stem, token
        return parts[0], parts[1]
    return Path(token).stem, token


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two or more benchmark CSV reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="+",
        metavar="[label:]file.csv",
        help="CSV files to compare. First is the baseline.",
    )
    parser.add_argument(
        "--label",
        action="append",
        metavar="name:file",
        dest="label_files",
        help="Explicit label:file pairs (alternative to positional args).",
    )
    parser.add_argument(
        "--mode",
        default="warm_latency",
        help="Filter rows by 'mode' column (default: warm_latency). Pass '' to include all.",
    )
    parser.add_argument(
        "--metric",
        default="mean_ms",
        choices=METRIC_COLS,
        help="Metric column to compare (default: mean_ms).",
    )
    parser.add_argument(
        "--impl",
        default=None,
        metavar="NAME",
        help="Filter rows by implementation name (e.g. ltembed, pytorch).",
    )
    args = parser.parse_args()

    tokens = args.files
    if args.label_files:
        tokens = args.label_files + tokens

    if len(tokens) < 2:
        parser.error("Need at least two CSV files to compare.")

    labels: list[str] = []
    indexes: list[dict] = []

    for token in tokens:
        label, path_str = parse_label_file(token)
        path = Path(path_str)
        if not path.exists():
            sys.exit(f"File not found: {path}")
        rows = load_csv(path)
        labels.append(label)
        indexes.append(index_rows(rows))
        print(f"Loaded {label!r}: {len(rows)} rows from {path}")

    print()
    mode_filter = args.mode if args.mode else None
    compare(labels, indexes, mode_filter, args.metric, impl_filter=args.impl)


if __name__ == "__main__":
    main()
