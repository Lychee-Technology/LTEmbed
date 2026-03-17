#!/usr/bin/env python3
"""
Profile projection-heavy GEMM paths on Linux ARM64 with perf.

The script builds `benchmark_ltembed`, records a `single/long` warm run by
default, exports perf reports, and annotates the hottest `matrixmultiply::*`
symbols that appear in the report.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = "single/long"
DEFAULT_OUTPUT_ROOT = ROOT / "perf-results"
DEFAULT_BINARY_PATH = ROOT / "target" / "release" / "benchmark_ltembed"
DEFAULT_MODEL_DIR = ROOT / "assets"


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def benchmark_command(args: argparse.Namespace, binary_path: Path) -> list[str]:
    return [
        str(binary_path),
        "--mode",
        "warm",
        "--scenario",
        str(args.scenario),
        "--model-dir",
        str(args.model_dir),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--threads",
        str(args.threads),
    ]


def perf_record_command(
    args: argparse.Namespace,
    binary_path: Path,
    perf_data_path: Path,
) -> list[str]:
    return [
        "perf",
        "record",
        "-F",
        str(args.perf_freq),
        "-e",
        str(args.perf_event),
        "-g",
        "--call-graph",
        str(args.call_graph),
        "--output",
        str(perf_data_path),
        "--",
        *benchmark_command(args, binary_path),
    ]


def perf_report_command(perf_data_path: Path, children: bool) -> list[str]:
    command = [
        "perf",
        "report",
        "--stdio",
        "--input",
        str(perf_data_path),
        "--sort",
        "overhead,comm,dso,symbol",
    ]
    if children:
        command.append("--children")
    else:
        command.append("--no-children")
    return command


def perf_annotate_command(perf_data_path: Path, symbol: str) -> list[str]:
    return [
        "perf",
        "annotate",
        "--stdio",
        "--input",
        str(perf_data_path),
        "--symbol",
        symbol,
    ]


def extract_matrixmultiply_symbols(report_text: str, limit: int) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for raw_line in report_text.splitlines():
        if "matrixmultiply::" not in raw_line or "[.]" not in raw_line:
            continue
        remainder = raw_line.split("[.]", 1)[1]
        match = re.search(r"(matrixmultiply::\S+)", remainder)
        if match is None:
            continue
        symbol = match.group(1)
        if not symbol.startswith("matrixmultiply::"):
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
        if len(symbols) >= limit:
            break
    return symbols


def sanitize_symbol_name(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", symbol).strip("_")


def format_command_failure(command: list[str], stderr: str) -> str:
    message = f"command failed: {shlex.join(command)}"
    details = stderr.strip()
    if details:
        message += f"\n{details}"
    return message


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record and export Linux perf data for projection-heavy LTEmbed GEMM paths."
    )
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--perf-freq", type=int, default=999)
    parser.add_argument("--perf-event", default="cpu-clock")
    parser.add_argument("--call-graph", default="dwarf")
    parser.add_argument("--top-symbols", type=int, default=3)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY_PATH)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def ensure_environment(args: argparse.Namespace) -> None:
    if platform.system().lower() != "linux":
        raise SystemExit("this script only supports Linux perf collection")
    if shutil.which("perf") is None:
        raise SystemExit("perf not found in PATH")
    if not args.skip_build and shutil.which("cargo") is None:
        raise SystemExit("cargo not found in PATH")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    scenario_slug = str(args.scenario).replace("/", "_")
    return DEFAULT_OUTPUT_ROOT / f"{timestamp_slug()}-{scenario_slug}"


def run_command(
    command: list[str],
    *,
    cwd: Path = ROOT,
    stdout_path: Path | None = None,
) -> str:
    try:
        if stdout_path is None:
            completed = subprocess.run(
                command,
                check=True,
                cwd=cwd,
                capture_output=True,
                text=True,
            )
            return completed.stdout

        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                check=True,
                cwd=cwd,
                stdout=handle,
                stderr=subprocess.PIPE,
                text=True,
            )
        return completed.stderr
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or exc.stdout or ""
        raise RuntimeError(format_command_failure(command, stderr)) from exc


def write_metadata(output_dir: Path, metadata: dict[str, object]) -> None:
    metadata_path = output_dir / "run-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        ensure_environment(args)

        output_dir = resolve_output_dir(args)
        output_dir.mkdir(parents=True, exist_ok=True)

        binary_path = args.binary
        perf_data_path = output_dir / "perf.data"
        report_path = output_dir / "perf-report.txt"
        report_children_path = output_dir / "perf-report-children.txt"
        symbol_report_path = output_dir / "perf-report-symbols.txt"
        annotate_dir = output_dir / "annotate"
        annotate_dir.mkdir(parents=True, exist_ok=True)

        commands: dict[str, list[str]] = {}

        if not args.skip_build:
            build_command = ["cargo", "build", "--release", "--bin", "benchmark_ltembed"]
            commands["build"] = build_command
            run_command(build_command)

        record_command = perf_record_command(args, binary_path, perf_data_path)
        commands["record"] = record_command
        run_command(record_command)

        report_command = perf_report_command(perf_data_path, children=False)
        commands["report"] = report_command
        run_command(report_command, stdout_path=report_path)

        report_children_command = perf_report_command(perf_data_path, children=True)
        commands["report_children"] = report_children_command
        run_command(report_children_command, stdout_path=report_children_path)

        symbol_report_command = [
            "perf",
            "report",
            "--stdio",
            "--input",
            str(perf_data_path),
            "--no-children",
            "--sort",
            "symbol",
        ]
        commands["report_symbols"] = symbol_report_command
        run_command(symbol_report_command, stdout_path=symbol_report_path)

        symbol_report_text = symbol_report_path.read_text(encoding="utf-8")
        symbols = extract_matrixmultiply_symbols(symbol_report_text, args.top_symbols)

        annotate_outputs: list[dict[str, str]] = []
        for index, symbol in enumerate(symbols, start=1):
            annotate_path = annotate_dir / f"{index:02d}-{sanitize_symbol_name(symbol)}.txt"
            annotate_command = perf_annotate_command(perf_data_path, symbol)
            commands[f"annotate_{index}"] = annotate_command
            run_command(annotate_command, stdout_path=annotate_path)
            annotate_outputs.append({"symbol": symbol, "path": str(annotate_path)})

        metadata = {
            "scenario": args.scenario,
            "model_dir": str(args.model_dir),
            "binary": str(binary_path),
            "output_dir": str(output_dir),
            "commands": commands,
            "artifacts": {
                "perf_data": str(perf_data_path),
                "report": str(report_path),
                "report_children": str(report_children_path),
                "report_symbols": str(symbol_report_path),
                "annotate": annotate_outputs,
            },
        }
        write_metadata(output_dir, metadata)

        print(f"perf capture complete: {output_dir}")
        print(f"report: {report_path}")
        print(f"report with children: {report_children_path}")
        print(f"symbol report: {symbol_report_path}")
        if annotate_outputs:
            print("annotate files:")
            for item in annotate_outputs:
                print(f"  {item['symbol']}: {item['path']}")
        else:
            print("annotate files: none (no matrixmultiply symbols found in perf report)")
        print("share back the report files or paste the top matrixmultiply symbol sections.")
        return 0
    except RuntimeError as err:
        print(err, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
