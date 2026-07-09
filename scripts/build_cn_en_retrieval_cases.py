#!/usr/bin/env python3
"""Generate a CN/EN cross-lingual retrieval-eval case from a translation-pair CSV.

The input CSV (``tests/CN_EN_Data.csv``) has two columns, ``Chinese`` and
``English``, one translation pair per row. We deterministically sample a subset of
pairs and emit a retrieval-eval JSON in the schema consumed by the benchmark
runners (``{"cases": [{name, documents, queries}]}``).

For each sampled pair ``i`` the corpus holds *both* languages as documents
(``pair_{i}_zh`` / ``pair_{i}_en``), and we emit *both* a Chinese and an English
query. Each query's relevant set is the pair's two documents, so a correct
multilingual model returns the self-language document and its cross-language
translation together. Sampling uses a fixed stride (no RNG), so every job that
runs this generator selects byte-identical pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "tests" / "CN_EN_Data.csv"
DEFAULT_NAME = "cn-en-crosslingual-v1"


def load_pairs(csv_path: Path) -> list[tuple[str, str]]:
    """Read (chinese, english) pairs, dropping rows with an empty side."""
    pairs: list[tuple[str, str]] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "Chinese" not in reader.fieldnames or "English" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must have 'Chinese' and 'English' columns")
        for record in reader:
            zh = (record.get("Chinese") or "").strip()
            en = (record.get("English") or "").strip()
            if zh and en:
                pairs.append((zh, en))
    if not pairs:
        raise ValueError(f"no usable translation pairs in {csv_path}")
    return pairs


def sample_pairs(pairs: list[tuple[str, str]], num_pairs: int) -> list[tuple[str, str]]:
    """Deterministically take ~``num_pairs`` pairs evenly spread across the file."""
    if num_pairs <= 0 or num_pairs >= len(pairs):
        return pairs
    stride = max(1, len(pairs) // num_pairs)
    return pairs[::stride][:num_pairs]


def build_case(pairs: list[tuple[str, str]], name: str) -> dict[str, Any]:
    documents: list[dict[str, str]] = []
    queries: list[dict[str, Any]] = []
    for index, (zh, en) in enumerate(pairs):
        zh_id = f"pair_{index}_zh"
        en_id = f"pair_{index}_en"
        relevant = [zh_id, en_id]
        documents.append({"id": zh_id, "text": zh})
        documents.append({"id": en_id, "text": en})
        queries.append({"id": f"q_{index}_zh", "text": zh, "relevant_document_ids": relevant})
        queries.append({"id": f"q_{index}_en", "text": en, "relevant_document_ids": relevant})
    return {"name": name, "documents": documents, "queries": queries}


def build_payload(csv_path: Path, num_pairs: int, name: str) -> dict[str, Any]:
    pairs = sample_pairs(load_pairs(csv_path), num_pairs)
    return {"cases": [build_case(pairs, name)]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=500)
    parser.add_argument("--name", default=DEFAULT_NAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args.csv, args.num_pairs, args.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    case = payload["cases"][0]
    print(
        f"wrote {args.output}: {len(case['documents'])} documents, "
        f"{len(case['queries'])} queries",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
