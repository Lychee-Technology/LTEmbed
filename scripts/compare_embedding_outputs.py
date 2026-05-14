#!/usr/bin/env python3
"""
Compare raw embedding outputs from LTEmbed and PyTorch correctness payloads.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ltembed-json", required=True)
    parser.add_argument("--pytorch-json", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-text", required=True)
    parser.add_argument("--first-values", type=int, default=8)
    return parser.parse_args(argv)


def load_payload(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_embeddings(payload: dict, scenario: str) -> list[list[float]]:
    for result in payload.get("results", []):
        if result.get("scenario") == scenario:
            embeddings = result.get("embeddings", [])
            if not embeddings:
                raise ValueError(f"scenario {scenario!r} has no embeddings")
            return embeddings
    raise ValueError(f"scenario {scenario!r} not found in payload")


def cosine_similarity(lhs: list[float], rhs: list[float]) -> float:
    if len(lhs) != len(rhs):
        raise ValueError("embedding lengths differ")
    dot = sum(a * b for a, b in zip(lhs, rhs))
    lhs_norm = math.sqrt(sum(a * a for a in lhs))
    rhs_norm = math.sqrt(sum(b * b for b in rhs))
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        raise ValueError("zero vector encountered")
    return dot / (lhs_norm * rhs_norm)


def l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def build_summary(
    *,
    scenario: str,
    ltembed_embedding: list[float],
    pytorch_embedding: list[float],
    first_values: int = 8,
) -> dict[str, object]:
    if len(ltembed_embedding) != len(pytorch_embedding):
        raise ValueError("embedding lengths differ")

    diffs = [abs(lhs - rhs) for lhs, rhs in zip(ltembed_embedding, pytorch_embedding)]
    rmse = math.sqrt(
        sum((lhs - rhs) * (lhs - rhs) for lhs, rhs in zip(ltembed_embedding, pytorch_embedding))
        / len(ltembed_embedding)
    )
    preview = first_values if first_values > 0 else len(ltembed_embedding)
    return {
        "scenario": scenario,
        "dimensions": len(ltembed_embedding),
        "cosine_similarity": cosine_similarity(ltembed_embedding, pytorch_embedding),
        "ltembed_l2_norm": l2_norm(ltembed_embedding),
        "pytorch_l2_norm": l2_norm(pytorch_embedding),
        "max_abs_error": max(diffs),
        "mean_abs_error": sum(diffs) / len(diffs),
        "rmse": rmse,
        "ltembed_first_values": ltembed_embedding[:preview],
        "pytorch_first_values": pytorch_embedding[:preview],
        "abs_error_first_values": diffs[:preview],
    }


def build_payload_summary(
    *,
    scenario: str,
    ltembed_embeddings: list[list[float]],
    pytorch_embeddings: list[list[float]],
    first_values: int = 8,
) -> dict[str, object]:
    if len(ltembed_embeddings) != len(pytorch_embeddings):
        raise ValueError("embedding counts differ")
    if not ltembed_embeddings:
        raise ValueError("no embeddings to compare")

    pair_summaries = [
        build_summary(
            scenario=scenario,
            ltembed_embedding=ltembed_embedding,
            pytorch_embedding=pytorch_embedding,
            first_values=first_values,
        )
        for ltembed_embedding, pytorch_embedding in zip(ltembed_embeddings, pytorch_embeddings)
    ]
    first = pair_summaries[0]
    return {
        "scenario": scenario,
        "num_embeddings": len(pair_summaries),
        "dimensions": first["dimensions"],
        "cosine_similarity": sum(item["cosine_similarity"] for item in pair_summaries)
        / len(pair_summaries),
        "cosine_similarity_min": min(item["cosine_similarity"] for item in pair_summaries),
        "cosine_similarity_max": max(item["cosine_similarity"] for item in pair_summaries),
        "ltembed_l2_norm": sum(item["ltembed_l2_norm"] for item in pair_summaries)
        / len(pair_summaries),
        "pytorch_l2_norm": sum(item["pytorch_l2_norm"] for item in pair_summaries)
        / len(pair_summaries),
        "max_abs_error": max(item["max_abs_error"] for item in pair_summaries),
        "mean_abs_error": sum(item["mean_abs_error"] for item in pair_summaries)
        / len(pair_summaries),
        "rmse": sum(item["rmse"] for item in pair_summaries) / len(pair_summaries),
        "ltembed_first_values": first["ltembed_first_values"],
        "pytorch_first_values": first["pytorch_first_values"],
        "abs_error_first_values": first["abs_error_first_values"],
    }


def render_text_summary(summary: dict[str, object]) -> str:
    lines = [
        f"scenario: {summary['scenario']}",
        f"num_embeddings: {summary['num_embeddings']}",
        f"dimensions: {summary['dimensions']}",
        f"cosine_similarity: {summary['cosine_similarity']:.6f}",
        f"cosine_similarity_min: {summary['cosine_similarity_min']:.6f}",
        f"cosine_similarity_max: {summary['cosine_similarity_max']:.6f}",
        f"ltembed_l2_norm: {summary['ltembed_l2_norm']:.6f}",
        f"pytorch_l2_norm: {summary['pytorch_l2_norm']:.6f}",
        f"max_abs_error: {summary['max_abs_error']:.6f}",
        f"mean_abs_error: {summary['mean_abs_error']:.6f}",
        f"rmse: {summary['rmse']:.6f}",
        f"ltembed_first_values: {summary['ltembed_first_values']}",
        f"pytorch_first_values: {summary['pytorch_first_values']}",
        f"abs_error_first_values: {summary['abs_error_first_values']}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ltembed_payload = load_payload(args.ltembed_json)
    pytorch_payload = load_payload(args.pytorch_json)
    summary = build_payload_summary(
        scenario=args.scenario,
        ltembed_embeddings=select_embeddings(ltembed_payload, args.scenario),
        pytorch_embeddings=select_embeddings(pytorch_payload, args.scenario),
        first_values=args.first_values,
    )

    output_json = Path(args.output_json)
    output_text = Path(args.output_text)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_text.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    output_text.write_text(render_text_summary(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
