#!/usr/bin/env python3
"""
Compare q4f16 ONNX embeddings against the HuggingFace PyTorch reference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def load_bench_pytorch_module():
    spec = importlib.util.spec_from_file_location("bench_pytorch", SCRIPT_DIR / "bench_pytorch.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


bench_pytorch = load_bench_pytorch_module()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--onnx-model-path", required=True)
    parser.add_argument("--scenario", default="single/medium")
    parser.add_argument("--output-dimension", type=int, default=bench_pytorch.RAW_DIM)
    parser.add_argument("--l2-normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-json")
    parser.add_argument("--output-text")
    parser.add_argument("--first-values", type=int, default=8)
    return parser.parse_args(argv)


def load_onnx_session(onnx_model_path: str | Path):
    import onnxruntime as ort

    return ort.InferenceSession(
        str(onnx_model_path),
        providers=["CPUExecutionProvider"],
    )


def pool_last_token_numpy(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    last_token_index = attention_mask.sum(axis=1) - 1
    batch_index = np.arange(last_hidden_state.shape[0])
    return last_hidden_state[batch_index, last_token_index]


def build_onnx_input_feed(session, encoded: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    input_names = {input_meta.name for input_meta in session.get_inputs()}
    return {name: np.asarray(value) for name, value in encoded.items() if name in input_names}


def resolve_output_name(session) -> str:
    output_names = [output_meta.name for output_meta in session.get_outputs()]
    if "last_hidden_state" in output_names:
        return "last_hidden_state"
    if not output_names:
        raise ValueError("onnx session has no outputs")
    return output_names[0]


def scenario_texts(scenario_name: str) -> list[dict[str, str]]:
    try:
        return list(bench_pytorch.SCENARIOS[scenario_name]["texts"])
    except KeyError as exc:
        raise ValueError(f"unknown scenario: {scenario_name}") from exc


def run_onnx_embeddings(
    *,
    session,
    tokenizer,
    texts: list[dict[str, str]],
    output_dimension: int,
    l2_normalize: bool,
) -> list[list[float]]:
    encoded = tokenizer(
        [bench_pytorch.prefixed_text(item) for item in texts],
        return_tensors="np",
        max_length=bench_pytorch.MAX_LENGTH,
        truncation=True,
        padding=True,
    )
    input_feed = build_onnx_input_feed(session, dict(encoded))
    output_name = resolve_output_name(session)
    outputs = session.run([output_name], input_feed)
    last_hidden_state = np.asarray(outputs[0], dtype=np.float32)
    pooled = pool_last_token_numpy(last_hidden_state, np.asarray(encoded["attention_mask"]))
    postprocessed = bench_pytorch.postprocess_embeddings(
        pooled,
        output_dimension=output_dimension,
        l2_normalize=l2_normalize,
    )
    return postprocessed.tolist()


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
    onnx_embedding: list[float],
    pytorch_embedding: list[float],
    first_values: int = 8,
) -> dict[str, object]:
    if len(onnx_embedding) != len(pytorch_embedding):
        raise ValueError("embedding lengths differ")

    diffs = [abs(lhs - rhs) for lhs, rhs in zip(onnx_embedding, pytorch_embedding)]
    rmse = math.sqrt(
        sum((lhs - rhs) * (lhs - rhs) for lhs, rhs in zip(onnx_embedding, pytorch_embedding))
        / len(onnx_embedding)
    )
    preview = first_values if first_values > 0 else len(onnx_embedding)
    return {
        "scenario": scenario,
        "dimensions": len(onnx_embedding),
        "cosine_similarity": cosine_similarity(onnx_embedding, pytorch_embedding),
        "onnx_l2_norm": l2_norm(onnx_embedding),
        "pytorch_l2_norm": l2_norm(pytorch_embedding),
        "max_abs_error": max(diffs),
        "mean_abs_error": sum(diffs) / len(diffs),
        "rmse": rmse,
        "onnx_first_values": onnx_embedding[:preview],
        "pytorch_first_values": pytorch_embedding[:preview],
        "abs_error_first_values": diffs[:preview],
    }


def build_payload_summary(
    *,
    scenario: str,
    onnx_embeddings: list[list[float]],
    pytorch_embeddings: list[list[float]],
    first_values: int = 8,
) -> dict[str, object]:
    if len(onnx_embeddings) != len(pytorch_embeddings):
        raise ValueError("embedding counts differ")
    if not onnx_embeddings:
        raise ValueError("no embeddings to compare")

    pair_summaries = [
        build_summary(
            scenario=scenario,
            onnx_embedding=onnx_embedding,
            pytorch_embedding=pytorch_embedding,
            first_values=first_values,
        )
        for onnx_embedding, pytorch_embedding in zip(onnx_embeddings, pytorch_embeddings)
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
        "onnx_l2_norm": sum(item["onnx_l2_norm"] for item in pair_summaries) / len(pair_summaries),
        "pytorch_l2_norm": sum(item["pytorch_l2_norm"] for item in pair_summaries)
        / len(pair_summaries),
        "max_abs_error": max(item["max_abs_error"] for item in pair_summaries),
        "mean_abs_error": sum(item["mean_abs_error"] for item in pair_summaries)
        / len(pair_summaries),
        "rmse": sum(item["rmse"] for item in pair_summaries) / len(pair_summaries),
        "onnx_first_values": first["onnx_first_values"],
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
        f"onnx_l2_norm: {summary['onnx_l2_norm']:.6f}",
        f"pytorch_l2_norm: {summary['pytorch_l2_norm']:.6f}",
        f"max_abs_error: {summary['max_abs_error']:.6f}",
        f"mean_abs_error: {summary['mean_abs_error']:.6f}",
        f"rmse: {summary['rmse']:.6f}",
        f"onnx_first_values: {summary['onnx_first_values']}",
        f"pytorch_first_values: {summary['pytorch_first_values']}",
        f"abs_error_first_values: {summary['abs_error_first_values']}",
    ]
    return "\n".join(lines) + "\n"


def compare_scenario(
    *,
    model_name_or_path: str,
    onnx_model_path: str,
    scenario: str,
    output_dimension: int,
    l2_normalize: bool,
    first_values: int,
) -> dict[str, object]:
    texts = scenario_texts(scenario)
    pytorch_model, tokenizer = bench_pytorch.load_model(model_name_or_path)
    pytorch_embeddings = bench_pytorch.embed_texts(
        pytorch_model,
        tokenizer,
        texts,
        output_dimension=output_dimension,
        l2_normalize=l2_normalize,
    )
    session = load_onnx_session(onnx_model_path)
    onnx_embeddings = run_onnx_embeddings(
        session=session,
        tokenizer=tokenizer,
        texts=texts,
        output_dimension=output_dimension,
        l2_normalize=l2_normalize,
    )
    return build_payload_summary(
        scenario=scenario,
        onnx_embeddings=onnx_embeddings,
        pytorch_embeddings=pytorch_embeddings,
        first_values=first_values,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = compare_scenario(
        model_name_or_path=args.model_name_or_path,
        onnx_model_path=args.onnx_model_path,
        scenario=args.scenario,
        output_dimension=args.output_dimension,
        l2_normalize=args.l2_normalize,
        first_values=args.first_values,
    )

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output_text:
        output_text = Path(args.output_text)
        output_text.parent.mkdir(parents=True, exist_ok=True)
        output_text.write_text(render_text_summary(summary), encoding="utf-8")
    if not args.output_text:
        print(render_text_summary(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
