#!/usr/bin/env python3
"""
PyTorch benchmark runner for jina-embeddings-v5-text-nano-retrieval.

Outputs machine-readable JSON for warm latency, cold start, or correctness.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import time

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging


RAW_DIM = 768
OUTPUT_DIM = 512
MAX_LENGTH = 8192

SHORT = {"kind": "query", "text": "Hello, world!"}
MEDIUM = {
    "kind": "query",
    "text": "What is the impact of large language models on software engineering productivity?",
}
LONG = {
    "kind": "document",
    "text": "The quick brown fox jumps over the lazy dog. " * 30,
}

SCENARIOS: dict[str, dict[str, object]] = {
    "single/short": {"batch_size": 1, "text_profile": "short", "texts": [SHORT]},
    "single/medium": {"batch_size": 1, "text_profile": "medium", "texts": [MEDIUM]},
    "single/long": {"batch_size": 1, "text_profile": "long", "texts": [LONG]},
    "batch/medium/1": {"batch_size": 1, "text_profile": "medium", "texts": [MEDIUM]},
    "batch/medium/4": {"batch_size": 4, "text_profile": "medium", "texts": [MEDIUM] * 4},
    "batch/medium/8": {"batch_size": 8, "text_profile": "medium", "texts": [MEDIUM] * 8},
    "batch/mixed/8": {
        "batch_size": 8,
        "text_profile": "mixed",
        "texts": [SHORT, MEDIUM, LONG, SHORT, MEDIUM, LONG, SHORT, MEDIUM],
    },
    "batch/medium/16": {"batch_size": 16, "text_profile": "medium", "texts": [MEDIUM] * 16},
}


def selected_scenarios(scenario_name: str | None) -> list[tuple[str, dict[str, object]]]:
    if scenario_name is None:
        return list(SCENARIOS.items())
    return [(scenario_name, SCENARIOS[scenario_name])]


def prefixed_text(item: dict[str, str]) -> str:
    prefix = "Query: " if item["kind"] == "query" else "Document: "
    return prefix + item["text"]


def postprocess_embeddings(
    v: np.ndarray,
    *,
    output_dimension: int = OUTPUT_DIM,
    l2_normalize: bool = True,
) -> np.ndarray:
    if v.shape[-1] != RAW_DIM:
        raise ValueError(f"expected raw dimension {RAW_DIM}, got {v.shape[-1]}")
    if output_dimension <= 0 or output_dimension > RAW_DIM:
        raise ValueError(f"output_dimension must be in [1, {RAW_DIM}], got {output_dimension}")
    output = v[..., :output_dimension]
    if not l2_normalize:
        return output
    norm = np.linalg.norm(output, axis=-1, keepdims=True)
    return output / np.maximum(norm, 1e-12)


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_token_index = attention_mask.sum(dim=1) - 1
    batch_index = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[batch_index, last_token_index]


def compute_stats(samples_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "p99_ms": float(np.percentile(samples_ms, 99)),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def load_model(model_name_or_path: str):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
    model.to("cpu")
    model.to(dtype=torch.float32)
    model.eval()
    return model, tokenizer


def embed_texts(
    model,
    tokenizer,
    texts: list[dict[str, str]],
    *,
    output_dimension: int = OUTPUT_DIM,
    l2_normalize: bool = True,
) -> list[list[float]]:
    encoded = tokenizer(
        [prefixed_text(item) for item in texts],
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
    )
    with torch.no_grad():
        output = model(**encoded)
    pooled = (
        last_token_pool(output.last_hidden_state, encoded["attention_mask"])
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
    )
    output = postprocess_embeddings(
        pooled,
        output_dimension=output_dimension,
        l2_normalize=l2_normalize,
    )
    return output.tolist()


def measure_warm_stats(
    model,
    tokenizer,
    scenario_name: str,
    warmup: int,
    iters: int,
    *,
    output_dimension: int = OUTPUT_DIM,
    l2_normalize: bool = True,
) -> dict[str, float]:
    texts = list(SCENARIOS[scenario_name]["texts"])
    for _ in range(warmup):
        embed_texts(
            model,
            tokenizer,
            texts,
            output_dimension=output_dimension,
            l2_normalize=l2_normalize,
        )

    samples_ms = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        embed_texts(
            model,
            tokenizer,
            texts,
            output_dimension=output_dimension,
            l2_normalize=l2_normalize,
        )
        samples_ms.append((time.perf_counter_ns() - start) / 1_000_000)
    return compute_stats(samples_ms)


def measure_cold_stats(
    model_name_or_path: str,
    scenario_name: str,
    *,
    output_dimension: int = OUTPUT_DIM,
    l2_normalize: bool = True,
) -> dict[str, float]:
    start = time.perf_counter_ns()
    model, tokenizer = load_model(model_name_or_path)
    embed_texts(
        model,
        tokenizer,
        list(SCENARIOS[scenario_name]["texts"]),
        output_dimension=output_dimension,
        l2_normalize=l2_normalize,
    )
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    return compute_stats([elapsed_ms])


def warm_payload(args) -> dict[str, object]:
    model, tokenizer = load_model(args.model_name_or_path)
    return {
        "implementation": "pytorch",
        "implementation_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "results": [
            {
                "scenario": scenario_name,
                "stats": measure_warm_stats(
                    model,
                    tokenizer,
                    scenario_name,
                    args.warmup,
                    args.iters,
                    output_dimension=args.output_dimension,
                    l2_normalize=args.l2_normalize,
                ),
            }
            for scenario_name, _ in selected_scenarios(args.scenario)
        ],
    }


def cold_payload(args) -> dict[str, object]:
    if not args.scenario:
        raise ValueError("--scenario is required for cold mode")
    return {
        "implementation": "pytorch",
        "implementation_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "scenario": args.scenario,
        "stats": measure_cold_stats(
            args.model_name_or_path,
            args.scenario,
            output_dimension=args.output_dimension,
            l2_normalize=args.l2_normalize,
        ),
    }


def correctness_payload(args) -> dict[str, object]:
    model, tokenizer = load_model(args.model_name_or_path)
    return {
        "implementation": "pytorch",
        "implementation_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "results": [
            {
                "scenario": scenario_name,
                "embeddings": embed_texts(
                    model,
                    tokenizer,
                    list(scenario["texts"]),
                    output_dimension=args.output_dimension,
                    l2_normalize=args.l2_normalize,
                ),
            }
            for scenario_name, scenario in selected_scenarios(args.scenario)
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["warm", "cold", "correctness"], required=True)
    parser.add_argument("--scenario")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dimension", type=int, default=OUTPUT_DIM)
    parser.add_argument("--l2-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    transformers_logging.set_verbosity_error()

    if args.mode == "warm":
        payload = warm_payload(args)
    elif args.mode == "cold":
        payload = cold_payload(args)
    else:
        payload = correctness_payload(args)

    print(json.dumps(payload))


if __name__ == "__main__":
    main()
