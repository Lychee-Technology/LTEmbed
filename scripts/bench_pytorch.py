#!/usr/bin/env python3
"""
PyTorch benchmark runner for intfloat/e5-small-v2.

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


SHORT = "query: Hello, world!"
MEDIUM = "query: What is the impact of large language models on software engineering productivity?"
LONG = "passage: " + "The quick brown fox jumps over the lazy dog. " * 30

SCENARIOS: dict[str, dict[str, object]] = {
    "single/short": {"batch_size": 1, "text_profile": "short", "texts": [SHORT]},
    "single/medium": {"batch_size": 1, "text_profile": "medium", "texts": [MEDIUM]},
    "single/long": {"batch_size": 1, "text_profile": "long", "texts": [LONG]},
    "batch/medium/1": {"batch_size": 1, "text_profile": "medium", "texts": [MEDIUM]},
    "batch/medium/4": {"batch_size": 4, "text_profile": "medium", "texts": [MEDIUM] * 4},
    "batch/medium/8": {"batch_size": 8, "text_profile": "medium", "texts": [MEDIUM] * 8},
    "batch/medium/16": {"batch_size": 16, "text_profile": "medium", "texts": [MEDIUM] * 16},
}


def mean_pool(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-12)


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
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()
    model.to("cpu")
    return model, tokenizer


def embed_texts(model, tokenizer, texts: list[str]) -> list[list[float]]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )
    with torch.no_grad():
        output = model(**encoded)
    pooled = mean_pool(output, encoded["attention_mask"]).cpu().numpy()
    normalized = l2_normalize(pooled)
    return normalized.tolist()


def measure_warm_stats(model, tokenizer, scenario_name: str, warmup: int, iters: int) -> dict[str, float]:
    texts = list(SCENARIOS[scenario_name]["texts"])
    for _ in range(warmup):
        embed_texts(model, tokenizer, texts)

    samples_ms = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        embed_texts(model, tokenizer, texts)
        samples_ms.append((time.perf_counter_ns() - start) / 1_000_000)
    return compute_stats(samples_ms)


def measure_cold_stats(model_name_or_path: str, scenario_name: str) -> dict[str, float]:
    start = time.perf_counter_ns()
    model, tokenizer = load_model(model_name_or_path)
    embed_texts(model, tokenizer, list(SCENARIOS[scenario_name]["texts"]))
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
                "stats": measure_warm_stats(model, tokenizer, scenario_name, args.warmup, args.iters),
            }
            for scenario_name in SCENARIOS
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
        "stats": measure_cold_stats(args.model_name_or_path, args.scenario),
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
                "embeddings": embed_texts(model, tokenizer, list(scenario["texts"])),
            }
            for scenario_name, scenario in SCENARIOS.items()
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["warm", "cold", "correctness"], required=True)
    parser.add_argument("--scenario")
    parser.add_argument("--model-name-or-path", required=True)
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
