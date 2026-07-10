#!/usr/bin/env python3
"""
Generate golden test fixtures from jina-embeddings-v5-text-nano-retrieval.

Requirements:
    pip install transformers torch numpy

Usage:
    python3 scripts/generate_fixtures.py

Output:
    tests/fixtures/test_fixtures.json
"""

import json
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "jinaai/jina-embeddings-v5-text-nano-retrieval"
OUTPUT_PATH = "tests/fixtures/test_fixtures.json"
RAW_DIM = 768
OUTPUT_DIM = 512
MAX_LENGTH = 8192

TEST_INPUTS = [
    {"kind": "query", "text": "Hello, world!"},
    {"kind": "query", "text": "What is machine learning?"},
    {"kind": "document", "text": "The quick brown fox jumps over the lazy dog."},
    {"kind": "query", "text": "人工智能"},
]


def prefixed_text(kind: str, text: str) -> str:
    prefix = "Query: " if kind == "query" else "Document: "
    return prefix + text


def truncate_and_normalize(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[-1] != RAW_DIM:
        raise ValueError(f"expected raw dimension {RAW_DIM}, got {embeddings.shape[-1]}")
    truncated = embeddings[..., :OUTPUT_DIM]
    norms = np.linalg.norm(truncated, axis=-1, keepdims=True)
    return truncated / np.maximum(norms, 1e-12)


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_token_index = attention_mask.sum(dim=1) - 1
    batch_index = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[batch_index, last_token_index]


def main():
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Force float32: this model's native dtype is bfloat16, but the golden is the F32
    # reference (and bf16 tensors cannot be converted to numpy directly).
    model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()

    fixtures = []
    with torch.no_grad():
        for item in TEST_INPUTS:
            encoded = tokenizer(
                prefixed_text(item["kind"], item["text"]),
                return_tensors="pt",
                max_length=MAX_LENGTH,
                truncation=True,
            )
            output = model(**encoded)
            pooled = last_token_pool(output.last_hidden_state, encoded["attention_mask"])
            embedding = truncate_and_normalize(pooled.squeeze(0).float().cpu().numpy())
            fixtures.append(
                {
                    "kind": item["kind"],
                    "text": item["text"],
                    "embedding": embedding.tolist(),
                }
            )
            print(f"  OK: {item['kind']} {item['text'][:60]!r}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": MODEL_NAME,
                "raw_dim": RAW_DIM,
                "dim": OUTPUT_DIM,
                "max_length": MAX_LENGTH,
                "fixtures": fixtures,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved {len(fixtures)} fixtures → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
