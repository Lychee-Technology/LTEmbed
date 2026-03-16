#!/usr/bin/env python3
"""
Generate golden test fixtures from e5-small-v2 via HuggingFace transformers.

Requirements:
    pip install transformers torch numpy huggingface_hub

Usage:
    python3 scripts/generate_fixtures.py

Output:
    tests/fixtures/test_fixtures.json
"""
import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "intfloat/e5-small-v2"
OUTPUT_PATH = "tests/fixtures/test_fixtures.json"

TEST_INPUTS = [
    "query: Hello, world!",
    "query: What is machine learning?",
    "passage: The quick brown fox jumps over the lazy dog.",
    "query: 人工智能",
]


def mean_pool(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # [1, seq, hidden]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def main():
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    fixtures = []
    with torch.no_grad():
        for text in TEST_INPUTS:
            encoded = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            output = model(**encoded)
            pooled = mean_pool(output, encoded["attention_mask"]).squeeze(0).numpy()
            normalized = l2_normalize(pooled)
            fixtures.append({"input": text, "embedding": normalized.tolist()})
            print(f"  OK: {text[:60]!r}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"model": MODEL_NAME, "dim": 384, "fixtures": fixtures}, f, indent=2)
    print(f"\nSaved {len(fixtures)} fixtures → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
