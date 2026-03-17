#!/usr/bin/env python3
"""
PyTorch CPU benchmark for e5-small-v2 embedding.

Measures warm-invocation latency for:
  - Single embed (short / medium / long inputs)
  - Batch embed (batch_size = 1, 4, 8, 16) on medium input

Run from repo root:
    python3 scripts/bench_pytorch.py

Options:
    --threads N   Number of PyTorch CPU threads (default: 1, to match single-threaded Rust)
    --warmup N    Warmup iterations before timing (default: 10)
    --iters N     Timed iterations (default: 100)
    --no-single-thread  Use all available CPU threads (shows PyTorch at its best)

Requirements:
    pip install transformers torch numpy
"""
import argparse
import time
import statistics
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "intfloat/e5-small-v2"

SHORT = "query: Hello, world!"
MEDIUM = "query: What is the impact of large language models on software engineering productivity?"
LONG = "passage: " + "The quick brown fox jumps over the lazy dog. " * 30

BATCH_SIZES = [1, 4, 8, 16]


def mean_pool(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # [B, seq, hidden]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-12)


def embed_single(model, tokenizer, text: str) -> np.ndarray:
    encoded = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        output = model(**encoded)
    pooled = mean_pool(output, encoded["attention_mask"]).squeeze(0).numpy()
    return l2_normalize(pooled)


def embed_batch(model, tokenizer, texts: list[str]) -> np.ndarray:
    encoded = tokenizer(
        texts, return_tensors="pt", max_length=512, truncation=True, padding=True
    )
    with torch.no_grad():
        output = model(**encoded)
    pooled = mean_pool(output, encoded["attention_mask"]).numpy()
    return l2_normalize(pooled)


def bench(fn, warmup: int, iters: int) -> dict:
    """Run fn() warmup+iters times, return stats over the timed iters (ms)."""
    for _ in range(warmup):
        fn()
    times_ns = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        times_ns.append(time.perf_counter_ns() - t0)
    times_ms = [t / 1_000_000 for t in times_ns]
    return {
        "mean": statistics.mean(times_ms),
        "median": statistics.median(times_ms),
        "p95": float(np.percentile(times_ms, 95)),
        "p99": float(np.percentile(times_ms, 99)),
        "min": min(times_ms),
        "max": max(times_ms),
    }


def fmt_row(label: str, s: dict) -> str:
    return (
        f"| {label:<30} "
        f"| {s['mean']:>9.2f} "
        f"| {s['median']:>11.2f} "
        f"| {s['p95']:>8.2f} "
        f"| {s['p99']:>8.2f} |"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--no-single-thread", action="store_true")
    args = parser.parse_args()

    if not args.no_single_thread:
        torch.set_num_threads(args.threads)
        thread_note = f"{args.threads} thread(s)"
    else:
        thread_note = f"all threads ({torch.get_num_threads()} available)"

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.to("cpu")
    print(f"Model loaded. Threads: {thread_note}, warmup={args.warmup}, iters={args.iters}\n")

    header = (
        f"## PyTorch CPU Benchmarks — {MODEL_NAME}\n"
        f"## threads={thread_note}, warmup={args.warmup}, iters={args.iters}\n\n"
        f"| {'Scenario':<30} | {'Mean (ms)':>9} | {'Median (ms)':>11} | {'p95 (ms)':>8} | {'p99 (ms)':>8} |\n"
        f"|{'-'*32}|{'-'*11}|{'-'*13}|{'-'*10}|{'-'*10}|"
    )
    print(header)

    # Single embed — three text lengths
    for label, text in [("single/short", SHORT), ("single/medium", MEDIUM), ("single/long", LONG)]:
        s = bench(lambda t=text: embed_single(model, tokenizer, t), args.warmup, args.iters)
        print(fmt_row(label, s))

    # Batch embed — varying batch size, medium text
    for bs in BATCH_SIZES:
        texts = [MEDIUM] * bs
        label = f"batch={bs}/medium"
        s = bench(lambda t=texts: embed_batch(model, tokenizer, t), args.warmup, args.iters)
        print(fmt_row(label, s))

    print()
    print("Note: compare with `RUSTFLAGS=\"-C target-cpu=native\" cargo bench` output.")
    print("      Rust criterion reports are in target/criterion/")


if __name__ == "__main__":
    main()
