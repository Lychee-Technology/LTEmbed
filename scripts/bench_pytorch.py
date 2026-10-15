#!/usr/bin/env python3
"""
PyTorch retrieval reference runner for EmbeddingGemma 2 (unsloth/embeddinggemma-2).

Runs the model's own SentenceTransformers pipeline (mean pooling over every token, prompt
included, then L2 normalization) in float32 on the CPU, with the model's "query" and
"document" prompts, the same way scripts/generate_fixtures.py builds the golden fixture.
Outputs machine-readable JSON for retrieval correctness. It never runs LTEmbed or a GGUF.

Every vector it returns is unit-norm. The pipeline's last module is the model's own
Normalize, so the magnitude is gone before this script sees the vector; it truncates to
--output-dimension and normalizes again, which gives the direction LTEmbed returns with
l2_normalize on, at any width. There is no unnormalized mode: the orchestrator compares
these vectors with LTEmbed's by cosine and normalizes both before ranking. LTEmbed's
unnormalized output is checked against upstream llama.cpp instead
(tests/fixtures/llama_cpp_reference.json).

Only the text tower is loaded (`vision_config=None, audio_config=None`).

Requirements (CPU wheels are enough):
    pip install "sentence-transformers>=6.1.0" "transformers>=5.19" torch torchvision pillow numpy
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys

import numpy as np
from pathlib import Path

# torch, transformers and sentence_transformers are imported only in the functions that load
# and run the model, so the rest of the module, and its tests, need only numpy.


RAW_DIM = 768
OUTPUT_DIM = 512
MAX_LENGTH = 8192
# The model's prompts, which LTEmbed applies as QUERY_PREFIX and DOCUMENT_PREFIX
# (src/engine/mod.rs). load_model refuses a model whose prompts differ.
PROMPTS = {"query": "task: search result | query: ", "document": "title: none | text: "}
DEFAULT_RETRIEVAL_EVAL_PATH = Path(__file__).resolve().with_name("retrieval_eval_cases.json")


def progress_label(mode: str, scenario_name: str, state: str) -> str:
    return f"{mode} {scenario_name} {state}"


def emit_progress(mode: str, scenario_name: str, state: str) -> None:
    print(progress_label(mode, scenario_name, state), file=sys.stderr, flush=True)


def truncate_and_normalize(v: np.ndarray, output_dimension: int = OUTPUT_DIM) -> np.ndarray:
    if v.shape[-1] != RAW_DIM:
        raise ValueError(f"expected raw dimension {RAW_DIM}, got {v.shape[-1]}")
    if output_dimension <= 0 or output_dimension > RAW_DIM:
        raise ValueError(
            f"expected output dimension in [1, {RAW_DIM}], got {output_dimension}"
        )
    truncated = v[..., :output_dimension]
    norm = np.linalg.norm(truncated, axis=-1, keepdims=True)
    return truncated / np.maximum(norm, 1e-12)


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
    import sentence_transformers
    import torch

    # stdout carries only the JSON payload.
    with contextlib.redirect_stdout(sys.stderr):
        model = sentence_transformers.SentenceTransformer(
            model_name_or_path,
            device="cpu",
            model_kwargs={"dtype": torch.float32},
            config_kwargs={"vision_config": None, "audio_config": None},
        )
    model.eval()
    prompts = {kind: model.prompts.get(kind) for kind in PROMPTS}
    if prompts != PROMPTS:
        raise ValueError(f"the model's prompts {prompts} differ from LTEmbed's {PROMPTS}")
    if next(model[0].auto_model.parameters()).dtype != torch.float32:
        raise ValueError("the model did not load in float32")
    return model


def count_tokens(model, kind: str, text: str) -> int:
    return int(model.preprocess([PROMPTS[kind] + text])["input_ids"].shape[1])


def embed_texts(
    model,
    texts: list[dict[str, str]],
    output_dimension: int = OUTPUT_DIM,
) -> list[list[float]]:
    """Embed each text with its kind's prompt, one at a time (no padding), like LTEmbed, as
    a unit-norm vector of `output_dimension` values.

    Raises instead of truncating an input longer than MAX_LENGTH tokens, as LTEmbed does.
    """
    rows = []
    for item in texts:
        kind = item["kind"]
        tokens = count_tokens(model, kind, item["text"])
        if tokens > MAX_LENGTH:
            raise ValueError(f"input of {tokens} tokens exceeds MAX_LENGTH {MAX_LENGTH}")
        embedding = model.encode(
            [item["text"]],
            prompt_name=kind,
            batch_size=1,
            convert_to_numpy=True,
        )[0]
        rows.append(np.asarray(embedding, dtype=np.float32))
    pooled = np.stack(rows) if rows else np.zeros((0, RAW_DIM), dtype=np.float32)
    return truncate_and_normalize(pooled, output_dimension=output_dimension).tolist()


def load_retrieval_eval_cases(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "cases" in payload:
        return list(payload["cases"])
    return [payload]


def retrieval_payload(args) -> dict[str, object]:
    import sentence_transformers
    import torch
    import transformers

    cases = load_retrieval_eval_cases(args.retrieval_eval_path)
    model = load_model(args.model_name_or_path)

    with torch.inference_mode():
        results = [
            {
                "dataset_name": str(case["name"]),
                "queries": [
                    {
                        "id": str(query["id"]),
                        "embedding": embedding,
                    }
                    for query, embedding in zip(
                        case["queries"],
                        embed_texts(
                            model,
                            [{"kind": "query", "text": str(query["text"])} for query in case["queries"]],
                            output_dimension=args.output_dimension,
                        ),
                        strict=True,
                    )
                ],
                "documents": [
                    {
                        "id": str(document["id"]),
                        "embedding": embedding,
                    }
                    for document, embedding in zip(
                        case["documents"],
                        embed_texts(
                            model,
                            [{"kind": "document", "text": str(document["text"])} for document in case["documents"]],
                            output_dimension=args.output_dimension,
                        ),
                        strict=True,
                    )
                ],
            }
            for case in cases
        ]
    return {
        "implementation": "pytorch",
        "implementation_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sentence_transformers_version": sentence_transformers.__version__,
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["warm", "cold", "correctness", "retrieval"], required=True)
    parser.add_argument("--scenario")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--retrieval-eval-path", type=Path, default=DEFAULT_RETRIEVAL_EVAL_PATH)
    parser.add_argument(
        "--fixture-path",
        type=Path,
        default=None,
        help="Optional resolved-fixture JSON overriding per-scenario texts.",
    )
    parser.add_argument("--output-dimension", type=int, default=OUTPUT_DIM)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    import torch
    from transformers.utils import logging as transformers_logging

    args = parse_args()
    torch.set_num_threads(args.threads)
    transformers_logging.set_verbosity_error()
    if args.mode != "retrieval":
        raise SystemExit(f"bench_pytorch now supports only --mode retrieval (got {args.mode})")
    payload = retrieval_payload(args)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
