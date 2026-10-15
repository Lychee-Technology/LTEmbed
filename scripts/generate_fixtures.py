#!/usr/bin/env python3
"""
Generate the golden embedding fixture from EmbeddingGemma 2 with SentenceTransformers in F32.

This is the independent reference: the original safetensors weights, run by PyTorch on the
CPU in float32 (not float16, which this model's activations do not support), with the
model's own SentenceTransformers pipeline (mean pooling over every token, prompt included,
then L2 normalization). It never runs LTEmbed, llama.cpp or a GGUF.

Only the text tower is loaded (`vision_config=None, audio_config=None`); the text embedding
does not depend on the vision and audio encoders.

Requirements (CPU wheels are enough):
    pip install "sentence-transformers>=6.1.0" "transformers>=5.19" torch torchvision pillow numpy

    `EmbeddingGemma2Processor` imports torchvision and pillow even for text. Install torch and
    torchvision from the same index (for example https://download.pytorch.org/whl/cpu), or the
    torchvision import fails with "operator torchvision::nms does not exist".

Usage:
    python3 scripts/generate_fixtures.py                     # downloads REPO at REVISION
    python3 scripts/generate_fixtures.py --model-dir DIR     # an `hf download --local-dir` copy

Output:
    tests/fixtures/test_fixtures.json

The fixture holds one 768-wide, L2-normalized vector per input. The Rust tests derive each
Matryoshka width (512/256/128) by truncating it and renormalizing, which is how the model
card says to use the smaller widths. Keep REVISION equal to TOKENIZER_REVISION in
scripts/assemble_gguf_bundle.py, the revision of the tests' reference tokenizer.json.
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path


REPO = "unsloth/embeddinggemma-2"
REVISION = "c8f89cc7ecb4ba5122cd15fdd487422b2b487daf"
UPSTREAM = "google/embeddinggemma-2"
OUTPUT_PATH = Path("tests/fixtures/test_fixtures.json")
CORPUS_PATH = Path("scripts/benchmark_corpus.json")
QUERY_PROMPT = "task: search result | query: "
DOCUMENT_PROMPT = "title: none | text: "
RAW_DIM = 768
DIMS = [768, 512, 256, 128]
MAX_LENGTH = 8192
# Inputs no longer than this many tokens are "short": the tests check them at every width.
SHORT_TOKENS = 512

# `relevant` names the document that must rank first among the documents for that query.
SHORT_CASES = [
    {"name": "query_en", "kind": "query", "text": "What is machine learning?",
     "relevant": "document_en"},
    {"name": "document_en", "kind": "document",
     "text": "Machine learning is a branch of artificial intelligence in which computers learn "
             "patterns from data instead of following explicitly programmed rules."},
    {"name": "query_zh", "kind": "query", "text": "如何重置密码？", "relevant": "document_zh"},
    {"name": "document_zh", "kind": "document",
     "text": "在登录页面点击“忘记密码”，然后按照邮件中的链接设置新密码。"},
    {"name": "query_ja", "kind": "query", "text": "東京で一番高い建物は何ですか？",
     "relevant": "document_ja"},
    {"name": "document_ja", "kind": "document",
     "text": "東京スカイツリーは高さ634メートルで、日本で最も高い建造物です。"},
    {"name": "query_ko", "kind": "query", "text": "서울의 인구는 얼마인가요?",
     "relevant": "document_ko"},
    {"name": "document_ko", "kind": "document",
     "text": "서울특별시의 인구는 약 940만 명으로 대한민국에서 가장 큰 도시이다."},
    {"name": "query_en_cross_fr", "kind": "query", "text": "Where is the Eiffel Tower located?",
     "relevant": "document_fr"},
    {"name": "document_fr", "kind": "document",
     "text": "La tour Eiffel est une tour de fer puddlé située à Paris, à l'extrémité "
             "nord-ouest du parc du Champ-de-Mars."},
    {"name": "document_de", "kind": "document",
     "text": "Der Rhein ist einer der längsten Flüsse Europas und fließt durch die Schweiz, "
             "Deutschland und die Niederlande."},
    {"name": "document_ru", "kind": "document",
     "text": "Москва — столица России и крупнейший по численности населения город страны."},
    {"name": "document_ar", "kind": "document", "text": "القاهرة هي عاصمة مصر وأكبر مدنها."},
    {"name": "query_code", "kind": "query",
     "text": "python function that reverses a singly linked list",
     "relevant": "document_code_python"},
    {"name": "document_code_python", "kind": "document",
     "text": "def reverse_list(head):\n    prev = None\n    while head:\n"
             "        head.next, prev, head = prev, head, head.next\n    return prev"},
    {"name": "document_code_rust", "kind": "document",
     "text": "fn fibonacci(n: u64) -> u64 {\n    match n {\n        0 => 0,\n        1 => 1,\n"
             "        _ => fibonacci(n - 1) + fibonacci(n - 2),\n    }\n}"},
    {"name": "query_zh_en_mixed", "kind": "query",
     "text": "Rust 的所有权（ownership）机制如何保证内存安全？",
     "relevant": "document_zh_en_mixed"},
    {"name": "document_zh_en_mixed", "kind": "document",
     "text": "Rust 通过所有权（ownership）和借用（borrowing）规则在编译期保证内存安全，"
             "因此不需要垃圾回收器（garbage collector）。"},
    {"name": "query_emoji", "kind": "query", "text": "🦀🚀 fast and safe systems programming"},
    {"name": "query_empty", "kind": "query", "text": ""},
    {"name": "document_whitespace", "kind": "document",
     "text": "line one\n\n  line two\t\ttabbed\r\nend   ."},
]


def corpus_text(scenario: str) -> str:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))[scenario]["text"]


def cut_to_tokens(count_tokens, prompt: str, base: str, target: int) -> str:
    """The longest prefix of `base`, ending in a non-space character, whose prompted text is
    exactly `target` tokens (BOS and EOS included). Raises if no prefix gives that count."""
    if count_tokens(prompt + base) < target:
        raise ValueError(f"base text is shorter than {target} tokens")
    low, high = 0, len(base)
    while low < high:  # largest length whose count is <= target
        mid = (low + high + 1) // 2
        if count_tokens(prompt + base[:mid]) <= target:
            low = mid
        else:
            high = mid - 1
    for length in range(low, max(low - 200, 0), -1):
        text = base[:length]
        if not text[-1].isspace() and count_tokens(prompt + text) == target:
            return text
    raise ValueError(f"no prefix of the base text is exactly {target} tokens")


def long_cases(count_tokens) -> list:
    long_text = corpus_text("long")
    repeated = "\n\n".join([long_text] * 3)
    return [
        {"name": "document_medium", "kind": "document", "text": corpus_text("medium")},
        {"name": "document_long_1k", "kind": "document",
         "text": cut_to_tokens(count_tokens, DOCUMENT_PROMPT, long_text, 1024)},
        {"name": "document_long_4k", "kind": "document", "text": long_text},
        # The longest input the engine accepts: MAX_LENGTH tokens with the prompt, BOS and EOS.
        {"name": "document_max_length", "kind": "document",
         "text": cut_to_tokens(count_tokens, DOCUMENT_PROMPT, repeated, MAX_LENGTH)},
    ]


def check_local_revision(model_dir: Path) -> None:
    """Require an `hf download --local-dir` copy of REPO at REVISION."""
    metadata = model_dir / ".cache/huggingface/download/model.safetensors.metadata"
    if not metadata.exists():
        raise SystemExit(f"{metadata} not found: download with `hf download {REPO} "
                         f"--revision {REVISION} --local-dir {model_dir}`")
    revision = metadata.read_text(encoding="utf-8").split("\n", 1)[0].strip()
    if revision != REVISION:
        raise SystemExit(f"{model_dir} holds revision {revision}, expected {REVISION}")


def format_json(value, level: int = 0) -> str:
    """Indented JSON that keeps each list of numbers, such as an embedding, on one line."""
    indent = "  " * (level + 1)
    if isinstance(value, dict) and value:
        items = [f"{indent}{json.dumps(key)}: {format_json(item, level + 1)}"
                 for key, item in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + "  " * level + "}"
    if isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        items = [indent + format_json(item, level + 1) for item in value]
        return "[\n" + ",\n".join(items) + "\n" + "  " * level + "]"
    return json.dumps(value, ensure_ascii=False)


def float32_list(vector) -> list:
    """The values as float32, written with the 9 significant digits that round-trip them."""
    return [float(f"{float(x):.9g}") for x in vector.astype("float32")]


def check_retrieval(fixtures: list, embeddings: dict) -> None:
    """Every query with a `relevant` document must rank it first among the documents."""
    documents = [f["name"] for f in fixtures if f["kind"] == "document"]
    for fixture in fixtures:
        relevant = fixture.get("relevant")
        if relevant is None:
            continue
        scores = {name: float(embeddings[fixture["name"]] @ embeddings[name])
                  for name in documents}
        ranked = sorted(scores, key=scores.get, reverse=True)
        if ranked[0] != relevant:
            raise SystemExit(f"{fixture['name']}: the reference ranks {ranked[0]} first, "
                             f"not {relevant}: {scores}")
        margin = scores[ranked[0]] - scores[ranked[1]]
        print(f"  retrieval {fixture['name']} -> {relevant} (margin {margin:.4f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path,
                        help=f"local copy of {REPO} at {REVISION} instead of downloading it")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0: default)")
    args = parser.parse_args()

    # Imported here so that the helpers above can be tested without these installed.
    import numpy as np
    import sentence_transformers
    import torch
    import transformers

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.model_dir:
        check_local_revision(args.model_dir)
        source, revision_kwargs = str(args.model_dir), {}
    else:
        source, revision_kwargs = REPO, {"revision": REVISION}

    print(f"Loading {REPO}@{REVISION} (text only, float32, CPU) ...")
    model = sentence_transformers.SentenceTransformer(
        source,
        device="cpu",
        model_kwargs={"dtype": torch.float32},
        config_kwargs={"vision_config": None, "audio_config": None},
        **revision_kwargs,
    )
    model.eval()
    prompts = {"query": model.prompts["query"], "document": model.prompts["document"]}
    if prompts != {"query": QUERY_PROMPT, "document": DOCUMENT_PROMPT}:
        raise SystemExit(f"the model's prompts changed: {prompts}")
    auto_model = model[0].auto_model
    if next(auto_model.parameters()).dtype != torch.float32:
        raise SystemExit("the model did not load in float32")

    def count_tokens(text: str) -> int:
        return int(model.preprocess([text])["input_ids"].shape[1])

    cases = SHORT_CASES + long_cases(count_tokens)
    fixtures, embeddings = [], {}
    with torch.inference_mode():
        for case in cases:
            prompted = prompts[case["kind"]] + case["text"]
            tokens = count_tokens(prompted)
            if tokens > MAX_LENGTH:
                raise SystemExit(f"{case['name']}: {tokens} tokens exceeds {MAX_LENGTH}")
            embedding = model.encode(
                [case["text"]],
                prompt_name=case["kind"],
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0].astype(np.float64)
            if embedding.shape != (RAW_DIM,) or not np.all(np.isfinite(embedding)):
                raise SystemExit(f"{case['name']}: unexpected embedding {embedding.shape}")
            embeddings[case["name"]] = embedding
            fixture = {"name": case["name"], "kind": case["kind"], "text": case["text"],
                       "tokens": tokens}
            if "relevant" in case:
                fixture["relevant"] = case["relevant"]
            fixture["embedding"] = float32_list(embedding)
            fixtures.append(fixture)
            print(f"  OK: {case['name']} ({tokens} tokens)")

    check_retrieval(fixtures, embeddings)

    output = {
        "generator": "scripts/generate_fixtures.py",
        "model": {"repo": REPO, "revision": REVISION, "upstream": UPSTREAM},
        "reference": {
            "implementation": "sentence-transformers (PyTorch), text tower only",
            "dtype": "float32",
            "device": "cpu",
            "attention": auto_model.config._attn_implementation,
            "pooling": "mean over all tokens, prompt included",
            "normalized": True,
            "versions": {
                "python": platform.python_version(),
                "sentence_transformers": sentence_transformers.__version__,
                "transformers": transformers.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        },
        "prompts": prompts,
        "raw_dim": RAW_DIM,
        "dims": DIMS,
        "max_length": MAX_LENGTH,
        "short_tokens": SHORT_TOKENS,
        "corpus_sha256": hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest(),
        "fixtures": fixtures,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(format_json(output) + "\n", encoding="utf-8")
    print(f"\nSaved {len(fixtures)} fixtures → {OUTPUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
