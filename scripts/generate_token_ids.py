#!/usr/bin/env python3
"""
Generate the token-id parity fixture for `HFTokenizer`.

Encodes a small set of inputs with the Python `tokenizers` library, once singly and once as a
single padded batch. It uses the model's tokenizer.json at the same revision that the CI bundle
downloads. `test_token_ids_match_python_tokenizers` in tests/integration_tests.rs encodes the
same inputs with `HFTokenizer` and requires identical input ids, attention masks and type ids.

Requirements:
    pip install tokenizers==<version>, where <version> is the Rust `tokenizers` crate version in
    Cargo.lock. Python and Rust `tokenizers` releases share version numbers, so the fixture then
    compares HFTokenizer with the Python binding of the same release. The script exits with the
    pip command to run if the installed version differs.

Usage:
    python3 scripts/generate_token_ids.py

Output:
    tests/fixtures/token_ids.json

Regenerate the fixture after changing INPUTS or bumping REVISION. Keep REVISION equal to
HF_REVISION in .github/workflows/ci.yml. The fixture records the sha256 of the tokenizer.json
it was generated from. If a bundle's tokenizer.json has a different hash, the Rust test fails
and says to rerun this script.
"""

import hashlib
import json
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path


REPO = "jinaai/jina-embeddings-v5-text-nano-retrieval"
REVISION = "ac5d898c8d382b17167c33e5c8af644a3519b47d"
OUTPUT_PATH = Path("tests/fixtures/token_ids.json")
CARGO_LOCK_PATH = Path("Cargo.lock")
MAX_LENGTH = 8192  # ltembed::engine::MAX_LENGTH: HFTokenizer rejects longer inputs

# Each input is {"text": ...} or {"repeat": {prefix, unit, count, suffix}}. Long inputs use
# the repeat form so that neither the fixture nor the Rust test stores them as a literal.
INPUTS = [
    # Retrieval-prefixed English and CJK text.
    {"name": "query_en", "input": {"text": "Query: What is machine learning?"}},
    {
        "name": "document_en",
        "input": {
            "text": "Document: The quick brown fox isn't lazy; it's 3.14159 times faster—"
            "really!"
        },
    },
    {"name": "query_zh", "input": {"text": "Query: 人工智能是什么？"}},
    {
        "name": "document_zh_en",
        "input": {
            "text": "Document: 机器学习（machine learning）是人工智能的一个分支。"
            "Deep learning 使用多层神经网络，例如 Transformer。"
        },
    },
    {"name": "query_ja_ko_emoji", "input": {"text": "Query: 東京の天気は？ 서울 날씨는? 🦀🚀"}},
    {
        "name": "document_whitespace_mix",
        "input": {"text": "Document: line one\n\n  line two\t\ttabbed\r\nend   "},
    },
    {"name": "empty", "input": {"text": ""}},
    # Added tokens. <|mask|> (id 128002) is lstrip: it absorbs the whitespace before it.
    {"name": "mask_no_space", "input": {"text": "Query: The capital of France is<|mask|>."}},
    {"name": "mask_one_space", "input": {"text": "Query: The capital of France is <|mask|>."}},
    {
        "name": "mask_many_spaces",
        "input": {"text": "Query: The capital of France is    \t<|mask|>."},
    },
    {
        "name": "adjacent_added_tokens",
        "input": {
            "text": "Document: <|begin_of_text|><|mask|><|mask|><|start_header_id|>user"
            "<|end_header_id|><|end_of_text|>"
        },
    },
    {
        "name": "unterminated_added_tokens",
        "input": {"text": "Query: an unterminated <|mask and <|mask| and <|"},
    },
    {"name": "uppercase_added_token", "input": {"text": "Query: upper case <|MASK|> is plain text"}},
    # 7,816 tokens with either regex backend, but different ids. Oniguruma's `\s+(?!\S)` leaves
    # the last space to form "Ġb". fancy-regex hits its backtrack limit there, so the whole run
    # stays one piece and "b" is encoded alone.
    {
        "name": "whitespace_run_1m",
        "input": {"repeat": {"prefix": "a", "unit": " ", "count": 1_000_000, "suffix": "b"}},
    },
]


def build_text(spec: dict) -> str:
    """Expand an input spec into the text to encode."""
    if "text" in spec:
        return spec["text"]
    repeat = spec["repeat"]
    return repeat["prefix"] + repeat["unit"] * repeat["count"] + repeat["suffix"]


def locked_tokenizers_version(cargo_lock: str) -> str:
    """The version of the Rust `tokenizers` crate locked in Cargo.lock."""
    versions = re.findall(r'^name = "tokenizers"\nversion = "([^"]+)"$', cargo_lock, re.MULTILINE)
    if len(versions) != 1:
        raise ValueError(f"expected one tokenizers package in Cargo.lock, found versions {versions}")
    return versions[0]


def encoding_fields(encoding) -> dict:
    return {
        "input_ids": list(encoding.ids),
        "attention_mask": list(encoding.attention_mask),
        "type_ids": list(encoding.type_ids),
    }


def strip_right_padding(fields: dict, pad_id: int, pad_type_id: int) -> dict:
    """Drop a batch row's trailing padding and record its length as "padding".

    Rows of a batch that contains the 1M-space input are padded to ~7,800 tokens, so storing
    them in full would make the fixture megabytes long. The Rust test appends the padding back
    before comparing. Raises unless the dropped tail is made up entirely of padding tokens.
    """
    ids, mask, type_ids = fields["input_ids"], fields["attention_mask"], fields["type_ids"]
    real = len(mask)
    while real > 0 and mask[real - 1] == 0:
        real -= 1
    if (
        0 in mask[:real]
        or any(token != pad_id for token in ids[real:])
        or any(type_id != pad_type_id for type_id in type_ids[real:])
    ):
        raise ValueError(f"batch row is not right-padded with pad id {pad_id}: {fields}")
    return {key: values[:real] for key, values in fields.items()} | {"padding": len(mask) - real}


def format_json(value, level: int = 0) -> str:
    """Indented JSON that keeps each list of scalars, such as a token-id list, on one line."""
    indent = "  " * (level + 1)
    if isinstance(value, dict) and value:
        items = [f"{indent}{json.dumps(key)}: {format_json(item, level + 1)}" for key, item in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + "  " * level + "}"
    if isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        items = [indent + format_json(item, level + 1) for item in value]
        return "[\n" + ",\n".join(items) + "\n" + "  " * level + "]"
    return json.dumps(value, ensure_ascii=False)


def main():
    # Imported here so that the helpers above can be tested without `tokenizers` installed.
    import tokenizers

    locked = locked_tokenizers_version(CARGO_LOCK_PATH.read_text(encoding="utf-8"))
    if tokenizers.__version__ != locked:
        raise SystemExit(
            f"Python tokenizers is {tokenizers.__version__}, but Cargo.lock locks the Rust crate "
            f"at {locked}. Run `pip install tokenizers=={locked}` and rerun this script."
        )

    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/tokenizer.json"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tokenizer.json"
        print(f"Downloading {url} ...")
        with urllib.request.urlopen(url) as response, open(path, "wb") as handle:
            shutil.copyfileobj(response, handle)
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        tokenizer = tokenizers.Tokenizer.from_file(str(path))

    padding = tokenizer.padding
    if padding is None or padding["direction"] != "right":
        raise SystemExit(f"expected a right-padding tokenizer, got padding={padding}")

    texts = [build_text(item["input"]) for item in INPUTS]
    singles = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
    batch = tokenizer.encode_batch(texts, add_special_tokens=True)

    cases = []
    for item, single, row in zip(INPUTS, singles, batch, strict=True):
        if len(single.ids) > MAX_LENGTH:
            raise SystemExit(f"{item['name']}: {len(single.ids)} tokens exceeds {MAX_LENGTH}")
        cases.append(
            {
                "name": item["name"],
                "input": item["input"],
                "single": encoding_fields(single),
                "batch": strip_right_padding(
                    encoding_fields(row), padding["pad_id"], padding["pad_type_id"]
                ),
            }
        )
        print(f"  OK: {item['name']} ({len(single.ids)} tokens)")

    fixture = {
        "tokenizer": {"repo": REPO, "revision": REVISION, "sha256": sha256},
        "tokenizers_version": tokenizers.__version__,
        "padding": {
            "direction": padding["direction"],
            "pad_id": padding["pad_id"],
            "pad_type_id": padding["pad_type_id"],
        },
        "cases": cases,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(format_json(fixture) + "\n", encoding="utf-8")
    print(f"\nSaved {len(cases)} cases → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
