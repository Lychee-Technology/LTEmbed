#!/usr/bin/env python3
"""
Generate the token-id parity fixture for `HFTokenizer`.

Encodes a small set of inputs with the Python `tokenizers` library, once singly and once as a
single batch. It uses the model's tokenizer.json at the same revision that the CI bundle
downloads. `test_token_ids_match_python_tokenizers` in tests/integration_tests.rs encodes the
same inputs with `HFTokenizer` and requires identical input ids, attention masks and type ids.

EmbeddingGemma 2's tokenizer.json configures no padding, so each batch row must equal the
input's single encoding, and the fixture stores only the single encodings ("padding": null).
A right-padding tokenizer would also store each batch row, without its padding.

Requirements:
    pip install tokenizers==<version>, where <version> is the Rust `tokenizers` crate version in
    Cargo.lock. Python and Rust `tokenizers` releases share version numbers, so the fixture then
    compares HFTokenizer with the Python binding of the same release. The script exits with the
    pip command to run if the installed version differs.

Usage:
    python3 scripts/generate_token_ids.py

Output:
    tests/fixtures/token_ids.json

Regenerate the fixture after changing INPUTS or bumping REVISION. Keep REPO and REVISION
equal to TOKENIZER_REPO and TOKENIZER_REVISION in scripts/assemble_gguf_bundle.py, which
builds the CI and release bundles, and update its TOKENIZER_SHA256 to the hash this script
records. If a bundle's tokenizer.json has a different hash, the Rust test fails and says to
rerun this script.
"""

import hashlib
import json
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path


REPO = "unsloth/embeddinggemma-2"
REVISION = "c8f89cc7ecb4ba5122cd15fdd487422b2b487daf"
OUTPUT_PATH = Path("tests/fixtures/token_ids.json")
CARGO_LOCK_PATH = Path("Cargo.lock")
MAX_LENGTH = 8192  # ltembed::engine::MAX_LENGTH: HFTokenizer rejects longer inputs

QUERY = "task: search result | query: "
DOCUMENT = "title: none | text: "

# Each input is {"text": ...} or {"repeat": {prefix, unit, count, suffix}}. Long inputs use
# the repeat form so that neither the fixture nor the Rust test stores them as a literal.
INPUTS = [
    # Retrieval-prompted English and CJK text.
    {"name": "query_en", "input": {"text": QUERY + "What is machine learning?"}},
    {
        "name": "document_en",
        "input": {
            "text": DOCUMENT + "The quick brown fox isn't lazy; it's 3.14159 times faster—really!"
        },
    },
    {"name": "query_zh", "input": {"text": QUERY + "人工智能是什么？"}},
    {
        "name": "document_zh_en",
        "input": {
            "text": DOCUMENT + "机器学习（machine learning）是人工智能的一个分支。"
            "Deep learning 使用多层神经网络，例如 Transformer。"
        },
    },
    {"name": "query_ja_ko_emoji", "input": {"text": QUERY + "東京の天気は？ 서울 날씨는? 🦀🚀"}},
    {
        "name": "document_whitespace_mix",
        "input": {"text": DOCUMENT + "line one\n\n  line two\t\ttabbed\r\nend   "},
    },
    # Characters outside the vocabulary fall back to byte tokens (<0xF0> ...).
    {
        "name": "byte_fallback",
        "input": {"text": DOCUMENT + "𓀀 ꙮ \u200b\u0007 \U0001F9EC tab\tend"},
    },
    {"name": "empty", "input": {"text": ""}},
    # Added tokens are matched in the text itself. None of them strips adjacent whitespace.
    {"name": "mask_no_space", "input": {"text": QUERY + "The capital of France is<mask>."}},
    {"name": "mask_one_space", "input": {"text": QUERY + "The capital of France is <mask>."}},
    {
        "name": "adjacent_added_tokens",
        "input": {"text": DOCUMENT + "<bos><eos><mask><|turn>user<turn|><pad><|image|>"},
    },
    {
        "name": "unterminated_added_tokens",
        "input": {"text": QUERY + "an unterminated <|turn and <turn| and <"},
    },
    {"name": "uppercase_added_token", "input": {"text": QUERY + "upper case <MASK> is plain text"}},
    # The normalizer turns every space into "▁" before the BPE model, which then merges runs
    # of up to 31 of them: 3,230 tokens for 100,000 spaces.
    {
        "name": "whitespace_run_100k",
        "input": {"repeat": {"prefix": "a", "unit": " ", "count": 100_000, "suffix": "b"}},
    },
    # A long mixed-script document (6,009 tokens).
    {
        "name": "long_zh_en_repeat",
        "input": {
            "repeat": {
                "prefix": DOCUMENT,
                "unit": "机器学习 machine learning ",
                "count": 1500,
                "suffix": ".",
            }
        },
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

    Rows of a batch that contains a long input are padded to its length, so storing them in
    full could make the fixture megabytes long. The Rust test appends the padding back before
    comparing. Raises unless the dropped tail is made up entirely of padding tokens.
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
    if padding is not None and padding["direction"] != "right":
        raise SystemExit(f"expected no padding or right padding, got padding={padding}")

    texts = [build_text(item["input"]) for item in INPUTS]
    singles = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
    batch = tokenizer.encode_batch(texts, add_special_tokens=True)

    cases = []
    for item, single, row in zip(INPUTS, singles, batch, strict=True):
        if len(single.ids) > MAX_LENGTH:
            raise SystemExit(f"{item['name']}: {len(single.ids)} tokens exceeds {MAX_LENGTH}")
        case = {"name": item["name"], "input": item["input"], "single": encoding_fields(single)}
        if padding is None:
            if encoding_fields(row) != case["single"]:
                raise SystemExit(f"{item['name']}: an unpadded batch row differs from encode()")
        else:
            case["batch"] = strip_right_padding(
                encoding_fields(row), padding["pad_id"], padding["pad_type_id"]
            )
        cases.append(case)
        print(f"  OK: {item['name']} ({len(single.ids)} tokens)")

    fixture = {
        "tokenizer": {"repo": REPO, "revision": REVISION, "sha256": sha256},
        "tokenizers_version": tokenizers.__version__,
        "padding": None if padding is None else {
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
