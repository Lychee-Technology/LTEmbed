import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "generate_token_ids.py"
PINS_PATH = ROOT / "scripts" / "assemble_gguf_bundle.py"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "token_ids.json"
CARGO_LOCK_PATH = ROOT / "Cargo.lock"


def load_module(path=MODULE_PATH, name="generate_token_ids"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def row(input_ids, attention_mask, type_ids):
    return {"input_ids": input_ids, "attention_mask": attention_mask, "type_ids": type_ids}


class GenerateTokenIdsTests(unittest.TestCase):
    def test_build_text_expands_repeat_spec(self):
        generate = load_module()
        spec = {"repeat": {"prefix": "a", "unit": " ", "count": 3, "suffix": "b"}}
        self.assertEqual(generate.build_text(spec), "a   b")
        self.assertEqual(generate.build_text({"text": "query: hi"}), "query: hi")

    def test_locked_tokenizers_version_reads_the_tokenizers_package(self):
        generate = load_module()
        lock = (
            '[[package]]\nname = "tokenizers"\nversion = "0.23.2"\n\n'
            '[[package]]\nname = "tokenizers-extra"\nversion = "1.0.0"\n'
        )
        self.assertEqual(generate.locked_tokenizers_version(lock), "0.23.2")
        two_versions = lock + lock.replace("0.23.2", "0.22.2")
        for bad in ("", two_versions):
            with self.assertRaises(ValueError):
                generate.locked_tokenizers_version(bad)
        # The real Cargo.lock must parse too, or regenerating the fixture would fail.
        self.assertRegex(
            generate.locked_tokenizers_version(CARGO_LOCK_PATH.read_text(encoding="utf-8")),
            r"^\d+\.\d+\.\d+",
        )

    def test_strip_right_padding_keeps_real_tokens_and_counts_padding(self):
        generate = load_module()
        padded = row([5, 6, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0])
        self.assertEqual(
            generate.strip_right_padding(padded, pad_id=0, pad_type_id=0),
            row([5, 6], [1, 1], [0, 0]) | {"padding": 2},
        )

    def test_strip_right_padding_keeps_unmasked_pad_token(self):
        generate = load_module()
        # A literal <pad> in the text is a real token: only the attention mask marks padding.
        unpadded = row([5, 0], [1, 1], [0, 0])
        self.assertEqual(
            generate.strip_right_padding(unpadded, pad_id=0, pad_type_id=0),
            unpadded | {"padding": 0},
        )

    def test_strip_right_padding_rejects_tail_that_is_not_padding(self):
        generate = load_module()
        for bad in (
            row([5, 7], [1, 0], [0, 0]),  # masked token is not the pad id
            row([5, 0], [1, 0], [0, 1]),  # masked token has the wrong type id
            row([0, 5], [0, 1], [0, 0]),  # left padding
        ):
            with self.assertRaises(ValueError):
                generate.strip_right_padding(bad, pad_id=0, pad_type_id=0)

    def test_format_json_round_trips_with_one_line_per_id_list(self):
        generate = load_module()
        data = {"cases": [{"name": "x", "single": row([1, 2], [1, 1], [0, 0])}], "empty": {}}
        text = generate.format_json(data)
        self.assertEqual(json.loads(text), data)
        self.assertIn('"input_ids": [1, 2]', text)

    def test_committed_fixture_matches_generator_and_bundle_pins(self):
        generate = load_module()
        pins = load_module(PINS_PATH, "assemble_gguf_bundle")
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(pins.TOKENIZER_REPO, generate.REPO)
        self.assertEqual(pins.TOKENIZER_REVISION, generate.REVISION)
        self.assertEqual(pins.TOKENIZER_SHA256, fixture["tokenizer"]["sha256"])
        self.assertEqual(fixture["tokenizer"]["repo"], generate.REPO)
        self.assertEqual(fixture["tokenizer"]["revision"], generate.REVISION)
        self.assertEqual(
            [(case["name"], case["input"]) for case in fixture["cases"]],
            [(item["name"], item["input"]) for item in generate.INPUTS],
        )

    def test_committed_fixture_has_no_padding_and_fits_the_engine(self):
        generate = load_module()
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        # EmbeddingGemma 2's tokenizer.json pads nothing, so no case stores a batch row.
        self.assertIsNone(fixture["padding"])
        for case in fixture["cases"]:
            self.assertNotIn("batch", case, case["name"])
            single = case["single"]
            self.assertLessEqual(len(single["input_ids"]), generate.MAX_LENGTH, case["name"])
            # Every encoding starts with <bos> (2) and ends with <eos> (1).
            self.assertEqual(single["input_ids"][0], 2, case["name"])
            self.assertEqual(single["input_ids"][-1], 1, case["name"])
            self.assertEqual(set(single["attention_mask"]), {1}, case["name"])


if __name__ == "__main__":
    unittest.main()
