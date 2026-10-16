"""Tests for the reference-data generators and the pins they share with CI.

The golden fixture (scripts/generate_fixtures.py) and the llama.cpp reference
(scripts/generate_llama_cpp_reference.py) are generated offline from pinned model files and
a pinned static llama.cpp release. These tests check the generators' pure helpers and that
the committed fixtures, the generators and the CI pins still name the same artifacts. They
also check that the other files repeating CI's static llama.cpp pins, the llama.cpp notice
and the ARM64 benchmark workflow, name the same release.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FIXTURES = ROOT / "tests" / "fixtures"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
BENCHMARK_WORKFLOW = ROOT / ".github" / "workflows" / "benchmark-arm64.yml"
LLAMA_CPP_NOTICE = ROOT / "licenses" / "llama.cpp" / "NOTICE.md"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def ci_env(name):
    values = re.findall(rf"^  {name}: (\S+)$", CI_WORKFLOW.read_text(encoding="utf-8"), re.M)
    assert len(values) == 1, (name, values)
    return values[0]


class GenerateFixturesTests(unittest.TestCase):
    def test_cut_to_tokens_finds_the_longest_prefix_with_the_exact_count(self):
        generate = load("generate_fixtures")

        def count_tokens(text):  # one token per word, plus BOS and EOS
            return len(text.split()) + 2

        base = "alpha beta gamma delta epsilon zeta"
        self.assertEqual(generate.cut_to_tokens(count_tokens, "p: ", base, 5), "alpha beta")
        with self.assertRaisesRegex(ValueError, "shorter"):
            generate.cut_to_tokens(count_tokens, "p: ", base, 100)

    def test_cut_to_tokens_rejects_an_unreachable_count(self):
        generate = load("generate_fixtures")

        def count_tokens(text):  # jumps from 3 to 5: no prefix has exactly 4
            return 3 if len(text) < 10 else 5

        with self.assertRaisesRegex(ValueError, "exactly 4"):
            generate.cut_to_tokens(count_tokens, "", "0123456789abcdef", 4)

    def test_float32_list_round_trips_float32(self):
        generate = load("generate_fixtures")
        values = np.array([0.1, -1e-8, 3.4028234e38, 1 / 3], dtype=np.float64)
        written = generate.float32_list(values)
        np.testing.assert_array_equal(np.array(written, dtype=np.float32),
                                      values.astype(np.float32))

    def test_format_json_round_trips_with_one_line_per_vector(self):
        generate = load("generate_fixtures")
        data = {"fixtures": [{"name": "x", "embedding": [0.5, -0.25]}], "dims": [768, 512]}
        text = generate.format_json(data)
        self.assertEqual(json.loads(text), data)
        self.assertIn('"embedding": [0.5, -0.25]', text)

    def test_check_local_revision_requires_the_pinned_revision(self):
        generate = load("generate_fixtures")
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            with self.assertRaisesRegex(SystemExit, "not found"):
                generate.check_local_revision(model_dir)
            metadata = model_dir / ".cache/huggingface/download/model.safetensors.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text("0" * 40 + "\nhash\n123\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "expected"):
                generate.check_local_revision(model_dir)
            metadata.write_text(generate.REVISION + "\nhash\n123\n", encoding="utf-8")
            generate.check_local_revision(model_dir)

    def test_cases_have_unique_names_and_relevant_documents_exist(self):
        generate = load("generate_fixtures")
        names = [case["name"] for case in generate.SHORT_CASES]
        self.assertEqual(len(names), len(set(names)))
        documents = {c["name"] for c in generate.SHORT_CASES if c["kind"] == "document"}
        for case in generate.SHORT_CASES:
            self.assertIn(case["kind"], ("query", "document"))
            self.assertTrue(case["name"].startswith(case["kind"]), case["name"])
            if "relevant" in case:
                self.assertEqual(case["kind"], "query")
                self.assertIn(case["relevant"], documents)


class LlamaCppReferenceTests(unittest.TestCase):
    def test_parse_tool_output_reads_the_token_ids_and_the_one_embedding(self):
        reference = load("generate_llama_cpp_reference")
        stdout = (
            "main: prompt 0: 'title: none | text: hi'\n"
            "main: number of tokens in prompt = 3\n"
            "     2 -> '<bos>'\n"
            " 15218 -> 'title'\n"
            "     1 -> '<eos>'\n"
            "\n\n"
            '{\n  "object": "list",\n  "data": [\n    {"object": "embedding", "index": 0,'
            ' "embedding": [1.5, -2.25]}\n  ]\n}\n'
        )
        self.assertEqual(reference.parse_tool_output(stdout), ([2, 15218, 1], [1.5, -2.25]))
        two = stdout.replace('"embedding": [1.5, -2.25]}',
                             '"embedding": [1.5]}, {"object": "embedding", "embedding": [2.0]}')
        with self.assertRaisesRegex(ValueError, "one embedding"):
            reference.parse_tool_output(two)

    def test_run_tool_refuses_an_input_ending_in_a_newline(self):
        reference = load("generate_llama_cpp_reference")
        with self.assertRaisesRegex(ValueError, "newline"):
            reference.run_tool(Path("/nonexistent"), Path("/nonexistent"), "text\n", 1)

    def test_tool_arguments_match_the_engine_context(self):
        reference = load("generate_llama_cpp_reference")
        args = reference.TOOL_ARGS
        pairs = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("-")}
        self.assertEqual(pairs["--pooling"], "mean")
        self.assertEqual(pairs["--attention"], "non-causal")
        self.assertEqual(pairs["--flash-attn"], "on")
        self.assertEqual(pairs["--embd-normalize"], "-1")
        for flag in ("-c", "-b", "-ub"):
            self.assertEqual(pairs[flag], str(reference.MAX_LENGTH))
        self.assertIn("--no-escape", args)

    def generate(self, reference, existing, tokens, reset, fail_if_run=False):
        """generate() of the x86_64-v3 section for one input whose tool run gives `tokens`,
        without the tool. Returns its result and the texts the tool was given."""
        golden = {
            "prompts": {"query": "task: search result | query: ", "document": "d: "},
            "raw_dim": 2,
            "fixtures": [{"name": "query_en", "kind": "query", "text": "What is ML? hello",
                          "tokens": len(tokens)}],
        }
        runs = []

        def embed(text):
            if fail_if_run:
                self.fail(f"the tool ran on {text!r}")
            runs.append(text)
            return list(tokens), [0.5, -1.25]

        with contextlib.redirect_stdout(io.StringIO()):
            result = reference.generate(golden, existing, "x86_64-v3", embed, reset)
        return result, runs

    @staticmethod
    def existing_reference(token_ids, variants=("aarch64-graviton2", "x86_64-v3"),
                           name="query_en"):
        section = {"static_release": "old", "archive_sha256": "old",
                   "embeddings": [{"name": name, "embedding": [1.0, 2.0]}]}
        return {"inputs": [{"name": name, "token_ids": token_ids}],
                "variants": {variant: dict(section) for variant in variants}}

    def test_reset_discards_the_token_ids_of_the_reference_it_replaces(self):
        reference = load("generate_llama_cpp_reference")
        existing = self.existing_reference([2, 100, 1])
        (inputs, variants), _ = self.generate(reference, existing, [2, 100, 7, 1], reset=True)
        self.assertEqual(inputs, [{"name": "query_en", "token_ids": [2, 100, 7, 1]}])
        self.assertEqual(sorted(variants), ["x86_64-v3"])
        self.assertEqual(variants["x86_64-v3"]["embeddings"],
                         [{"name": "query_en", "embedding": [0.5, -1.25]}])

    def test_without_reset_a_kept_section_must_have_the_same_token_ids(self):
        reference = load("generate_llama_cpp_reference")
        existing = self.existing_reference([2, 100, 1])
        with self.assertRaisesRegex(SystemExit, "query_en: token ids differ from the "
                                                "aarch64-graviton2 run.*--reset"):
            self.generate(reference, existing, [2, 100, 7, 1], reset=False)
        (inputs, variants), _ = self.generate(reference, existing, [2, 100, 1], reset=False)
        self.assertEqual(inputs, existing["inputs"])
        self.assertEqual(variants["aarch64-graviton2"], existing["variants"]["aarch64-graviton2"])
        self.assertEqual(variants["x86_64-v3"]["static_release"], reference.STATIC_RELEASE)

    def test_without_another_section_nothing_constrains_the_token_ids(self):
        # Regenerating the only section replaces it, token ids included.
        reference = load("generate_llama_cpp_reference")
        existing = self.existing_reference([2, 100, 1], variants=("x86_64-v3",))
        (inputs, variants), _ = self.generate(reference, existing, [2, 100, 7, 1], reset=False)
        self.assertEqual(inputs, [{"name": "query_en", "token_ids": [2, 100, 7, 1]}])
        self.assertEqual(sorted(variants), ["x86_64-v3"])

    def test_a_section_from_other_inputs_fails_before_the_tool_runs(self):
        reference = load("generate_llama_cpp_reference")
        existing = self.existing_reference([2, 100, 1], name="query_old")
        with self.assertRaisesRegex(SystemExit, "aarch64-graviton2 section was generated "
                                                "from other inputs"):
            self.generate(reference, existing, [2, 100, 1], reset=False, fail_if_run=True)
        _, runs = self.generate(reference, existing, [2, 100, 1], reset=True)
        self.assertEqual(runs, ["task: search result | query: What is ML? hello"])

    def test_static_release_pins_match_ci(self):
        reference = load("generate_llama_cpp_reference")
        self.assertEqual(reference.STATIC_RELEASE, ci_env("STATIC_LLAMA_TAG"))
        self.assertEqual(reference.LLAMA_CPP_COMMIT, ci_env("STATIC_LLAMA_LLAMA_CPP_COMMIT"))
        self.assertEqual(reference.VARIANTS["aarch64-graviton2"],
                         ci_env("STATIC_LLAMA_SHA256_AARCH64"))
        self.assertEqual(reference.VARIANTS["x86_64-v3"], ci_env("STATIC_LLAMA_SHA256_X86_64"))


class PinConsistencyTests(unittest.TestCase):
    """The committed fixtures, the generators and the bundle pins name the same files."""

    def setUp(self):
        self.pins = load("assemble_gguf_bundle")

    def test_golden_fixture_comes_from_the_bundle_tokenizer_revision(self):
        generate = load("generate_fixtures")
        golden = json.loads((FIXTURES / "test_fixtures.json").read_text(encoding="utf-8"))
        self.assertEqual((generate.REPO, generate.REVISION),
                         (self.pins.TOKENIZER_REPO, self.pins.TOKENIZER_REVISION))
        self.assertEqual(golden["model"],
                         {"repo": generate.REPO, "revision": generate.REVISION,
                          "upstream": self.pins.UPSTREAM_MODEL})
        self.assertEqual(golden["prompts"], {"query": self.pins.QUERY_PREFIX,
                                             "document": self.pins.DOCUMENT_PREFIX})
        self.assertEqual(golden["raw_dim"], self.pins.RAW_EMBEDDING_DIMENSION)
        self.assertEqual(golden["max_length"], self.pins.MAX_LENGTH)
        self.assertEqual(golden["reference"]["dtype"], "float32")
        self.assertEqual([f["name"] for f in golden["fixtures"]][:len(generate.SHORT_CASES)],
                         [case["name"] for case in generate.SHORT_CASES])

    def test_token_id_fixture_comes_from_the_bundle_tokenizer(self):
        generate = load("generate_token_ids")
        fixture = json.loads((FIXTURES / "token_ids.json").read_text(encoding="utf-8"))
        self.assertEqual((generate.REPO, generate.REVISION),
                         (self.pins.TOKENIZER_REPO, self.pins.TOKENIZER_REVISION))
        self.assertEqual(fixture["tokenizer"]["sha256"], self.pins.TOKENIZER_SHA256)

    def test_llama_cpp_reference_comes_from_the_release_gguf(self):
        reference = load("generate_llama_cpp_reference")
        filename, digest = self.pins.GGUF_FILES[self.pins.DEFAULT_QUANT]
        self.assertEqual(reference.GGUF, {"repo": self.pins.GGUF_REPO,
                                          "revision": self.pins.GGUF_REVISION,
                                          "file": filename, "sha256": digest})
        committed = json.loads((FIXTURES / "llama_cpp_reference.json").read_text(encoding="utf-8"))
        self.assertEqual(committed["gguf"], reference.GGUF)
        self.assertEqual(committed["llama_cpp"]["commit"], reference.LLAMA_CPP_COMMIT)
        self.assertEqual(committed["llama_cpp"]["arguments"], reference.TOOL_ARGS)
        self.assertEqual(sorted(committed["variants"]), sorted(reference.VARIANTS))
        for variant, section in committed["variants"].items():
            self.assertEqual(section["static_release"], reference.STATIC_RELEASE, variant)
            self.assertEqual(section["archive_sha256"], reference.VARIANTS[variant], variant)


class BenchmarkWorkflowPinTests(unittest.TestCase):
    """benchmark-arm64.yml sets its own copy of ci.yml's static llama.cpp pins."""

    def test_benchmark_workflow_pins_the_ci_static_release(self):
        pins = {}
        for name, value in re.findall(r"^ +(STATIC_LLAMA_\w+): *(.*?) *$",
                                      BENCHMARK_WORKFLOW.read_text(encoding="utf-8"), re.M):
            pins.setdefault(name, []).append(value)
        # It runs on ARM64 runners only, so it pins no x86_64 tarball.
        self.assertEqual(sorted(pins), ["STATIC_LLAMA_LLAMA_CPP_COMMIT", "STATIC_LLAMA_REPO",
                                        "STATIC_LLAMA_SHA256_AARCH64", "STATIC_LLAMA_TAG"])
        for name, values in pins.items():
            self.assertEqual(values, [ci_env(name)], name)


class LlamaCppNoticeTests(unittest.TestCase):
    """licenses/llama.cpp/NOTICE.md attributes the static release that CI links against."""

    def setUp(self):
        self.notice = LLAMA_CPP_NOTICE.read_text(encoding="utf-8")

    def test_notice_names_the_ci_static_release_and_llama_cpp_commit(self):
        tags = set(re.findall(r"release \*\*`([^`]+)`\*\*", self.notice))
        tags |= set(re.findall(r"static-llama-cpp-rs-builder/releases/tag/([^>\s]+)", self.notice))
        self.assertEqual(tags, {ci_env("STATIC_LLAMA_TAG")})
        commits = set(re.findall(r"at commit\s+\*\*`([0-9a-f]{40})`\*\*", self.notice))
        commits |= set(re.findall(r"ggml-org/llama\.cpp/(?:tree/)?([0-9a-f]{40})", self.notice))
        self.assertEqual(commits, {ci_env("STATIC_LLAMA_LLAMA_CPP_COMMIT")})

    def test_license_copies_have_the_sha256_the_notice_records(self):
        recorded = dict(re.findall(r"\[(LICENSE-[\w-]+\.txt)\]\(\./\1\), sha256\s+`([0-9a-f]{64})`",
                                   self.notice))
        self.assertEqual(sorted(recorded), ["LICENSE-BUILDER-MIT.txt", "LICENSE-MIT.txt"])
        for name, digest in recorded.items():
            actual = hashlib.sha256((LLAMA_CPP_NOTICE.parent / name).read_bytes()).hexdigest()
            self.assertEqual(actual, digest, name)


if __name__ == "__main__":
    unittest.main()
