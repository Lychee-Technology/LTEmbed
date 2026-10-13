import contextlib
import hashlib
import importlib.util
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "assemble_gguf_bundle.py"
ENGINE_PATH = ROOT / "src" / "engine" / "mod.rs"
WORKFLOWS = ROOT / ".github" / "workflows"


def load_module():
    spec = importlib.util.spec_from_file_location("assemble_gguf_bundle", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class AssembleTests(unittest.TestCase):
    """assemble() on small local files, with the pins patched to their hashes."""

    def setUp(self):
        self.module = load_module()
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.gguf = self.tmp / "model-source.gguf"
        self.gguf.write_bytes(b"GGUF fake weights")
        self.tokenizer = self.tmp / "tokenizer-source.json"
        self.tokenizer.write_bytes(b'{"model": {}}')
        quant = self.module.DEFAULT_QUANT
        self.enterContext(mock.patch.dict(
            self.module.GGUF_FILES,
            {quant: (f"embeddinggemma-2-{quant}.gguf", sha256(self.gguf.read_bytes()))},
        ))
        self.enterContext(mock.patch.object(
            self.module, "TOKENIZER_SHA256", sha256(self.tokenizer.read_bytes())
        ))
        self.enterContext(mock.patch.dict("os.environ", {"GITHUB_SHA": "abc123"}))

    def assemble(self, destination, **kwargs):
        return self.module.assemble(destination, self.module.DEFAULT_QUANT,
                                    kwargs.get("gguf", self.gguf))

    def main(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = self.module.main([str(arg) for arg in argv])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_writes_exactly_the_bundle_files_with_their_checksums(self):
        bundle = self.tmp / "bundle"
        hashes = self.assemble(bundle)
        self.assertEqual(sorted(p.name for p in bundle.iterdir()), self.module.OUTPUT_FILES)
        self.assertEqual((bundle / "model.gguf").read_bytes(), self.gguf.read_bytes())
        # The engine tokenizes with the GGUF's own vocabulary.
        self.assertNotIn("tokenizer.json", self.module.OUTPUT_FILES)
        self.assertEqual(
            json.loads((bundle / "build-info.json").read_text(encoding="utf-8")),
            self.module.build_info(self.module.DEFAULT_QUANT),
        )
        sums = (bundle / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        self.assertEqual(sums, [f"{hashes[n]}  {n}" for n in self.module.BUNDLE_FILES])
        for name in self.module.BUNDLE_FILES:
            self.assertEqual(hashes[name], sha256((bundle / name).read_bytes()))

    def test_provenance_records_sources_hashes_and_commit(self):
        bundle = self.tmp / "bundle"
        hashes = self.assemble(bundle)
        provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["ltembed_commit"], "abc123")
        self.assertEqual(provenance["model"]["upstream"], "google/embeddinggemma-2")
        model = provenance["files"]["model.gguf"]
        self.assertEqual(model["repo"], self.module.GGUF_REPO)
        self.assertEqual(model["revision"], self.module.GGUF_REVISION)
        self.assertEqual(model["sha256"], hashes["model.gguf"])
        self.assertIn(f"/resolve/{self.module.GGUF_REVISION}/", model["url"])
        self.assertEqual(provenance["files"]["build-info.json"]["sha256"],
                         hashes["build-info.json"])
        self.assertEqual(sorted(provenance["files"]), sorted(self.module.BUNDLE_FILES))

    def test_hash_mismatch_fails_and_leaves_no_bundle(self):
        other = self.tmp / "other.gguf"
        other.write_bytes(b"not the pinned weights")
        bundle = self.tmp / "bundle"
        with self.assertRaisesRegex(ValueError, "expected the pinned"):
            self.assemble(bundle, gguf=other)
        self.assertFalse(bundle.exists())
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()),
                         ["model-source.gguf", "other.gguf", "tokenizer-source.json"])

    def test_refuses_a_non_empty_destination(self):
        bundle = self.tmp / "bundle"
        bundle.mkdir()
        (bundle / "mmproj-BF16.gguf").write_bytes(b"projector")
        with self.assertRaisesRegex(ValueError, "not empty"):
            self.assemble(bundle)

    def test_main_reports_errors_without_a_traceback(self):
        code, _, stderr = self.main(self.tmp / "b", "--gguf", self.tokenizer)
        self.assertEqual(code, 1)
        self.assertIn("error: model.gguf", stderr)

    def test_reference_tokenizer_is_written_outside_the_bundle(self):
        bundle, reference = self.tmp / "bundle", self.tmp / "ref" / "tokenizer.json"
        code, stdout, stderr = self.main(bundle, "--gguf", self.gguf,
                                         "--reference-tokenizer", reference,
                                         "--tokenizer", self.tokenizer)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(reference.read_bytes(), self.tokenizer.read_bytes())
        self.assertEqual(sorted(p.name for p in bundle.iterdir()), self.module.OUTPUT_FILES)
        self.assertEqual(sorted(p.name for p in reference.parent.iterdir()), ["tokenizer.json"])
        self.assertIn(f"{self.module.TOKENIZER_SHA256}  {reference}", stdout)

    def test_reference_tokenizer_inside_the_bundle_is_refused(self):
        bundle = self.tmp / "bundle"
        code, _, stderr = self.main(bundle, "--gguf", self.gguf,
                                    "--reference-tokenizer", bundle / "tokenizer.json",
                                    "--tokenizer", self.tokenizer)
        self.assertEqual(code, 1)
        self.assertIn("must be outside the bundle", stderr)
        self.assertFalse(bundle.exists())

    def test_tokenizer_needs_reference_tokenizer(self):
        bundle = self.tmp / "bundle"
        code, _, stderr = self.main(bundle, "--gguf", self.gguf, "--tokenizer", self.tokenizer)
        self.assertEqual(code, 1)
        self.assertIn("needs --reference-tokenizer", stderr)
        self.assertFalse(bundle.exists())

    def test_reference_tokenizer_mismatch_keeps_the_previous_file(self):
        reference = self.tmp / "reference-tokenizer.json"
        reference.write_bytes(b"previous")
        other = self.tmp / "other-tokenizer.json"
        other.write_bytes(b'{"model": {"other": 1}}')
        with self.assertRaisesRegex(ValueError, "expected the pinned"):
            self.module.fetch_reference_tokenizer(reference, other)
        self.assertEqual(reference.read_bytes(), b"previous")
        self.module.fetch_reference_tokenizer(reference, self.tokenizer)
        self.assertEqual(reference.read_bytes(), self.tokenizer.read_bytes())
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()),
                         ["model-source.gguf", "other-tokenizer.json",
                          "reference-tokenizer.json", "tokenizer-source.json"])


class PinTests(unittest.TestCase):
    def test_build_info_matches_the_engine_contract(self):
        module = load_module()
        source = ENGINE_PATH.read_text(encoding="utf-8")

        def constant(name):
            match = re.search(rf"pub const {name}: [^=]+= (.+);", source)
            self.assertIsNotNone(match, name)
            return json.loads(match.group(1))

        metadata = module.build_info(module.DEFAULT_QUANT)["model_metadata"]
        self.assertEqual(metadata["query_prefix"], constant("QUERY_PREFIX"))
        self.assertEqual(metadata["document_prefix"], constant("DOCUMENT_PREFIX"))
        self.assertEqual(metadata["raw_embedding_dimension"], constant("RAW_EMBEDDING_DIMENSION"))
        self.assertEqual(metadata["output_embedding_dimension"],
                         constant("RAW_EMBEDDING_DIMENSION"))
        self.assertEqual(metadata["max_length"], constant("MAX_LENGTH"))
        self.assertEqual(metadata["pooling"], "mean")

    def test_build_info_is_the_documented_one_for_the_release_quant(self):
        module = load_module()
        self.assertEqual(module.DEFAULT_QUANT, "UD-Q5_K_XL")
        self.assertEqual(module.build_info("UD-Q5_K_XL"), {
            "target_id": "unsloth/embeddinggemma-2-GGUF",
            "model_metadata": {
                "model_format": "gguf",
                "quant": "UD-Q5_K_XL",
                "pooling": "mean",
                "input_kind": "retrieval",
                "query_prefix": "task: search result | query: ",
                "document_prefix": "title: none | text: ",
                "raw_embedding_dimension": 768,
                "output_embedding_dimension": 768,
                "max_length": 8192,
            },
        })

    def test_pins_are_commits_and_hashes_and_never_a_projector(self):
        module = load_module()
        for revision in (module.GGUF_REVISION, module.TOKENIZER_REVISION):
            self.assertRegex(revision, r"^[0-9a-f]{40}$")
        for quant, (filename, digest) in module.GGUF_FILES.items():
            self.assertEqual(filename, f"embeddinggemma-2-{quant}.gguf")
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertNotIn("mmproj", filename.lower())
        self.assertNotIn("F16", module.GGUF_FILES)
        self.assertRegex(module.TOKENIZER_SHA256, r"^[0-9a-f]{64}$")
        with self.assertRaisesRegex(ValueError, "unknown quant"):
            module.build_info("F16")

    def test_workflows_assemble_bundles_with_this_script_only(self):
        for name in ("ci.yml", "release-bundles.yml", "benchmark-arm64.yml"):
            workflow = (WORKFLOWS / name).read_text(encoding="utf-8")
            self.assertIn("scripts/assemble_gguf_bundle.py", workflow, name)
            self.assertNotIn("huggingface.co/", workflow, name)
            self.assertNotIn("resolve/main", workflow, name)


if __name__ == "__main__":
    unittest.main()
