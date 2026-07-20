import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "write_benchmark_metadata.py"


def load_module():
    spec = importlib.util.spec_from_file_location("write_benchmark_metadata", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


AARCH64_CPUINFO = """\
processor\t: 0
BogoMIPS\t: 2100.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp
CPU implementer\t: 0x41
"""


class WriteBenchmarkMetadataTests(unittest.TestCase):
    def _make_env(self, tmp: Path) -> dict[str, Path]:
        bundle = tmp / "bundle"
        bundle.mkdir()
        (bundle / "model.gguf").write_bytes(b"g" * 1000)
        (bundle / "tokenizer.json").write_bytes(b"t" * 300)
        (bundle / "build-info.json").write_bytes(b"b" * 50)
        static_dir = tmp / "static"
        static_dir.mkdir()
        (static_dir / "build-info.json").write_text(
            json.dumps({"artifact_contract_version": "3", "link_line": []}), encoding="utf-8"
        )
        cpuinfo = tmp / "cpuinfo"
        cpuinfo.write_text(AARCH64_CPUINFO, encoding="utf-8")
        return {"bundle": bundle, "static": static_dir, "cpuinfo": cpuinfo}

    def _argv(self, tmp: Path, env: dict[str, Path], **overrides) -> list[str]:
        options = {
            "--quant": "Q5_K_M",
            "--model-id": "test/model",
            "--model-file": "v5-nano-retrieval-Q5_K_M.gguf",
            "--bundle-dir": str(env["bundle"]),
            "--static-llama-dir": str(env["static"]),
            "--static-llama-tag": "v0.1.151-1",
            "--static-llama-sha256": "abc123",
            "--runner-labels": "ubuntu-24.04-arm",
            "--threads": "1",
            "--warmup-iters": "10",
            "--timed-iters": "100",
            "--cold-iters": "10",
            "--output-dimension": "512",
            "--git-sha": "deadbeef",
            "--cpuinfo-path": str(env["cpuinfo"]),
            "--output": str(tmp / "metadata.json"),
        }
        options.update(overrides)
        argv: list[str] = []
        for key, value in options.items():
            argv.extend([key, value])
        return argv

    def _write(self, tmp: Path, env: dict[str, Path], **overrides) -> dict:
        module = load_module()
        code = module.main(self._argv(tmp, env, **overrides))
        self.assertEqual(code, 0)
        return json.loads((tmp / "metadata.json").read_text(encoding="utf-8"))

    def test_numeric_fields_are_json_numbers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metadata = self._write(tmp, self._make_env(tmp))
        for key in (
            "schema_version",
            "model_size_bytes",
            "bundle_size_bytes",
            "static_llama_contract_version",
            "threads",
            "warmup_iters",
            "timed_iters",
            "cold_iters",
            "output_dimension",
        ):
            self.assertIsInstance(metadata[key], int, key)
        self.assertIsInstance(metadata["l2_normalize"], bool)

    def test_bundle_size_sums_all_bundle_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metadata = self._write(tmp, self._make_env(tmp))
        self.assertEqual(metadata["model_size_bytes"], 1000)
        self.assertEqual(metadata["bundle_size_bytes"], 1000 + 300 + 50)

    def test_model_sha256_matches_content(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metadata = self._write(tmp, self._make_env(tmp))
        self.assertEqual(metadata["model_sha256"], hashlib.sha256(b"g" * 1000).hexdigest())

    def test_cpu_flags_parsed_from_aarch64_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metadata = self._write(tmp, self._make_env(tmp))
        self.assertIn("asimd", metadata["cpu_flags"])
        self.assertIn("aes", metadata["cpu_flags"])

    def test_contract_version_and_static_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metadata = self._write(tmp, self._make_env(tmp))
        self.assertEqual(metadata["static_llama_contract_version"], 3)
        self.assertEqual(metadata["static_llama_tag"], "v0.1.151-1")
        self.assertEqual(metadata["static_llama_sha256"], "abc123")

    def test_scenarios_default_to_all_five(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metadata = self._write(tmp, self._make_env(tmp))
        self.assertEqual(
            metadata["scenarios"],
            ["single/zh", "single/en", "single/medium", "single/long", "batch/medium/8"],
        )

    def test_rejects_non_positive_run_parameters(self):
        import contextlib
        import io

        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            env = self._make_env(tmp)
            for key, bad in (
                ("--cold-iters", "0"),
                ("--timed-iters", "0"),
                ("--threads", "0"),
                ("--output-dimension", "0"),
                ("--warmup-iters", "-1"),
            ):
                with contextlib.redirect_stderr(io.StringIO()), \
                     self.assertRaises(SystemExit, msg=key) as ctx:
                    module.parse_args(self._argv(tmp, env, **{key: bad}))
                self.assertEqual(ctx.exception.code, 2)
            # warmup of zero is legitimate
            args = module.parse_args(self._argv(tmp, env, **{"--warmup-iters": "0"}))
            self.assertEqual(args.warmup_iters, 0)

    def test_identity_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metadata = self._write(tmp, self._make_env(tmp))
        self.assertEqual(metadata["backend"], "llama.cpp")
        self.assertEqual(metadata["quant"], "Q5_K_M")
        self.assertEqual(metadata["model_file"], "v5-nano-retrieval-Q5_K_M.gguf")
        self.assertEqual(metadata["git_sha"], "deadbeef")
        self.assertEqual(metadata["runner_labels"], "ubuntu-24.04-arm")


if __name__ == "__main__":
    unittest.main()
