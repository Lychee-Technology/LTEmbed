import csv
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_embedding_benchmarks.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_embedding_benchmarks", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BenchmarkOrchestratorTests(unittest.TestCase):
    def test_scenarios_include_batch_mixed_profile(self):
        bench = load_module()
        scenario = bench.scenario_from_name("batch/mixed/8")

        self.assertEqual(scenario.name, "batch/mixed/8")
        self.assertEqual(scenario.batch_size, 8)
        self.assertEqual(scenario.text_profile, "mixed")

    def test_write_csv_report_uses_fixed_header_order(self):
        bench = load_module()
        row = {field: "" for field in bench.CSV_FIELDNAMES}
        row["implementation"] = "ltembed"
        row["scenario"] = "single/short"
        row["mode"] = "warm_latency"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.csv"
            bench.write_csv_report([row], output_path)
            with output_path.open(newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader)
                values = next(reader)

        self.assertEqual(header, bench.CSV_FIELDNAMES)
        self.assertEqual(values[header.index("implementation")], "ltembed")
        self.assertEqual(values[header.index("scenario")], "single/short")
        self.assertEqual(values[header.index("mode")], "warm_latency")

    def test_build_correctness_row_marks_failure_below_threshold(self):
        bench = load_module()
        row = bench.build_correctness_row(
            base_fields={
                "implementation": "ltembed",
                "scenario": "single/short",
                "mode": "correctness",
            },
            cosine_similarity=0.998,
            threshold=0.999,
        )
        self.assertEqual(row["status"], "fail")
        self.assertEqual(row["cosine_similarity_vs_pytorch"], "0.998000")

    def test_ltembed_warm_command_includes_optional_scenario(self):
        bench = load_module()
        args = type(
            "Args",
            (),
            {
                "model_dir": Path("assets"),
                "ort_bundle_dir": Path("ort_bundle"),
                "output_dimension": 512,
                "l2_normalize": True,
                "warmup": 5,
                "iters": 10,
                "threads": 1,
                "scenario": "single/medium",
            },
        )
        command = bench.ltembed_warm_command(args)
        self.assertIn("--scenario", command)
        self.assertIn("single/medium", command)
        self.assertIn("--ort-bundle-dir", command)
        self.assertIn("--output-dimension", command)
        self.assertIn("--l2-normalize", command)

    def test_ltembed_commands_include_optional_cargo_features(self):
        bench = load_module()
        args = type(
            "Args",
            (),
            {
                "model_dir": Path("assets"),
                "ort_bundle_dir": Path("ort_bundle"),
                "output_dimension": 512,
                "l2_normalize": True,
                "warmup": 5,
                "iters": 10,
                "threads": 1,
                "scenario": None,
                "ltembed_cargo_features": "vendored-blas",
            },
        )

        warm_command = bench.ltembed_warm_command(args)
        cold_command = bench.ltembed_cold_command(args, "single/long")
        correctness_command = bench.ltembed_correctness_command(args)

        for command in (warm_command, cold_command, correctness_command):
            self.assertEqual(
                command[:6],
                ["cargo", "run", "--quiet", "--release", "--features", "vendored-blas"],
            )

    def test_ltembed_commands_use_ort_bundle_contract(self):
        bench = load_module()
        args = type(
            "Args",
            (),
            {
                "model_dir": Path("assets"),
                "ort_bundle_dir": Path("ort_bundle"),
                "output_dimension": 512,
                "l2_normalize": True,
                "warmup": 5,
                "iters": 10,
                "threads": 1,
                "scenario": None,
                "ltembed_cargo_features": "",
            },
        )

        warm_command = bench.ltembed_warm_command(args)
        cold_command = bench.ltembed_cold_command(args, "single/long")
        correctness_command = bench.ltembed_correctness_command(args)

        for command in (warm_command, cold_command, correctness_command):
            self.assertIn("--ort-bundle-dir", command)
            self.assertIn("ort_bundle", command)
            self.assertNotIn("--model-dir", command)

    def test_pytorch_commands_keep_model_dir_contract(self):
        bench = load_module()
        args = type(
            "Args",
            (),
            {
                "model_dir": Path("assets"),
                "output_dimension": 768,
                "l2_normalize": True,
                "warmup": 5,
                "iters": 10,
                "threads": 1,
            },
        )

        warm_command = bench.pytorch_warm_command(args)
        cold_command = bench.pytorch_cold_command(args, "single/long")
        correctness_command = bench.pytorch_correctness_command(args)

        for command in (warm_command, cold_command, correctness_command):
            self.assertIn("--model-name-or-path", command)
            self.assertIn("assets", command)
            self.assertIn("--output-dimension", command)
            self.assertIn("768", command)
            self.assertIn("--l2-normalize", command)
            self.assertIn("true", command)

    def test_benchmark_workflow_downloads_builder_bundle_and_hf_weights(self):
        workflow = (ROOT / ".github" / "workflows" / "benchmark-arm64.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("minimal-ort-builder", workflow)
        self.assertIn("v1.0.13", workflow)
        self.assertIn(
            "jinaai__jina-embeddings-v5-text-nano-retrieval_int8_linux-arm64.tar.gz",
            workflow,
        )
        self.assertIn("snapshot_download(", workflow)
        self.assertIn('--ort-bundle-dir "$ORT_BUNDLE_DIR"', workflow)
        self.assertIn('output_dimension:', workflow)

    def test_benchmark_workflow_installs_cpu_only_pytorch(self):
        workflow = (ROOT / ".github" / "workflows" / "benchmark-arm64.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("https://download.pytorch.org/whl/cpu", workflow)
        self.assertIn("python -m pip install --index-url https://download.pytorch.org/whl/cpu torch", workflow)

    def test_benchmark_workflow_downloads_hf_remote_code_files(self):
        workflow = (ROOT / ".github" / "workflows" / "benchmark-arm64.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn('"*.py"', workflow)

    def test_resolved_notes_is_empty_for_current_runners(self):
        bench = load_module()
        self.assertEqual(bench.resolved_notes("ltembed", {}), "")
        self.assertEqual(bench.resolved_notes("pytorch", {"backend": "ignored"}), "")

    def test_run_json_command_logs_labeled_start_and_finish_messages(self):
        bench = load_module()
        stderr = io.StringIO()

        with (
            mock.patch.object(
                bench.subprocess,
                "run",
                return_value=SimpleNamespace(stdout='{"status":"ok"}'),
            ) as run_mock,
            contextlib.redirect_stderr(stderr),
        ):
            payload = bench.run_json_command(["python3", "tool.py"], "pytorch warm")

        self.assertEqual(payload["status"], "ok")
        self.assertIn("START pytorch warm", stderr.getvalue())
        self.assertIn("DONE pytorch warm", stderr.getvalue())
        run_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
