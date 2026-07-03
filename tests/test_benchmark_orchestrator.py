import csv
import contextlib
import importlib.util
import io
import json
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

    def test_build_benchmark_command_passes_optional_scenario_for_warm(self):
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
        command = bench.build_benchmark_command("ltembed", "warm", args)
        self.assertIn("--scenario", command)
        self.assertIn("single/medium", command)
        self.assertIn("--ort-bundle-dir", command)
        self.assertIn("--output-dimension", command)
        self.assertIn("--l2-normalize", command)

    def test_build_benchmark_command_includes_cargo_features_for_ltembed(self):
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

        warm_command = bench.build_benchmark_command("ltembed", "warm", args)
        cold_command = bench.build_benchmark_command("ltembed", "cold", args, "single/long")
        correctness_command = bench.build_benchmark_command("ltembed", "correctness", args)

        for command in (warm_command, cold_command, correctness_command):
            self.assertEqual(
                command[:6],
                ["cargo", "run", "--quiet", "--release", "--features", "vendored-blas"],
            )

    def test_build_benchmark_command_uses_ort_bundle_for_ltembed(self):
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

        warm_command = bench.build_benchmark_command("ltembed", "warm", args)
        cold_command = bench.build_benchmark_command("ltembed", "cold", args, "single/long")
        correctness_command = bench.build_benchmark_command("ltembed", "correctness", args)

        for command in (warm_command, cold_command, correctness_command):
            self.assertIn("--ort-bundle-dir", command)
            self.assertIn("ort_bundle", command)
            self.assertNotIn("--model-dir", command)

    def test_build_benchmark_command_uses_model_dir_for_pytorch(self):
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

        warm_command = bench.build_benchmark_command("pytorch", "warm", args)
        cold_command = bench.build_benchmark_command("pytorch", "cold", args, "single/long")
        correctness_command = bench.build_benchmark_command("pytorch", "correctness", args)

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
        self.assertIn("v1.0.15", workflow)
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

    def test_benchmark_workflow_enables_ltembed_stage_profiling(self):
        workflow = (ROOT / ".github" / "workflows" / "benchmark-arm64.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn('export LTEMBED_PROFILE="1"', workflow)


    def test_build_benchmark_command_passes_custom_threads_for_ltembed(self):
        bench = load_module()
        args = SimpleNamespace(
            ort_bundle_dir=Path("ort_bundle"),
            output_dimension=512,
            l2_normalize=True,
            warmup=5,
            iters=10,
            threads=4,
            scenario=None,
            ltembed_cargo_features="",
        )
        command = bench.build_benchmark_command("ltembed", "warm", args)
        self.assertIn("--threads", command)
        self.assertIn("4", command)

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


    def test_collect_retrieval_eval_rows_produces_correct_csv_row(self):
        bench = load_module()

        retrieval_json = {
            "cases": [
                {
                    "name": "mini-retrieval-v1",
                    "documents": [
                        {"id": "d1", "text": "Rust ownership protects memory safety."},
                        {"id": "d2", "text": "Java uses a garbage collector."},
                    ],
                    "queries": [
                        {
                            "id": "q1",
                            "text": "How does Rust avoid a garbage collector?",
                            "relevant_document_ids": ["d1"],
                        },
                        {
                            "id": "q2",
                            "text": "What supports nearest-neighbor search?",
                            "relevant_document_ids": ["d2"],
                        },
                    ],
                }
            ]
        }

        mock_payload = {
            "implementation": "ltembed",
            "implementation_version": "abc123",
            "results": [
                {
                    "dataset_name": "mini-retrieval-v1",
                    "queries": [
                        {"id": "q1", "embedding": [0.0, 1.0, 0.0]},
                        {"id": "q2", "embedding": [0.0, 1.0, 0.0]},
                    ],
                    "documents": [
                        {"id": "d1", "embedding": [0.0, 0.0, 1.0]},
                        {"id": "d2", "embedding": [0.0, 1.0, 0.0]},
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            retrieval_path = tmp / "retrieval_eval_cases.json"
            retrieval_path.write_text(json.dumps(retrieval_json), encoding="utf-8")

            args = SimpleNamespace(
                run_id="run-1",
                model_id="test-model",
                model_source="huggingface",
                retrieval_eval_path=retrieval_path,
                threads=1,
                warmup=0,
                iters=0,
            )

            host = {
                "host_os": "linux",
                "host_arch": "arm64",
                "cpu_model": "test-cpu",
                "runner_labels": "",
            }

            with mock.patch.dict(
                bench.RUNNERS,
                {
                    "ltembed": {
                        "retrieval": lambda _a: ["ltembed-retrieval-cmd"],
                        "version": lambda: "abc123",
                    },
                    "pytorch": {
                        "retrieval": lambda _a: ["pytorch-retrieval-cmd"],
                        "version": lambda: "",
                    },
                },
            ), mock.patch.object(
                bench,
                "run_json_command",
                return_value=mock_payload,
            ) as run_mock:
                ctx = bench.RunContext(
                    run_id="run-1",
                    timestamp_utc="2026-01-01T00:00:00+00:00",
                    model_id=args.model_id,
                    model_source=args.model_source,
                    git_revision="abc123",
                    host=host,
                )
                rows, payloads = bench.collect_retrieval_eval_rows(args=args, ctx=ctx)

        self.assertEqual(len(rows), 2)
        self.assertEqual(run_mock.call_count, 2)

        labels = [call_args[0][1] for call_args in run_mock.call_args_list]
        self.assertIn("ltembed retrieval", labels)
        self.assertIn("pytorch retrieval", labels)

        row = rows[0]
        self.assertEqual(row["implementation"], "ltembed")
        self.assertEqual(row["scenario"], "mini-retrieval-v1")
        self.assertEqual(row["mode"], "retrieval_eval")
        self.assertEqual(row["batch_size"], "2")
        self.assertEqual(row["text_profile"], "retrieval_eval")
        self.assertEqual(row["mean_ms"], "")
        self.assertEqual(row["query_count"], "2")
        self.assertEqual(row["recall_at_1"], "0.500000")
        self.assertEqual(row["recall_at_3"], "1.000000")
        self.assertEqual(row["mrr_at_3"], "0.750000")
        self.assertEqual(row["cosine_similarity_vs_pytorch"], "")

    def test_compute_retrieval_metrics_zeroes_reciprocal_rank_beyond_3(self):
        bench = load_module()

        retrieval_case = {
            "queries": [
                {"id": "q1", "relevant_document_ids": ["d_rel"]},
            ],
        }
        metrics = bench.compute_retrieval_metrics(
            retrieval_case,
            query_embeddings={"q1": [1.0, 0.0, 0.0]},
            document_embeddings={
                "d1": [1.0, 0.0, 0.0],
                "d2": [0.9, 0.1, 0.0],
                "d3": [0.8, 0.2, 0.0],
                "d_rel": [0.0, 1.0, 0.0],
            },
        )

        self.assertEqual(metrics["query_count"], 1)
        self.assertEqual(metrics["recall_at_1"], 0.0)
        self.assertEqual(metrics["recall_at_3"], 0.0)
        self.assertEqual(metrics["mrr_at_3"], 0.0)

    # ── Characterization: golden-assert _run output ───────────────

    def test_golden_run_produces_expected_row_counts_and_summary(self):
        bench = load_module()

        warm_stats = {
            "mean_ms": 1.5, "median_ms": 1.3, "p95_ms": 2.0, "p99_ms": 2.5,
            "min_ms": 1.0, "max_ms": 3.0,
        }
        cold_stats = {
            "mean_ms": 500.0, "median_ms": 500.0, "p95_ms": 500.0,
            "p99_ms": 500.0, "min_ms": 500.0, "max_ms": 500.0,
        }

        warm_scenarios = ["single/short", "batch/mixed/8"]

        def _warm_payload(impl):
            if impl == "ltembed":
                return {
                    "implementation": "ltembed",
                    "implementation_version": "sha123",
                    "results": [
                        {"scenario": s, "stats": warm_stats}
                        for s in warm_scenarios
                    ],
                }
            return {
                "implementation": "pytorch",
                "implementation_version": "2.5.0",
                "transformers_version": "4.45.0",
                "results": [
                    {"scenario": s, "stats": warm_stats}
                    for s in warm_scenarios
                ],
            }

        def _cold_payload(scenario, impl):
            if impl == "ltembed":
                version = "sha123"
                transforms = None
            else:
                version = "2.5.0"
                transforms = {"transformers_version": "4.45.0"}
            payload = {
                "implementation": impl,
                "implementation_version": version,
                "scenario": scenario,
                "stats": cold_stats,
            }
            if transforms:
                payload.update(transforms)
            return payload

        embedding = [1.0, 0.0, 0.0]

        def _correctness_payload(impl):
            if impl == "ltembed":
                return {
                    "implementation": "ltembed",
                    "implementation_version": "sha123",
                    "results": [
                        {"scenario": "single/short", "embeddings": [embedding]},
                    ],
                }
            return {
                "implementation": "pytorch",
                "implementation_version": "2.5.0",
                "transformers_version": "4.45.0",
                "results": [
                    {"scenario": "single/short", "embeddings": [embedding]},
                ],
            }

        retrieval_payload = {
            "implementation": "ltembed",
            "implementation_version": "sha123",
            "results": [{
                "dataset_name": "mini-retrieval-v1",
                "queries": [
                    {"id": "q1", "embedding": [0.0, 1.0, 0.0]},
                    {"id": "q2", "embedding": [0.0, 1.0, 0.0]},
                ],
                "documents": [
                    {"id": "d1", "embedding": [0.0, 0.0, 1.0]},
                    {"id": "d2", "embedding": [0.0, 1.0, 0.0]},
                ],
            }],
        }

        retrieval_json = {
            "cases": [{
                "name": "mini-retrieval-v1",
                "documents": [
                    {"id": "d1", "text": "Rust ownership protects memory safety."},
                    {"id": "d2", "text": "Java uses a garbage collector."},
                ],
                "queries": [
                    {
                        "id": "q1",
                        "text": "How does Rust avoid a garbage collector?",
                        "relevant_document_ids": ["d1"],
                    },
                    {
                        "id": "q2",
                        "text": "What supports nearest-neighbor search?",
                        "relevant_document_ids": ["d2"],
                    },
                ],
            }],
        }

        def mock_run(_command, label):
            if " warm" in label or label.endswith(" warm"):
                if "ltembed" in label:
                    return _warm_payload("ltembed")
                return _warm_payload("pytorch")
            if " cold" in label:
                # label: "ltembed cold single/short", "pytorch cold single/short", etc
                scenario = label.rsplit(" ", 1)[1]
                impl = label.split(" ")[0]
                return _cold_payload(scenario, impl)
            if " correctness" in label:
                if "ltembed" in label:
                    return _correctness_payload("ltembed")
                return _correctness_payload("pytorch")
            if " retrieval" in label:
                return retrieval_payload
            raise AssertionError(f"unexpected label: {label!r}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            retrieval_path = tmp / "retrieval_eval.json"
            retrieval_path.write_text(json.dumps(retrieval_json))
            output_csv = tmp / "report.csv"
            output_summary = tmp / "summary.txt"

            args = SimpleNamespace(
                run_id="golden-001",
                model_id="test-model",
                model_source="huggingface",
                model_dir=ROOT / "assets",
                ort_bundle_dir=ROOT / "ort_bundle",
                output_dimension=512,
                l2_normalize=True,
                warmup=10,
                iters=100,
                threads=1,
                scenario=None,
                ltembed_cargo_features="",
                retrieval_eval_path=retrieval_path,
                include_cold_start=True,
                include_correctness=True,
                include_retrieval_eval=True,
                correctness_threshold=0.98,
                output_csv=output_csv,
                output_summary=output_summary,
            )

            host = {
                "host_os": "linux",
                "host_arch": "arm64",
                "cpu_model": "test-cpu",
                "runner_labels": "",
            }

            with mock.patch.object(bench, "run_json_command", side_effect=mock_run), \
                 mock.patch.object(bench, "SCENARIOS", [
                     bench.scenario_from_name("single/short"),
                     bench.scenario_from_name("batch/mixed/8"),
                 ]), \
                 mock.patch.dict(bench.RUNNERS, {
                     "ltembed": {
                         "warm": lambda a: ["lt-warm"],
                         "cold": lambda a, s: ["lt-cold"],
                         "correctness": lambda a: ["lt-correct"],
                         "retrieval": lambda a: ["lt-retrieval"],
                         "version": lambda: "sha123",
                     },
                     "pytorch": {
                         "warm": lambda a: ["pt-warm"],
                         "cold": lambda a, s: ["pt-cold"],
                         "correctness": lambda a: ["pt-correct"],
                         "retrieval": lambda a: ["pt-retrieval"],
                         "version": lambda: "",
                     },
                 }):
                rows: list[dict[str, str]] = []
                exit_code = bench._run(
                    args=args,
                    timestamp="2026-01-01T00:00:00",
                    git_revision="abc123",
                    host=host,
                    rows=rows,
                )

                self.assertEqual(exit_code, 0)

                # warm: 2 impls x 2 scenarios = 4
                # cold: 2 impls x 2 scenarios = 4
                # correctness: 2 impls x 1 scenario = 2
                # retrieval: 2 impls x 1 case = 2
                self.assertEqual(len(rows), 12)

                modes = [r["mode"] for r in rows]
                self.assertEqual(modes.count("warm_latency"), 4)
                self.assertEqual(modes.count("cold_start"), 4)
                self.assertEqual(modes.count("correctness"), 2)
                self.assertEqual(modes.count("retrieval_eval"), 2)

                # spot-check warm
                warm_row = [r for r in rows if r["mode"] == "warm_latency"][0]
                self.assertEqual(warm_row["model_id"], "test-model")
                self.assertEqual(warm_row["mean_ms"], "1.500000")
                self.assertEqual(warm_row["p95_ms"], "2.000000")
                self.assertIn(warm_row["implementation"], {"ltembed", "pytorch"})

                # spot-check cold
                cold_lt = [r for r in rows if r["mode"] == "cold_start" and r["implementation"] == "ltembed"][0]
                self.assertEqual(cold_lt["mean_ms"], "500.000000")
                self.assertEqual(cold_lt["batch_size"], "1")

                # spot-check correctness
                corr_lt = [r for r in rows if r["mode"] == "correctness" and r["implementation"] == "ltembed"][0]
                self.assertEqual(corr_lt["cosine_similarity_vs_pytorch"], "1.000000")
                self.assertEqual(corr_lt["status"], "pass")

                # spot-check retrieval
                ret_rows = [r for r in rows if r["mode"] == "retrieval_eval"]
                self.assertEqual(len(ret_rows), 2)
                ret_lt = [r for r in ret_rows if r["implementation"] == "ltembed"][0]
                self.assertEqual(ret_lt["query_count"], "2")
                self.assertEqual(ret_lt["recall_at_1"], "0.500000")
                self.assertEqual(ret_lt["recall_at_3"], "1.000000")
                self.assertEqual(ret_lt["mrr_at_3"], "0.750000")

                # written CSV must match the collected rows exactly
                with output_csv.open(newline="") as fh:
                    reader = csv.DictReader(fh)
                    self.assertEqual(reader.fieldnames, bench.CSV_FIELDNAMES)
                    csv_rows = list(reader)
                expected_csv_rows = [
                    {field: row.get(field, "") for field in bench.CSV_FIELDNAMES}
                    for row in rows
                ]
                self.assertEqual(csv_rows, expected_csv_rows)

                # summary must match line for line
                self.assertEqual(
                    output_summary.read_text().splitlines(),
                    [
                        "run_id=golden-001",
                        "git_sha=abc123",
                        "model_id=test-model",
                        "model_source=huggingface",
                        f"python_version={bench.python_version()}",
                        f"rust_version={bench.rust_version()}",
                        "ltembed_version=sha123",
                        "pytorch_version=2.5.0",
                        "transformers_version=4.45.0",
                        "cold_start=enabled",
                        "correctness=enabled",
                        "retrieval_eval=enabled",
                    ],
                )


if __name__ == "__main__":
    unittest.main()
