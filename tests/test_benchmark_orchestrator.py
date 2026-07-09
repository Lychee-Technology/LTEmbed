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


def ltembed_args(**overrides):
    base = dict(
        bundle_dir=Path("gguf_bundle"),
        model_dir=Path("assets"),
        output_dimension=512,
        l2_normalize=True,
        warmup=5,
        iters=10,
        threads=1,
        scenario=None,
        ltembed_cargo_features="",
        retrieval_eval_path=Path("retrieval.json"),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class CommandBuilderTests(unittest.TestCase):
    def test_scenarios_include_batch_mixed_profile(self):
        bench = load_module()
        scenario = bench.scenario_from_name("batch/mixed/8")
        self.assertEqual(scenario.name, "batch/mixed/8")
        self.assertEqual(scenario.batch_size, 8)
        self.assertEqual(scenario.text_profile, "mixed")

    def test_ltembed_command_uses_bundle_dir_and_mode(self):
        bench = load_module()
        with mock.patch.object(bench, "_prebuilt_ltembed_binary", return_value=None):
            command = bench.build_benchmark_command("ltembed", "warm", ltembed_args())
        self.assertIn("--bundle-dir", command)
        self.assertIn("gguf_bundle", command)
        self.assertIn("--mode", command)
        self.assertIn("warm", command)
        self.assertNotIn("--model-dir", command)
        self.assertNotIn("--ort-bundle-dir", command)

    def test_ltembed_command_falls_back_to_cargo_with_features(self):
        bench = load_module()
        args = ltembed_args(ltembed_cargo_features="vendored-blas")
        with mock.patch.object(bench, "_prebuilt_ltembed_binary", return_value=None):
            command = bench.build_benchmark_command("ltembed", "warm", args)
        self.assertEqual(
            command[:8],
            ["cargo", "run", "--quiet", "--release", "--features", "vendored-blas", "--bin", "benchmark_ltembed"],
        )
        self.assertEqual(command[8], "--")

    def test_ltembed_command_uses_prebuilt_binary_when_present(self):
        bench = load_module()
        fake = Path("/tmp/target/release/benchmark_ltembed")
        with mock.patch.object(bench, "_prebuilt_ltembed_binary", return_value=fake):
            command = bench.build_benchmark_command("ltembed", "cold", ltembed_args(), "single/long")
        self.assertEqual(command[0], str(fake))
        self.assertNotIn("cargo", command)
        self.assertNotIn("--bin", command)
        self.assertEqual(command[1:4], ["--mode", "cold", "--bundle-dir"])
        self.assertIn("single/long", command)

    def test_pytorch_command_uses_model_dir(self):
        bench = load_module()
        args = ltembed_args(model_dir=Path("assets"), output_dimension=768)
        command = bench.build_benchmark_command("pytorch", "correctness", args)
        self.assertIn("--model-name-or-path", command)
        self.assertIn("assets", command)
        self.assertIn("--output-dimension", command)
        self.assertIn("768", command)
        self.assertNotIn("--bundle-dir", command)

    def test_command_appends_fixture_path_for_both_runners(self):
        bench = load_module()
        args = ltembed_args(resolved_fixture_path=Path("artifacts/resolved_fixture.json"))
        with mock.patch.object(bench, "_prebuilt_ltembed_binary", return_value=None):
            for impl in ("ltembed", "pytorch"):
                command = bench.build_benchmark_command(impl, "correctness", args)
                self.assertIn("--fixture-path", command)
                self.assertIn("artifacts/resolved_fixture.json", command)

    def test_retrieval_command_passes_eval_path(self):
        bench = load_module()
        args = ltembed_args(retrieval_eval_path=Path("cn_en.json"))
        command = bench.build_benchmark_command("pytorch", "retrieval", args)
        self.assertIn("--retrieval-eval-path", command)
        self.assertIn("cn_en.json", command)


class CorpusAndFixtureTests(unittest.TestCase):
    def test_load_corpus_texts_sorts_by_length_and_skips_empty(self):
        bench = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corpus.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"text": "medium chunk here", "token_count": 50, "position": 1}),
                        json.dumps({"text": "", "token_count": 5, "position": 2}),
                        json.dumps({"text": "tiny", "token_count": 3, "position": 3}),
                        "   ",
                        json.dumps({"text": "the longest chunk", "token_count": 900, "position": 4}),
                    ]
                ),
                encoding="utf-8",
            )
            texts = bench.load_corpus_texts(path)
        self.assertEqual(texts, ["tiny", "medium chunk here", "the longest chunk"])

    def test_resolve_fixture_selects_distinct_batches_and_kinds(self):
        bench = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corpus.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps({"text": f"chunk number {i}", "token_count": i, "position": i})
                    for i in range(1, 41)
                ),
                encoding="utf-8",
            )
            fixture = bench.resolve_fixture(path, bench.SCENARIOS)

        scenarios = fixture["scenarios"]
        self.assertEqual(list(scenarios.keys()), [s.name for s in bench.SCENARIOS])
        self.assertEqual(scenarios["single/short"][0]["kind"], "query")
        self.assertEqual(scenarios["single/long"][0]["kind"], "document")
        batch = scenarios["batch/medium/8"]
        self.assertEqual(len(batch), 8)
        self.assertEqual(len({item["text"] for item in batch}), 8)
        mixed = scenarios["batch/mixed/8"]
        self.assertEqual(len(mixed), 8)
        self.assertEqual(mixed[0]["kind"], "query")
        self.assertEqual(mixed[2]["kind"], "document")


class CsvAndRowTests(unittest.TestCase):
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
        self.assertIn("both_at_3", header)
        self.assertEqual(values[header.index("implementation")], "ltembed")

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


class RetrievalMetricTests(unittest.TestCase):
    def test_single_relevant_is_backward_compatible(self):
        bench = load_module()
        case = {"queries": [
            {"id": "q1", "relevant_document_ids": ["d1"]},
            {"id": "q2", "relevant_document_ids": ["d2"]},
        ]}
        metrics = bench.compute_retrieval_metrics(
            case,
            query_embeddings={"q1": [0.0, 1.0, 0.0], "q2": [0.0, 1.0, 0.0]},
            document_embeddings={"d1": [0.0, 0.0, 1.0], "d2": [0.0, 1.0, 0.0]},
        )
        self.assertEqual(metrics["query_count"], 2)
        self.assertAlmostEqual(metrics["recall_at_1"], 0.5)
        self.assertAlmostEqual(metrics["recall_at_3"], 1.0)
        self.assertAlmostEqual(metrics["mrr_at_3"], 0.75)
        self.assertAlmostEqual(metrics["both_at_3"], 1.0)

    def test_both_at_3_requires_all_relevant_in_top3(self):
        bench = load_module()
        # qA: both relevant (self + translation) land in top-3 -> both@3 hit, recall@3 = 1.0
        # qB: translation pushed to rank 4 -> both@3 miss, recall@3 = 0.5
        case = {"queries": [
            {"id": "qA", "relevant_document_ids": ["self", "trans"]},
            {"id": "qB", "relevant_document_ids": ["selfB", "transB"]},
        ]}
        metrics = bench.compute_retrieval_metrics(
            case,
            query_embeddings={"qA": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "qB": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]},
            document_embeddings={
                # qA space: self (rank1) + trans (rank2) both land in top-3.
                "self": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "trans": [0.9, 0.0, 0.44, 0.0, 0.0, 0.0],
                # qB space: selfB is rank1 but two distractors outrank transB, pushing it to rank4.
                "selfB": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "distractor_1": [0.0, 0.8, 0.0, 0.0, 0.6, 0.0],
                "distractor_2": [0.0, 0.7, 0.0, 0.0, 0.0, 0.7],
                "transB": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            },
        )
        self.assertEqual(metrics["query_count"], 2)
        self.assertAlmostEqual(metrics["both_at_3"], 0.5)
        self.assertAlmostEqual(metrics["recall_at_3"], 0.75)

    def test_relevant_beyond_top3_scores_zero(self):
        bench = load_module()
        case = {"queries": [{"id": "q1", "relevant_document_ids": ["d_rel"]}]}
        metrics = bench.compute_retrieval_metrics(
            case,
            query_embeddings={"q1": [1.0, 0.0, 0.0]},
            document_embeddings={
                "d1": [1.0, 0.0, 0.0],
                "d2": [0.9, 0.1, 0.0],
                "d3": [0.8, 0.2, 0.0],
                "d_rel": [0.0, 1.0, 0.0],
            },
        )
        self.assertEqual(metrics["recall_at_1"], 0.0)
        self.assertEqual(metrics["recall_at_3"], 0.0)
        self.assertEqual(metrics["both_at_3"], 0.0)
        self.assertEqual(metrics["mrr_at_3"], 0.0)

    def test_retrieval_row_includes_both_at_3(self):
        bench = load_module()
        row = bench.retrieval_eval_row_from_metrics(
            base_fields={field: "" for field in bench.CSV_FIELDNAMES},
            metrics={
                "query_count": 4,
                "recall_at_1": 0.5,
                "recall_at_3": 0.75,
                "both_at_3": 0.5,
                "mrr_at_3": 0.9,
            },
        )
        self.assertEqual(row["query_count"], "4")
        self.assertEqual(row["both_at_3"], "0.500000")
        self.assertEqual(row["recall_at_3"], "0.750000")


class ReferenceModeTests(unittest.TestCase):
    def test_gather_payload_reads_pytorch_from_reference(self):
        bench = load_module()
        reference = {"correctness": {"impl": "pytorch-correctness"}, "retrieval": {"impl": "pytorch-retrieval"}}
        with mock.patch.object(bench, "run_json_command") as run_mock:
            payload = bench.gather_payload("pytorch", "correctness", ltembed_args(), reference=reference)
        self.assertEqual(payload, {"impl": "pytorch-correctness"})
        run_mock.assert_not_called()

    def test_gather_payload_runs_ltembed_subprocess(self):
        bench = load_module()
        reference = {"correctness": {"impl": "pytorch"}}
        with mock.patch.object(bench, "run_json_command", return_value={"impl": "ltembed"}) as run_mock, \
             mock.patch.object(bench, "_prebuilt_ltembed_binary", return_value=None):
            payload = bench.gather_payload("ltembed", "correctness", ltembed_args(), reference=reference)
        self.assertEqual(payload, {"impl": "ltembed"})
        run_mock.assert_called_once()

    def test_emit_reference_runs_only_pytorch_and_writes_embeddings(self):
        bench = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "reference.json"
            args = ltembed_args(
                emit_reference=out,
                output_csv=Path(tmpdir) / "report.csv",
                fixture_path=None,
                resolved_fixture_path=None,
                model_id="m",
                model_source="huggingface",
            )

            def fake_run(command, label):
                self.assertIn("bench_pytorch.py", " ".join(command))  # pytorch only
                mode = command[command.index("--mode") + 1]
                return {"implementation": "pytorch", "mode": mode, "results": []}

            with mock.patch.object(bench, "run_json_command", side_effect=fake_run):
                code = bench._emit_reference(args=args)

            self.assertEqual(code, 0)
            payload = json.loads(out.read_text())
        self.assertEqual(set(payload.keys()), {"correctness", "retrieval"})
        self.assertEqual(payload["correctness"]["mode"], "correctness")
        self.assertEqual(payload["retrieval"]["mode"], "retrieval")


class RunTests(unittest.TestCase):
    def _make_args(self, tmp, **overrides):
        bench = load_module()
        retrieval_json = {"cases": [{
            "name": "cn-en-crosslingual-v1",
            "documents": [
                {"id": "pair_0_zh", "text": "他感冒了"},
                {"id": "pair_0_en", "text": "He caught a cold."},
            ],
            "queries": [
                {"id": "q_0_zh", "text": "他感冒了", "relevant_document_ids": ["pair_0_zh", "pair_0_en"]},
                {"id": "q_0_en", "text": "He caught a cold.", "relevant_document_ids": ["pair_0_zh", "pair_0_en"]},
            ],
        }]}
        retrieval_path = tmp / "cn_en.json"
        retrieval_path.write_text(json.dumps(retrieval_json), encoding="utf-8")
        args = SimpleNamespace(
            run_id="run-1",
            model_id="test-model",
            model_source="huggingface",
            model_dir=ROOT / "assets",
            bundle_dir=ROOT / "gguf_bundle",
            output_dimension=512,
            l2_normalize=True,
            warmup=10,
            iters=100,
            threads=1,
            scenario=None,
            ltembed_cargo_features="",
            retrieval_eval_path=retrieval_path,
            fixture_path=None,
            emit_reference=None,
            reference_path=None,
            include_cold_start=True,
            include_correctness=True,
            include_retrieval_eval=True,
            correctness_threshold=0.98,
            output_csv=tmp / "report.csv",
            output_summary=tmp / "summary.txt",
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return bench, args

    @staticmethod
    def _embeddings_payload(impl, version, scenarios):
        return {
            "implementation": impl,
            "implementation_version": version,
            **({"transformers_version": "4.45.0"} if impl == "pytorch" else {}),
            "results": [{"scenario": s, "embeddings": [[1.0, 0.0, 0.0]]} for s in scenarios],
        }

    @staticmethod
    def _retrieval_payload(impl, version):
        return {
            "implementation": impl,
            "implementation_version": version,
            "results": [{
                "dataset_name": "cn-en-crosslingual-v1",
                "queries": [
                    {"id": "q_0_zh", "embedding": [1.0, 0.0, 0.0]},
                    {"id": "q_0_en", "embedding": [0.9, 0.1, 0.0]},
                ],
                "documents": [
                    {"id": "pair_0_zh", "embedding": [1.0, 0.0, 0.0]},
                    {"id": "pair_0_en", "embedding": [0.9, 0.1, 0.0]},
                ],
            }],
        }

    def _warm_payload(self, impl, version, scenarios):
        stats = {"mean_ms": 1.5, "median_ms": 1.3, "p95_ms": 2.0, "p99_ms": 2.5, "min_ms": 1.0, "max_ms": 3.0}
        return {
            "implementation": impl,
            "implementation_version": version,
            **({"transformers_version": "4.45.0"} if impl == "pytorch" else {}),
            "results": [{"scenario": s, "stats": stats} for s in scenarios],
        }

    def _cold_payload(self, impl, version, scenario):
        stats = {"mean_ms": 500.0, "median_ms": 500.0, "p95_ms": 500.0, "p99_ms": 500.0, "min_ms": 500.0, "max_ms": 500.0}
        return {"implementation": impl, "implementation_version": version, "scenario": scenario, "stats": stats,
                **({"transformers_version": "4.45.0"} if impl == "pytorch" else {})}

    def _run_with_mocks(self, bench, args, scenarios, reference=None):
        host = {"host_os": "linux", "host_arch": "arm64", "cpu_model": "cpu", "runner_labels": ""}

        def mock_run(command, label):
            impl = label.split(" ")[0]
            version = "sha123" if impl == "ltembed" else "2.5.0"
            if " warm" in label:
                return self._warm_payload(impl, version, scenarios)
            if " cold" in label:
                return self._cold_payload(impl, version, label.rsplit(" ", 1)[1])
            if " correctness" in label:
                return self._embeddings_payload(impl, version, scenarios)
            if " retrieval" in label:
                return self._retrieval_payload(impl, version)
            raise AssertionError(f"unexpected label {label!r}")

        with mock.patch.object(bench, "run_json_command", side_effect=mock_run) as run_mock, \
             mock.patch.object(bench, "SCENARIOS", [bench.scenario_from_name(s) for s in scenarios]), \
             mock.patch.object(bench, "_prebuilt_ltembed_binary", return_value=None):
            rows: list[dict] = []
            code = bench._run(
                args=args, timestamp="2026-01-01T00:00:00", git_revision="abc123", host=host, rows=rows,
            )
        return code, rows, run_mock

    def test_standalone_run_runs_both_impls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bench, args = self._make_args(tmp)
            scenarios = ["single/short", "batch/mixed/8"]
            code, rows, run_mock = self._run_with_mocks(bench, args, scenarios)

        self.assertEqual(code, 0)
        modes = [r["mode"] for r in rows]
        self.assertEqual(modes.count("warm_latency"), 4)   # 2 impls x 2 scenarios
        self.assertEqual(modes.count("cold_start"), 4)     # 2 impls x 2 scenarios
        self.assertEqual(modes.count("correctness"), 4)    # 2 impls x 2 scenarios
        self.assertEqual(modes.count("retrieval_eval"), 2)  # 2 impls x 1 case
        # ltembed correctness cosine vs pytorch reference == 1.0 (identical mocked embeddings)
        corr = [r for r in rows if r["mode"] == "correctness" and r["implementation"] == "ltembed"][0]
        self.assertEqual(corr["cosine_similarity_vs_pytorch"], "1.000000")
        # both@3 present on retrieval rows
        ret = [r for r in rows if r["mode"] == "retrieval_eval" and r["implementation"] == "ltembed"][0]
        self.assertEqual(ret["both_at_3"], "1.000000")

    def test_reference_mode_skips_pytorch_latency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            scenarios = ["single/short", "single/medium"]
            reference = {
                "correctness": self._embeddings_payload("pytorch", "2.5.0", scenarios),
                "retrieval": self._retrieval_payload("pytorch", "2.5.0"),
            }
            reference_path = tmp / "reference.json"
            reference_path.write_text(json.dumps(reference), encoding="utf-8")
            bench, args = self._make_args(tmp, reference_path=reference_path)
            code, rows, run_mock = self._run_with_mocks(bench, args, scenarios, reference=reference)

            self.assertEqual(code, 0)
            # No pytorch subprocess launched at all (labels are ltembed-only).
            labels = [call.args[1] for call in run_mock.call_args_list]
            self.assertTrue(all(label.startswith("ltembed") for label in labels), labels)
            modes = [r["mode"] for r in rows]
            self.assertEqual(modes.count("warm_latency"), 2)    # ltembed only x 2 scenarios
            self.assertEqual(modes.count("cold_start"), 2)      # ltembed only x 2 scenarios
            self.assertEqual(modes.count("correctness"), 4)     # ltembed + pytorch(from ref) x 2
            self.assertEqual(modes.count("retrieval_eval"), 2)  # ltembed + pytorch(from ref)
            # summary still records the pytorch version pulled from the reference
            summary = (tmp / "summary.txt").read_text()
            self.assertIn("ltembed_version=abc123", summary)
            self.assertIn("pytorch_version=2.5.0", summary)
            self.assertIn("transformers_version=4.45.0", summary)


class WorkflowTests(unittest.TestCase):
    def _workflow(self):
        return (ROOT / ".github" / "workflows" / "benchmark-arm64.yml").read_text(encoding="utf-8")

    def test_reference_job_emits_reference_and_generates_cn_en(self):
        workflow = self._workflow()
        self.assertIn("build_cn_en_retrieval_cases.py", workflow)
        self.assertIn("--emit-reference reference/reference.json", workflow)
        self.assertIn("name: benchmark-reference", workflow)

    def test_matrix_jobs_consume_reference_and_prebuild(self):
        workflow = self._workflow()
        self.assertIn("--reference-path reference/reference.json", workflow)
        self.assertIn("--retrieval-eval-path reference/cn_en_retrieval_cases.json", workflow)
        self.assertIn("cargo build --release --bin benchmark_ltembed", workflow)

    def test_reference_job_installs_cpu_only_pytorch(self):
        workflow = self._workflow()
        self.assertIn("https://download.pytorch.org/whl/cpu", workflow)

    def test_benchmark_workflow_enables_ltembed_stage_profiling(self):
        workflow = self._workflow()
        self.assertIn('export LTEMBED_PROFILE="1"', workflow)


if __name__ == "__main__":
    unittest.main()
