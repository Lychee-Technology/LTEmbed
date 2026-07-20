import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "render_benchmark_report.py"

CSV_FIELDS = [
    "implementation",
    "mode",
    "scenario",
    "batch_size",
    "text_profile",
    "threads",
    "warmup_iters",
    "timed_iters",
    "mean_ms",
    "median_ms",
    "p95_ms",
    "p99_ms",
    "min_ms",
    "max_ms",
    "cosine_similarity_vs_pytorch",
    "recall_at_1",
    "recall_at_3",
    "both_at_3",
    "mrr_at_3",
]

SCENARIOS = ["single/zh", "single/en", "single/medium", "single/long", "batch/medium/8"]


def load_module():
    spec = importlib.util.spec_from_file_location("render_benchmark_report", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(**kw):
    row = {field: "" for field in CSV_FIELDS}
    row.update(kw)
    return row


def _latency_row(mode, scenario, base_ms, timed_iters):
    batch_size = "8" if scenario.startswith("batch/") else "1"
    return _row(
        implementation="ltembed",
        mode=mode,
        scenario=scenario,
        batch_size=batch_size,
        text_profile=scenario.split("/")[1],
        threads="1",
        warmup_iters="10" if mode == "warm_latency" else "0",
        timed_iters=timed_iters,
        mean_ms=f"{base_ms:.6f}",
        median_ms=f"{base_ms * 0.9:.6f}",
        p95_ms=f"{base_ms * 1.5:.6f}",
        p99_ms=f"{base_ms * 1.8:.6f}",
        min_ms=f"{base_ms * 0.8:.6f}",
        max_ms=f"{base_ms * 2.0:.6f}",
    )


def _write_quant(
    base: Path,
    quant: str,
    *,
    model_bytes: int,
    bundle_bytes: int | None = None,
    warm: float = 8.0,
    golden_cosines=(0.995, 0.996),
    dynamic_cosines=(0.994, 0.995),
    both_at_3: float = 0.9,
):
    d = base / f"benchmark-arm64-{quant}"
    d.mkdir(parents=True)
    rows = []
    for scenario in SCENARIOS:
        rows.append(_latency_row("warm_latency", scenario, warm, "100"))
        rows.append(_latency_row("cold_start", scenario, warm * 50, "10"))
    rows.append(
        _row(
            implementation="ltembed",
            mode="retrieval_eval",
            scenario="cn-en-crosslingual-v1",
            recall_at_1="0.800000",
            recall_at_3="0.950000",
            both_at_3=f"{both_at_3:.6f}",
            mrr_at_3="0.900000",
        )
    )
    for index, cosine in enumerate(golden_cosines):
        rows.append(
            _row(
                implementation="ltembed",
                mode="golden_parity",
                scenario=f"golden/query/{index}",
                cosine_similarity_vs_pytorch=f"{cosine:.6f}",
            )
        )
    for index, cosine in enumerate(dynamic_cosines):
        rows.append(
            _row(
                implementation="ltembed",
                mode="correctness",
                scenario=f"cn-en/s{index}",
                cosine_similarity_vs_pytorch=f"{cosine:.6f}",
            )
        )
    with (d / "benchmark-report.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (d / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "llama.cpp",
                "quant": quant,
                "model_id": "m",
                "model_file": f"v5-nano-retrieval-{quant}.gguf",
                "model_sha256": "cafe",
                "model_size_bytes": model_bytes,
                "bundle_size_bytes": bundle_bytes if bundle_bytes is not None else model_bytes + 1_000_000,
                "static_llama_tag": "v0.1.151-1",
                "static_llama_sha256": "beef",
                "static_llama_contract_version": 3,
                "runner_labels": "ubuntu-24.04-arm",
                "cpu_model": "Neoverse-N1",
                "cpu_flags": ["asimd"],
                "threads": 1,
                "cold_iters": 10,
                "output_dimension": 512,
                "l2_normalize": True,
                "scenarios": SCENARIOS,
                "git_sha": "abc123",
            }
        )
    )


class RecordTests(unittest.TestCase):
    def _collect(self, tmpdir):
        mod = load_module()
        base = Path(tmpdir) / "bench-results"
        base.mkdir()
        _write_quant(base, "Q5_K_M", model_bytes=177_000_000)
        return mod, *mod.collect_results(base)

    def test_one_record_per_scenario_and_phase(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod, quants, records = self._collect(tmpdir)
        self.assertEqual(len(records), len(SCENARIOS) * 2)  # warm + cold each
        keys = {(r["scenario"], r["phase"]) for r in records}
        self.assertIn(("batch/medium/8", "warm"), keys)
        self.assertIn(("single/long", "cold"), keys)

    def test_records_carry_full_latency_distribution_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod, quants, records = self._collect(tmpdir)
        record = next(r for r in records if r["scenario"] == "batch/medium/8" and r["phase"] == "warm")
        self.assertEqual(record["quant"], "Q5_K_M")
        self.assertEqual(record["batch_size"], 8)
        self.assertEqual(record["text_profile"], "medium")
        self.assertEqual(record["warmup_iters"], 10)
        self.assertEqual(record["timed_iters"], 100)
        self.assertEqual(
            set(record["latency_ms"]), {"min", "mean", "median", "p95", "p99", "max"}
        )
        self.assertAlmostEqual(record["latency_ms"]["mean"], 8.0)
        self.assertAlmostEqual(record["latency_ms"]["p99"], 14.4)
        # denormalized metadata (issue #150 per-record fields)
        self.assertEqual(record["backend"], "llama.cpp")
        self.assertEqual(record["model_file"], "v5-nano-retrieval-Q5_K_M.gguf")
        self.assertEqual(record["model_sha256"], "cafe")
        self.assertEqual(record["static_llama_tag"], "v0.1.151-1")
        self.assertEqual(record["static_llama_contract_version"], 3)
        self.assertEqual(record["runner_labels"], "ubuntu-24.04-arm")
        self.assertEqual(record["cpu_flags"], ["asimd"])
        self.assertEqual(record["git_sha"], "abc123")
        # run/tuning configuration is denormalized onto every record
        self.assertEqual(record["output_dimension"], 512)
        self.assertIs(record["l2_normalize"], True)
        self.assertEqual(record["cold_iters"], 10)

    def test_cold_records_reflect_cold_iters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod, quants, records = self._collect(tmpdir)
        cold = next(r for r in records if r["phase"] == "cold")
        self.assertEqual(cold["warmup_iters"], 0)
        self.assertEqual(cold["timed_iters"], 10)


class SummaryTests(unittest.TestCase):
    def test_quant_summary_has_parity_split_and_lambda_fit(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "bench-results"
            base.mkdir()
            _write_quant(
                base,
                "Q5_K_M",
                model_bytes=177_000_000,
                golden_cosines=(0.995, 0.991),
                dynamic_cosines=(0.997, 0.999),
            )
            quants, _ = mod.collect_results(base)
        q = quants[0]
        self.assertAlmostEqual(q["golden_parity"]["mean_cosine"], 0.993)
        self.assertAlmostEqual(q["golden_parity"]["min_cosine"], 0.991)
        self.assertEqual(q["golden_parity"]["count"], 2)
        self.assertTrue(q["golden_parity"]["pass"])
        self.assertAlmostEqual(q["dynamic_parity"]["mean_cosine"], 0.998)
        self.assertTrue(q["lambda_fit"])
        self.assertAlmostEqual(q["retrieval"]["both_at_3"], 0.9)

    def test_legacy_gguf_metadata_keys_still_summarize(self):
        mod = load_module()
        quant = mod.summarize_quant(
            {"quant": "Q8_0", "gguf_file": "x.gguf", "gguf_size_bytes": 244_000_000}, []
        )
        self.assertEqual(quant["model_file"], "x.gguf")
        self.assertEqual(quant["model_size_bytes"], 244_000_000)
        self.assertFalse(quant["lambda_fit"] is None)


class RecommendationTests(unittest.TestCase):
    def test_recommends_smallest_fitting_quant_passing_gate(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "bench-results"
            base.mkdir()
            _write_quant(base, "IQ4_NL", model_bytes=140_000_000, golden_cosines=(0.95, 0.96))
            _write_quant(base, "Q5_K_M", model_bytes=177_000_000, golden_cosines=(0.995, 0.993))
            _write_quant(base, "Q8_0", model_bytes=244_000_000, golden_cosines=(0.997, 0.996))
            quants, _ = mod.collect_results(base)
        recommended, reason = mod.recommend(quants)
        # IQ4_NL fails the gate; Q5_K_M is the smallest passing quant that fits.
        self.assertEqual(recommended, "Q5_K_M")
        self.assertIn("Lambda", reason)

    def test_lambda_budget_excludes_oversized_quant_that_passes_parity(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "bench-results"
            base.mkdir()
            # Only the oversized quant passes parity -> no recommendation.
            _write_quant(base, "IQ4_NL", model_bytes=140_000_000, golden_cosines=(0.95, 0.96))
            _write_quant(base, "Q8_0", model_bytes=280_000_000, golden_cosines=(0.997, 0.996))
            quants, _ = mod.collect_results(base)
        recommended, reason = mod.recommend(quants)
        self.assertIsNone(recommended)
        self.assertIn("Q8_0", reason)
        self.assertIn("exceed", reason)

    def test_no_recommendation_when_none_pass_gate(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "bench-results"
            base.mkdir()
            _write_quant(base, "IQ4_NL", model_bytes=140_000_000, golden_cosines=(0.90, 0.91))
            _write_quant(base, "Q5_K_M", model_bytes=177_000_000, golden_cosines=(0.95, 0.94))
            quants, _ = mod.collect_results(base)
        recommended, reason = mod.recommend(quants)
        self.assertIsNone(recommended)
        self.assertIn("no quant met", reason)

    def test_gate_falls_back_to_dynamic_parity_without_golden(self):
        mod = load_module()
        quant = {
            "quant": "Q5_K_M",
            "bundle_size_bytes": 178_000_000,
            "bundle_size_mb": 169.8,
            "lambda_fit": True,
            "warm_mean_ms": 9.0,
            "golden_parity": None,
            "dynamic_parity": {"mean_cosine": 0.995, "min_cosine": 0.99, "count": 4},
        }
        recommended, _ = mod.recommend([quant])
        self.assertEqual(recommended, "Q5_K_M")

    def test_gate_uses_mean_not_min(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "bench-results"
            base.mkdir()
            # one low outlier but mean >= gate -> still passes
            _write_quant(
                base, "IQ4_NL", model_bytes=140_000_000,
                golden_cosines=(0.97, 0.997, 0.997, 0.997, 0.997),
            )
            _write_quant(base, "Q5_K_M", model_bytes=177_000_000, golden_cosines=(0.99, 0.99))
            quants, _ = mod.collect_results(base)
        recommended, _ = mod.recommend(quants)
        self.assertEqual(recommended, "IQ4_NL")


class LambdaBudgetContractTests(unittest.TestCase):
    def test_budget_is_package_limit_minus_binary_allowance(self):
        mod = load_module()
        self.assertEqual(mod.LAMBDA_PACKAGE_LIMIT_BYTES, 250 * 1024 * 1024)
        self.assertEqual(
            mod.LAMBDA_BUDGET_BYTES,
            mod.LAMBDA_PACKAGE_LIMIT_BYTES - mod.LAMBDA_BINARY_ALLOWANCE_BYTES,
        )

    def test_real_world_q8_bundle_is_over_budget(self):
        # The #150 tradeoff: Q8_0's ~233 MB GGUF (+~16 MB tokenizer -> ~238.5 MB bundle)
        # must be excluded, while Q5_K_M's ~177.5 MB bundle fits.
        mod = load_module()
        q8 = mod.summarize_quant(
            {"quant": "Q8_0", "model_size_bytes": 232_900_000, "bundle_size_bytes": 250_100_000},
            [],
        )
        q5 = mod.summarize_quant(
            {"quant": "Q5_K_M", "model_size_bytes": 168_900_000, "bundle_size_bytes": 186_100_000},
            [],
        )
        self.assertFalse(q8["lambda_fit"])
        self.assertTrue(q5["lambda_fit"])


class IncompleteMatrixTests(unittest.TestCase):
    def test_missing_quant_suppresses_recommendation_and_labels_report(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "bench-results"
            base.mkdir()
            _write_quant(base, "IQ4_NL", model_bytes=140_000_000, golden_cosines=(0.95, 0.96))
            _write_quant(base, "Q5_K_M", model_bytes=177_000_000, golden_cosines=(0.995, 0.993))
            out = Path(tmpdir) / "out"
            code = mod.main(
                [str(base), str(out), "--expected-quants", '["IQ4_NL", "Q5_K_M", "Q8_0"]']
            )
            self.assertEqual(code, 0)
            payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
            report = (out / "report.md").read_text(encoding="utf-8")

        self.assertEqual(payload["missing_quants"], ["Q8_0"])
        self.assertFalse(payload["complete"])
        self.assertIsNone(payload["recommendation"]["quant"])
        self.assertIn("incomplete quant matrix", payload["recommendation"]["reason"])
        self.assertIn("INCOMPLETE RUN", report)
        self.assertIn("`Q8_0`", report)

    def test_csv_expected_quants_also_accepted(self):
        mod = load_module()
        self.assertEqual(
            mod.parse_expected_quants("IQ4_NL, Q5_K_M,Q8_0"),
            ["IQ4_NL", "Q5_K_M", "Q8_0"],
        )
        self.assertEqual(
            mod.parse_expected_quants('["IQ4_NL","Q8_0"]'), ["IQ4_NL", "Q8_0"]
        )
        self.assertEqual(mod.parse_expected_quants(""), [])

    def test_complete_matrix_keeps_recommendation(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "bench-results"
            base.mkdir()
            _write_quant(base, "Q5_K_M", model_bytes=177_000_000)
            out = Path(tmpdir) / "out"
            code = mod.main([str(base), str(out), "--expected-quants", "Q5_K_M"])
            self.assertEqual(code, 0)
            payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
        self.assertTrue(payload["complete"])
        self.assertEqual(payload["recommendation"]["quant"], "Q5_K_M")


class PartialCoverageTests(unittest.TestCase):
    def test_partial_latency_coverage_is_labeled(self):
        mod = load_module()
        quants = [{
            "quant": "Q5_K_M",
            "scenarios": ["single/zh", "single/en"],
            "model_id": "m",
            "runner_labels": "arm",
            "golden_parity": {"mean_cosine": 0.995, "min_cosine": 0.99, "count": 2, "pass": True},
            "bundle_size_bytes": 178_000_000,
            "bundle_size_mb": 169.8,
            "model_size_mb": 160.0,
            "lambda_fit": True,
        }]
        records = [{
            "quant": "Q5_K_M", "scenario": "single/zh", "phase": "warm",
            "latency_ms": {"min": 1, "mean": 1, "median": 1, "p95": 1, "p99": 1, "max": 1},
        }]
        notes = mod.latency_coverage_notes(quants, records)
        self.assertEqual(len(notes), 2)
        self.assertIn("no warm records for single/en", notes[0])
        self.assertIn("no cold-start records", notes[1])
        report = mod.build_report(quants, records, ("Q5_K_M", "reason"))
        self.assertIn("Partial latency coverage", report)

    def test_full_coverage_has_no_notes(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "bench-results"
            base.mkdir()
            _write_quant(base, "Q5_K_M", model_bytes=177_000_000)
            quants, records = mod.collect_results(base)
        self.assertEqual(mod.latency_coverage_notes(quants, records), [])
        self.assertNotIn("Partial latency coverage", mod.build_report(quants, records, ("Q5_K_M", "r")))


class OutputTests(unittest.TestCase):
    def test_results_json_schema_and_report_sections(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "bench-results"
            base.mkdir()
            _write_quant(base, "Q5_K_M", model_bytes=177_000_000)
            _write_quant(base, "Q8_0", model_bytes=280_000_000, golden_cosines=(0.997, 0.996))
            out = Path(tmpdir) / "out"
            code = mod.main([str(base), str(out)])
            self.assertEqual(code, 0)
            payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
            report = (out / "report.md").read_text(encoding="utf-8")

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["quality_gate"], 0.99)
        self.assertEqual(payload["lambda_package_limit_bytes"], 250 * 1024 * 1024)
        self.assertEqual(payload["lambda_budget_bytes"], 230 * 1024 * 1024)
        self.assertEqual(len(payload["records"]), 2 * len(SCENARIOS) * 2)
        self.assertEqual([q["quant"] for q in payload["quants"]], ["Q5_K_M", "Q8_0"])
        self.assertEqual(payload["recommendation"]["quant"], "Q5_K_M")

        self.assertIn("## Size & Lambda fit", report)
        self.assertIn("## Parity vs FP32", report)
        self.assertIn("## Retrieval quality", report)
        self.assertIn("## Latency (ms, per scenario)", report)
        self.assertIn("Recommended quant: `Q5_K_M`", report)
        self.assertIn("batch/medium/8", report)
        # the oversized quant is called out explicitly
        self.assertIn("Over budget: `Q8_0`", report)

    def test_empty_input_renders_placeholder(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "bench-results"
            base.mkdir()
            quants, records = mod.collect_results(base)
            report = mod.build_report(quants, records, mod.recommend(quants))
        self.assertIn("No quant results were found", report)


if __name__ == "__main__":
    unittest.main()
