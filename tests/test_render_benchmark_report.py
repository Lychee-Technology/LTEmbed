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
    "mean_ms",
    "cosine_similarity_vs_pytorch",
    "recall_at_1",
    "recall_at_3",
    "both_at_3",
    "mrr_at_3",
]


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


def _write_quant(base: Path, quant: str, size_bytes: int, warm: float, cosines, both_at_3: float = 0.9):
    d = base / f"benchmark-arm64-{quant}"
    d.mkdir(parents=True)
    rows = [
        _row(implementation="ltembed", mode="warm_latency", scenario="single/medium", mean_ms=f"{warm:.6f}"),
        _row(
            implementation="ltembed",
            mode="retrieval_eval",
            scenario="cn-en-crosslingual-v1",
            recall_at_1="0.800000",
            recall_at_3="0.950000",
            both_at_3=f"{both_at_3:.6f}",
            mrr_at_3="0.900000",
        ),
    ]
    for i, c in enumerate(cosines):
        rows.append(
            _row(
                implementation="ltembed",
                mode="correctness",
                scenario=f"s{i}",
                cosine_similarity_vs_pytorch=f"{c:.6f}",
            )
        )
    with (d / "benchmark-report.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (d / "metadata.json").write_text(
        json.dumps({"quant": quant, "gguf_size_bytes": size_bytes, "model_id": "m", "runner_labels": "arm"})
    )


class RenderBenchmarkReportTests(unittest.TestCase):
    def test_recommends_smallest_quant_passing_quality_gate(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "bench-results"
            base.mkdir()
            _write_quant(base, "IQ4_NL", 190_000_000, warm=8.0, cosines=[0.95, 0.96], both_at_3=0.85)
            _write_quant(base, "Q5_K_M", 230_000_000, warm=9.5, cosines=[0.995, 0.993], both_at_3=0.92)
            _write_quant(base, "Q8_0", 350_000_000, warm=12.0, cosines=[0.997, 0.996], both_at_3=0.94)

            results = mod.collect_results(base)
            self.assertEqual([r["quant"] for r in results], ["IQ4_NL", "Q5_K_M", "Q8_0"])

            recommended, _reason = mod.recommend(results)
            # IQ4_NL fails the 0.99 gate; Q5_K_M is the smallest that passes.
            self.assertEqual(recommended, "Q5_K_M")

            q5 = next(r for r in results if r["quant"] == "Q5_K_M")
            self.assertAlmostEqual(q5["min_cosine"], 0.993)
            self.assertAlmostEqual(q5["both_at_3"], 0.92)

            report = mod.build_report(results)
            self.assertIn("Recommended quant: `Q5_K_M`", report)
            self.assertIn("vs PyTorch FP32", report)

    def test_no_recommendation_when_none_pass_gate(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "bench-results"
            base.mkdir()
            _write_quant(base, "IQ4_NL", 190_000_000, warm=8.0, cosines=[0.90, 0.91])
            _write_quant(base, "Q5_K_M", 230_000_000, warm=9.5, cosines=[0.95, 0.94])
            results = mod.collect_results(base)
            recommended, reason = mod.recommend(results)
            # None clears the 0.99 gate -> no recommendation (do not fall back to "least bad").
            self.assertIsNone(recommended)
            self.assertIn("no quant met", reason)
            self.assertIn("No recommendation", mod.build_report(results))

    def test_gate_uses_mean_not_min(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "bench-results"
            base.mkdir()
            # IQ4_NL: one low outlier (0.97) but mean ~0.992 -> passes mean gate despite low min_cosine
            _write_quant(base, "IQ4_NL", 190_000_000, warm=8.0, cosines=[0.97, 0.997, 0.997, 0.997, 0.997])
            _write_quant(base, "Q5_K_M", 230_000_000, warm=9.5, cosines=[0.99, 0.99])
            results = mod.collect_results(base)
            recommended, _ = mod.recommend(results)
            self.assertEqual(recommended, "IQ4_NL")  # smallest with mean >= gate


if __name__ == "__main__":
    unittest.main()
