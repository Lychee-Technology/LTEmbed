import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "bench_pytorch.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bench_pytorch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BenchPyTorchTests(unittest.TestCase):
    def test_scenarios_match_issue_38_plan(self):
        bench = load_module()
        self.assertEqual(
            list(bench.SCENARIOS.keys()),
            [
                "single/short",
                "single/medium",
                "single/long",
                "batch/medium/1",
                "batch/medium/4",
                "batch/medium/8",
                "batch/medium/16",
            ],
        )
        self.assertEqual(bench.SCENARIOS["batch/medium/16"]["batch_size"], 16)

    def test_compute_stats_uses_fixed_keys(self):
        bench = load_module()
        stats = bench.compute_stats([10.0, 20.0, 30.0, 40.0])
        self.assertEqual(
            set(stats.keys()),
            {"mean_ms", "median_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"},
        )
        self.assertEqual(stats["mean_ms"], 25.0)
        self.assertEqual(stats["median_ms"], 25.0)


if __name__ == "__main__":
    unittest.main()
