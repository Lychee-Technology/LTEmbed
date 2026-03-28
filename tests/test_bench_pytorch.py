import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


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
                "batch/mixed/8",
                "batch/medium/16",
            ],
        )
        self.assertEqual(bench.SCENARIOS["batch/medium/16"]["batch_size"], 16)
        self.assertEqual(bench.SCENARIOS["batch/mixed/8"]["text_profile"], "mixed")

    def test_compute_stats_uses_fixed_keys(self):
        bench = load_module()
        stats = bench.compute_stats([10.0, 20.0, 30.0, 40.0])
        self.assertEqual(
            set(stats.keys()),
            {"mean_ms", "median_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"},
        )
        self.assertEqual(stats["mean_ms"], 25.0)
        self.assertEqual(stats["median_ms"], 25.0)

    def test_embed_texts_casts_bfloat16_model_output_before_numpy(self):
        bench = load_module()

        class FakeTokenizer:
            def __call__(self, texts, **kwargs):
                self.seen_texts = texts
                return {
                    "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.int64),
                }

        class FakeModel:
            def __call__(self, **encoded):
                return SimpleNamespace(
                    last_hidden_state=torch.arange(
                        3 * bench.RAW_DIM, dtype=torch.bfloat16
                    ).reshape(1, 3, bench.RAW_DIM)
                )

        embeddings = bench.embed_texts(
            FakeModel(),
            FakeTokenizer(),
            [{"kind": "query", "text": "hello"}],
        )

        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), bench.OUTPUT_DIM)

    def test_progress_label_includes_mode_scenario_and_state(self):
        bench = load_module()
        label = bench.progress_label("warm", "batch/mixed/8", "start")
        self.assertEqual(label, "warm batch/mixed/8 start")


if __name__ == "__main__":
    unittest.main()
