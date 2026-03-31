import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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

    def test_load_model_moves_cpu_model_to_float32(self):
        bench = load_module()

        class FakeModel:
            def __init__(self):
                self.eval_called = False
                self.to_calls = []

            def eval(self):
                self.eval_called = True
                return self

            def to(self, *args, **kwargs):
                self.to_calls.append((args, kwargs))
                return self

        fake_model = FakeModel()

        with (
            mock.patch.object(
                bench.AutoTokenizer,
                "from_pretrained",
                return_value="fake-tokenizer",
            ) as tokenizer_from_pretrained,
            mock.patch.object(
                bench.AutoModel,
                "from_pretrained",
                return_value=fake_model,
            ) as model_from_pretrained,
        ):
            model, tokenizer = bench.load_model("fake-model")

        self.assertIs(model, fake_model)
        self.assertEqual(tokenizer, "fake-tokenizer")
        self.assertTrue(fake_model.eval_called)
        tokenizer_from_pretrained.assert_called_once_with(
            "fake-model",
            trust_remote_code=True,
        )
        model_from_pretrained.assert_called_once_with(
            "fake-model",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self.assertIn((("cpu",), {}), fake_model.to_calls)
        self.assertIn(((), {"dtype": torch.float32}), fake_model.to_calls)

    def test_warm_payload_filters_to_requested_scenario(self):
        bench = load_module()
        args = SimpleNamespace(
            model_name_or_path="fake-model",
            warmup=2,
            iters=3,
            scenario="single/medium",
        )

        with (
            mock.patch.object(bench, "load_model", return_value=("model", "tokenizer")),
            mock.patch.object(
                bench,
                "measure_warm_stats",
                return_value={"mean_ms": 1.0, "median_ms": 1.0, "p95_ms": 1.0, "p99_ms": 1.0, "min_ms": 1.0, "max_ms": 1.0},
            ) as measure_warm_stats,
        ):
            payload = bench.warm_payload(args)

        self.assertEqual([row["scenario"] for row in payload["results"]], ["single/medium"])
        measure_warm_stats.assert_called_once_with("model", "tokenizer", "single/medium", 2, 3)

    def test_correctness_payload_filters_to_requested_scenario(self):
        bench = load_module()
        args = SimpleNamespace(
            model_name_or_path="fake-model",
            scenario="single/medium",
        )

        with (
            mock.patch.object(bench, "load_model", return_value=("model", "tokenizer")),
            mock.patch.object(bench, "embed_texts", return_value=[[0.1, 0.2]]) as embed_texts,
        ):
            payload = bench.correctness_payload(args)

        self.assertEqual([row["scenario"] for row in payload["results"]], ["single/medium"])
        embed_texts.assert_called_once()


if __name__ == "__main__":
    unittest.main()
