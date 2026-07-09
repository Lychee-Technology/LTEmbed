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
    def test_scenarios_and_fixture_machinery_removed(self):
        bench = load_module()
        self.assertFalse(hasattr(bench, "SCENARIOS"))
        self.assertFalse(hasattr(bench, "apply_fixture"))
        self.assertFalse(hasattr(bench, "warm_payload"))

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

    def test_embed_texts_respects_output_dimension_override(self):
        bench = load_module()

        class FakeTokenizer:
            def __call__(self, texts, **kwargs):
                return {
                    "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.int64),
                }

        class FakeModel:
            def __call__(self, **encoded):
                return SimpleNamespace(
                    last_hidden_state=torch.arange(
                        3 * bench.RAW_DIM, dtype=torch.float32
                    ).reshape(1, 3, bench.RAW_DIM)
                )

        embeddings = bench.embed_texts(
            FakeModel(),
            FakeTokenizer(),
            [{"kind": "query", "text": "hello"}],
            output_dimension=bench.RAW_DIM,
        )

        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), bench.RAW_DIM)

    def test_progress_label_includes_mode_scenario_and_state(self):
        bench = load_module()
        self.assertEqual(bench.progress_label("retrieval", "cn-en-crosslingual-v1", "start"),
                         "retrieval cn-en-crosslingual-v1 start")

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


if __name__ == "__main__":
    unittest.main()
