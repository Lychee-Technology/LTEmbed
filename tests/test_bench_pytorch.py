import importlib.util
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np
    import sentence_transformers  # noqa: F401
    import torch
except ModuleNotFoundError:  # heavy optional deps; CI's python-tests job runs without them
    torch = None


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "bench_pytorch.py"
ENGINE_PATH = ROOT / "src" / "engine" / "mod.rs"


def load_module():
    spec = importlib.util.spec_from_file_location("bench_pytorch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeModel:
    """Stands in for a SentenceTransformer: records each encode call and returns a fixed
    768-wide vector per kind."""

    def __init__(self, prompts, tokens=5, dtype=None):
        self.prompts = prompts
        self.tokens = tokens
        self.calls = []
        self.eval_called = False
        parameter = SimpleNamespace(dtype=dtype)
        self.auto_model = SimpleNamespace(parameters=lambda: iter([parameter]))

    def __getitem__(self, index):
        assert index == 0
        return SimpleNamespace(auto_model=self.auto_model)

    def eval(self):
        self.eval_called = True
        return self

    def preprocess(self, texts):
        return {"input_ids": np.zeros((len(texts), self.tokens), dtype=np.int64)}

    def encode(self, texts, prompt_name, batch_size, convert_to_numpy):
        self.calls.append((list(texts), prompt_name, batch_size))
        value = 1.0 if prompt_name == "query" else 2.0
        return np.full((len(texts), 768), value, dtype=np.float32)


@unittest.skipUnless(torch is not None, "torch/sentence-transformers not installed")
class BenchPyTorchTests(unittest.TestCase):
    def test_prompts_match_the_engine_prefixes(self):
        bench = load_module()
        source = ENGINE_PATH.read_text(encoding="utf-8")
        for kind, constant in (("query", "QUERY_PREFIX"), ("document", "DOCUMENT_PREFIX")):
            match = re.search(rf'pub const {constant}: &str = "([^"]*)";', source)
            self.assertIsNotNone(match, constant)
            self.assertEqual(bench.PROMPTS[kind], match.group(1))

    def test_compute_stats_uses_fixed_keys(self):
        bench = load_module()
        stats = bench.compute_stats([10.0, 20.0, 30.0, 40.0])
        self.assertEqual(
            set(stats.keys()),
            {"mean_ms", "median_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"},
        )
        self.assertEqual(stats["mean_ms"], 25.0)
        self.assertEqual(stats["median_ms"], 25.0)

    def test_embed_texts_uses_each_kinds_prompt_one_input_at_a_time(self):
        bench = load_module()
        model = FakeModel(bench.PROMPTS)
        embeddings = bench.embed_texts(
            model,
            [{"kind": "query", "text": "q"}, {"kind": "document", "text": "d"}],
        )
        self.assertEqual(model.calls, [(["q"], "query", 1), (["d"], "document", 1)])
        self.assertEqual([len(e) for e in embeddings], [bench.OUTPUT_DIM] * 2)
        for embedding in embeddings:
            self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=6)

    def test_embed_texts_respects_output_dimension_and_normalization(self):
        bench = load_module()
        embeddings = bench.embed_texts(
            FakeModel(bench.PROMPTS),
            [{"kind": "document", "text": "d"}],
            output_dimension=bench.RAW_DIM,
            l2_normalize=False,
        )
        self.assertEqual(len(embeddings[0]), bench.RAW_DIM)
        self.assertEqual(embeddings[0][0], 2.0)

    def test_embed_texts_rejects_input_over_max_length_instead_of_truncating(self):
        bench = load_module()
        model = FakeModel(bench.PROMPTS, tokens=bench.MAX_LENGTH + 1)
        with self.assertRaisesRegex(ValueError, "exceeds MAX_LENGTH"):
            bench.embed_texts(model, [{"kind": "query", "text": "long"}])
        self.assertEqual(model.calls, [])

    def test_progress_label_includes_mode_scenario_and_state(self):
        bench = load_module()
        self.assertEqual(bench.progress_label("retrieval", "cn-en-crosslingual-v1", "start"),
                         "retrieval cn-en-crosslingual-v1 start")

    def test_load_model_loads_the_float32_text_tower_on_cpu(self):
        bench = load_module()
        fake = FakeModel(bench.PROMPTS, dtype=torch.float32)
        with mock.patch.object(
            bench.sentence_transformers, "SentenceTransformer", return_value=fake
        ) as constructor:
            self.assertIs(bench.load_model("fake-model"), fake)
        constructor.assert_called_once_with(
            "fake-model",
            device="cpu",
            model_kwargs={"dtype": torch.float32},
            config_kwargs={"vision_config": None, "audio_config": None},
        )
        self.assertTrue(fake.eval_called)

    def test_load_model_rejects_other_prompts_or_dtype(self):
        bench = load_module()
        for fake, message in (
            (FakeModel(dict(bench.PROMPTS, query="query: "), dtype=torch.float32), "prompts"),
            (FakeModel(bench.PROMPTS, dtype=torch.bfloat16), "float32"),
        ):
            with mock.patch.object(
                bench.sentence_transformers, "SentenceTransformer", return_value=fake
            ), self.assertRaisesRegex(ValueError, message):
                bench.load_model("fake-model")


if __name__ == "__main__":
    unittest.main()
