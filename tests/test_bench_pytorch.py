import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import sentence_transformers
    import torch
    import transformers
except ModuleNotFoundError:  # heavy optional deps; CI's Python script tests job runs without them
    torch = None


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "bench_pytorch.py"
ORCHESTRATOR_PATH = ROOT / "scripts" / "run_embedding_benchmarks.py"
ENGINE_PATH = ROOT / "src" / "engine" / "mod.rs"
# The runner imports these only in the functions that load and run the model.
# BenchPyTorchTests blocks them, so its tests show that the rest runs with numpy alone, as in
# CI's Python script tests job, even where they are installed.
TORCH_MODULES = ("torch", "transformers", "sentence_transformers")


def load_module(path=MODULE_PATH, name="bench_pytorch"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# The raw pooled and projected vector per kind, before the pipeline's final Normalize: the
# vector LTEmbed returns with l2_normalize off. Far from unit norm, as the real one is.
RAW = {
    "query": np.arange(1, 769, dtype=np.float32) * 3.0,
    "document": np.linspace(-2.0, 5.0, 768, dtype=np.float32) * 700.0,
}


class FakeModel:
    """Stands in for a SentenceTransformer: records each encode call and returns a fixed
    768-wide vector per kind. Like the real pipeline, whose last module is Normalize, it
    returns the RAW vector normalized."""

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
        raw = RAW[prompt_name]
        return np.tile(raw / np.linalg.norm(raw), (len(texts), 1))


class BenchPyTorchTests(unittest.TestCase):
    def setUp(self):
        # `import name` raises ModuleNotFoundError while sys.modules maps name to None.
        self.enterContext(mock.patch.dict(sys.modules, dict.fromkeys(TORCH_MODULES)))

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

    def test_embed_texts_gives_ltembeds_normalized_direction_at_every_width(self):
        # The pipeline's Normalize has removed the magnitude: what is left is the direction
        # of the raw vector truncated to the width, which LTEmbed returns with l2_normalize on.
        bench = load_module()
        for width in (768, 512, 256, 128):
            [embedding] = bench.embed_texts(
                FakeModel(bench.PROMPTS),
                [{"kind": "document", "text": "d"}],
                output_dimension=width,
            )
            truncated = RAW["document"][:width]
            np.testing.assert_allclose(embedding, truncated / np.linalg.norm(truncated),
                                       rtol=1e-5, atol=1e-7)

    def test_there_is_no_unnormalized_mode(self):
        bench = load_module()
        argv = ["bench_pytorch.py", "--mode", "retrieval", "--model-name-or-path", "m",
                "--l2-normalize", "false"]
        with mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stderr(io.StringIO()) as stderr, \
                self.assertRaises(SystemExit):
            bench.parse_args()
        self.assertIn("unrecognized arguments: --l2-normalize false", stderr.getvalue())

    def test_parse_args_accepts_the_orchestrators_command(self):
        bench = load_module()
        orchestrator = load_module(ORCHESTRATOR_PATH, "run_embedding_benchmarks")
        args = SimpleNamespace(
            model_dir=Path("model"), output_dimension=256, l2_normalize=False, threads=2,
            warmup=1, iters=2, retrieval_eval_path=Path("cases.json"),
            resolved_fixture_path=Path("fixture.json"),
        )
        command = orchestrator.build_benchmark_command("pytorch", "retrieval", args)
        self.assertEqual(Path(command[1]), MODULE_PATH)
        with mock.patch.object(sys, "argv", command[1:]):
            parsed = bench.parse_args()
        self.assertEqual((parsed.mode, parsed.output_dimension, parsed.threads),
                         ("retrieval", 256, 2))
        self.assertEqual(parsed.retrieval_eval_path, Path("cases.json"))

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


@unittest.skipUnless(torch is not None, "torch/sentence-transformers not installed")
class BenchPyTorchWithTorchTests(unittest.TestCase):
    def test_load_model_loads_the_float32_text_tower_on_cpu(self):
        bench = load_module()
        fake = FakeModel(bench.PROMPTS, dtype=torch.float32)
        with mock.patch.object(
            sentence_transformers, "SentenceTransformer", return_value=fake
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
                sentence_transformers, "SentenceTransformer", return_value=fake
            ), self.assertRaisesRegex(ValueError, message):
                bench.load_model("fake-model")

    def test_main_prints_only_the_retrieval_payload(self):
        bench = load_module()
        model = FakeModel(bench.PROMPTS)
        case = {
            "name": "case",
            "queries": [{"id": "q1", "text": "q"}],
            "documents": [{"id": "d1", "text": "d"}, {"id": "d2", "text": "e"}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            cases_path = Path(tmpdir) / "cases.json"
            cases_path.write_text(json.dumps({"cases": [case]}), encoding="utf-8")
            argv = ["bench_pytorch.py", "--mode", "retrieval", "--model-name-or-path", "m",
                    "--retrieval-eval-path", str(cases_path), "--output-dimension", "256",
                    "--threads", "3"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(bench, "load_model", return_value=model) as load_model, \
                    mock.patch.object(torch, "set_num_threads") as set_num_threads, \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                bench.main()
        load_model.assert_called_once_with("m")
        set_num_threads.assert_called_once_with(3)
        self.assertEqual(model.calls,
                         [(["q"], "query", 1), (["d"], "document", 1), (["e"], "document", 1)])
        payload = json.loads(stdout.getvalue())
        [result] = payload.pop("results")
        self.assertEqual(payload, {
            "implementation": "pytorch",
            "implementation_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "sentence_transformers_version": sentence_transformers.__version__,
        })
        self.assertEqual(result["dataset_name"], "case")
        self.assertEqual([item["id"] for item in result["queries"]], ["q1"])
        self.assertEqual([item["id"] for item in result["documents"]], ["d1", "d2"])
        for item in result["queries"] + result["documents"]:
            self.assertEqual(len(item["embedding"]), 256)


if __name__ == "__main__":
    unittest.main()
