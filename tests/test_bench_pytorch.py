import importlib.util
import json
import tempfile
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

    def test_postprocess_embedding_can_skip_truncation_and_normalization(self):
        bench = load_module()
        raw = torch.zeros((1, bench.RAW_DIM), dtype=torch.float32)
        raw[0, 0] = 3.0
        raw[0, 1] = 4.0
        raw[0, 2] = 10.0

        output = bench.postprocess_embeddings(
            raw.numpy(),
            output_dimension=3,
            l2_normalize=False,
        )

        self.assertEqual(output.tolist(), [[3.0, 4.0, 10.0]])

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
            output_dimension=bench.OUTPUT_DIM,
            l2_normalize=True,
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
        measure_warm_stats.assert_called_once_with(
            "model",
            "tokenizer",
            "single/medium",
            2,
            3,
            output_dimension=bench.OUTPUT_DIM,
            l2_normalize=True,
        )

    def test_correctness_payload_filters_to_requested_scenario(self):
        bench = load_module()
        args = SimpleNamespace(
            model_name_or_path="fake-model",
            scenario="single/medium",
            output_dimension=bench.RAW_DIM,
            l2_normalize=False,
        )

        with (
            mock.patch.object(bench, "load_model", return_value=("model", "tokenizer")),
            mock.patch.object(bench, "embed_texts", return_value=[[0.1, 0.2]]) as embed_texts,
        ):
            payload = bench.correctness_payload(args)

        self.assertEqual([row["scenario"] for row in payload["results"]], ["single/medium"])
        embed_texts.assert_called_once()

    def test_retrieval_payload_embeds_queries_and_documents_from_case_file(self):
        bench = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            retrieval_eval_path = Path(tmpdir) / "retrieval_eval_cases.json"
            retrieval_eval_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "name": "mini-retrieval-v1",
                                "documents": [
                                    {"id": "d1", "text": "Rust ownership protects memory safety."},
                                    {"id": "d2", "text": "Vector databases power semantic search."},
                                ],
                                "queries": [
                                    {
                                        "id": "q1",
                                        "text": "How does Rust avoid garbage collection?",
                                        "relevant_document_ids": ["d1"],
                                    }
                                ],
                            },
                            {
                                "name": "mini-retrieval-hard-v1",
                                "documents": [
                                    {"id": "d3", "text": "BM25 ranks lexical results with token statistics."},
                                    {"id": "d4", "text": "ANN indexes accelerate dense vector retrieval."},
                                ],
                                "queries": [
                                    {
                                        "id": "q2",
                                        "text": "What system supports nearest-neighbor search over dense embeddings?",
                                        "relevant_document_ids": ["d4"],
                                    }
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                model_name_or_path="fake-model",
                retrieval_eval_path=retrieval_eval_path,
                output_dimension=bench.OUTPUT_DIM,
                l2_normalize=True,
            )

            with (
                mock.patch.object(bench, "load_model", return_value=("model", "tokenizer")),
                mock.patch.object(
                    bench,
                    "embed_texts",
                    side_effect=[
                        [[1.0, 0.0]],
                        [[0.0, 1.0], [0.5, 0.5]],
                        [[0.2, 0.8]],
                        [[0.9, 0.1], [0.4, 0.6]],
                    ],
                ) as embed_texts,
            ):
                payload = bench.retrieval_payload(args)

        self.assertEqual([result["dataset_name"] for result in payload["results"]], ["mini-retrieval-v1", "mini-retrieval-hard-v1"])
        self.assertEqual(payload["results"][0]["queries"], [{"id": "q1", "embedding": [1.0, 0.0]}])
        self.assertEqual(
            payload["results"][0]["documents"],
            [
                {"id": "d1", "embedding": [0.0, 1.0]},
                {"id": "d2", "embedding": [0.5, 0.5]},
            ],
        )
        self.assertEqual(payload["results"][1]["queries"], [{"id": "q2", "embedding": [0.2, 0.8]}])
        self.assertEqual(
            payload["results"][1]["documents"],
            [
                {"id": "d3", "embedding": [0.9, 0.1]},
                {"id": "d4", "embedding": [0.4, 0.6]},
            ],
        )
        self.assertEqual(
            embed_texts.call_args_list[0].args[2],
            [{"kind": "query", "text": "How does Rust avoid garbage collection?"}],
        )
        self.assertEqual(
            embed_texts.call_args_list[1].args[2],
            [
                {"kind": "document", "text": "Rust ownership protects memory safety."},
                {"kind": "document", "text": "Vector databases power semantic search."},
            ],
        )
        self.assertEqual(
            embed_texts.call_args_list[2].args[2],
            [{"kind": "query", "text": "What system supports nearest-neighbor search over dense embeddings?"}],
        )


if __name__ == "__main__":
    unittest.main()
