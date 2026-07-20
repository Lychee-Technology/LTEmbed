import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

# The script under test loads bench_pytorch (and thus torch) at import time; torch is a
# heavy optional dep the CI python-tests job does not install.
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "compare_q4f16_onnx_vs_pytorch.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compare_q4f16_onnx_vs_pytorch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch is not None, "torch not installed")
class CompareQ4f16OnnxVsPyTorchTests(unittest.TestCase):
    def test_pool_last_token_numpy_uses_attention_mask_lengths(self):
        compare = load_module()
        last_hidden_state = np.array(
            [
                [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
                [[40.0, 41.0], [50.0, 51.0], [60.0, 61.0]],
            ],
            dtype=np.float32,
        )
        attention_mask = np.array(
            [
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=np.int64,
        )

        pooled = compare.pool_last_token_numpy(last_hidden_state, attention_mask)

        np.testing.assert_allclose(
            pooled,
            np.array([[20.0, 21.0], [60.0, 61.0]], dtype=np.float32),
        )

    def test_build_onnx_input_feed_only_passes_declared_inputs(self):
        compare = load_module()

        class FakeInput:
            def __init__(self, name):
                self.name = name

        class FakeSession:
            def get_inputs(self):
                return [FakeInput("input_ids"), FakeInput("attention_mask")]

        encoded = {
            "input_ids": np.array([[1, 2]], dtype=np.int64),
            "attention_mask": np.array([[1, 1]], dtype=np.int64),
            "token_type_ids": np.array([[0, 0]], dtype=np.int64),
        }

        input_feed = compare.build_onnx_input_feed(FakeSession(), encoded)

        self.assertEqual(set(input_feed.keys()), {"input_ids", "attention_mask"})

    def test_main_writes_summary_outputs(self):
        compare = load_module()
        summary = {
            "scenario": "single/medium",
            "num_embeddings": 1,
            "dimensions": 3,
            "cosine_similarity": 0.5,
            "cosine_similarity_min": 0.5,
            "cosine_similarity_max": 0.5,
            "onnx_l2_norm": 1.0,
            "pytorch_l2_norm": 1.0,
            "max_abs_error": 0.8,
            "mean_abs_error": 0.4,
            "rmse": 0.5,
            "onnx_first_values": [1.0, 0.0, 0.0],
            "pytorch_first_values": [0.6, 0.8, 0.0],
            "abs_error_first_values": [0.4, 0.8, 0.0],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_json = tmp / "summary.json"
            output_text = tmp / "summary.txt"
            with mock.patch.object(compare, "compare_scenario", return_value=summary):
                exit_code = compare.main(
                    [
                        "--model-name-or-path",
                        "hf-model",
                        "--onnx-model-path",
                        "onnx/model_q4f16.onnx",
                        "--output-json",
                        str(output_json),
                        "--output-text",
                        str(output_text),
                    ]
                )

            self.assertEqual(exit_code, 0)
            written_json = json.loads(output_json.read_text(encoding="utf-8"))
            written_text = output_text.read_text(encoding="utf-8")

        self.assertEqual(written_json["scenario"], "single/medium")
        self.assertIn("scenario: single/medium", written_text)
        self.assertIn("cosine_similarity: 0.500000", written_text)


if __name__ == "__main__":
    unittest.main()
