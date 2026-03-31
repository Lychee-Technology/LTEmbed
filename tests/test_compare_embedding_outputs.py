import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "compare_embedding_outputs.py"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "compare-raw-embeddings-arm64.yml"


def load_module():
    spec = importlib.util.spec_from_file_location("compare_embedding_outputs", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CompareEmbeddingOutputsTests(unittest.TestCase):
    def test_build_summary_computes_expected_metrics(self):
        compare = load_module()
        summary = compare.build_summary(
            scenario="single/medium",
            ltembed_embedding=[1.0, 0.0, 0.0],
            pytorch_embedding=[0.6, 0.8, 0.0],
            first_values=3,
        )

        self.assertEqual(summary["scenario"], "single/medium")
        self.assertEqual(summary["dimensions"], 3)
        self.assertAlmostEqual(summary["cosine_similarity"], 0.6)
        self.assertAlmostEqual(summary["ltembed_l2_norm"], 1.0)
        self.assertAlmostEqual(summary["pytorch_l2_norm"], 1.0)
        self.assertAlmostEqual(summary["max_abs_error"], 0.8)
        self.assertAlmostEqual(summary["mean_abs_error"], (0.4 + 0.8 + 0.0) / 3.0)
        self.assertEqual(summary["ltembed_first_values"], [1.0, 0.0, 0.0])
        self.assertEqual(summary["pytorch_first_values"], [0.6, 0.8, 0.0])

    def test_main_writes_json_and_text_outputs(self):
        compare = load_module()
        ltembed_payload = {
            "results": [
                {
                    "scenario": "single/medium",
                    "embeddings": [[1.0, 0.0, 0.0]],
                }
            ]
        }
        pytorch_payload = {
            "results": [
                {
                    "scenario": "single/medium",
                    "embeddings": [[0.6, 0.8, 0.0]],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ltembed_path = tmp / "ltembed.json"
            pytorch_path = tmp / "pytorch.json"
            output_json = tmp / "summary.json"
            output_text = tmp / "summary.txt"
            ltembed_path.write_text(json.dumps(ltembed_payload), encoding="utf-8")
            pytorch_path.write_text(json.dumps(pytorch_payload), encoding="utf-8")

            exit_code = compare.main(
                [
                    "--ltembed-json",
                    str(ltembed_path),
                    "--pytorch-json",
                    str(pytorch_path),
                    "--scenario",
                    "single/medium",
                    "--output-json",
                    str(output_json),
                    "--output-text",
                    str(output_text),
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads(output_json.read_text(encoding="utf-8"))
            text = output_text.read_text(encoding="utf-8")

        self.assertAlmostEqual(summary["cosine_similarity"], 0.6)
        self.assertIn("scenario: single/medium", text)
        self.assertIn("cosine_similarity: 0.600000", text)

    def test_workflow_provides_ci_entrypoint_for_raw_embedding_compare(self):
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

        self.assertIn("name: compare-raw-embeddings-arm64", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("builder_release_tag:", workflow)
        self.assertIn('default: "v1.0.16-b"', workflow)
        self.assertIn("builder_asset_name:", workflow)
        self.assertIn(
            'default: "jinaai__jina-embeddings-v5-text-nano-retrieval_q4f16_linux-arm64.tar.gz"',
            workflow,
        )
        self.assertIn("builder_asset_sha256:", workflow)
        self.assertIn(
            'default: "6ff82b97bb7b77f1e8b220007cf2b059b4a6dc92cebd2dc5d53984dc27e23b88"',
            workflow,
        )
        self.assertIn('default: "single/medium"', workflow)
        self.assertIn('default: "768"', workflow)
        self.assertIn('default: false', workflow)
        self.assertIn("scripts/bench_pytorch.py", workflow)
        self.assertIn("benchmark_ltembed", workflow)
        self.assertIn("scripts/compare_embedding_outputs.py", workflow)
        self.assertIn("raw-embedding-compare.json", workflow)
        self.assertIn("raw-embedding-compare.txt", workflow)


if __name__ == "__main__":
    unittest.main()
