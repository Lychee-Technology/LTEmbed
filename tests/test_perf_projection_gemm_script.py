import argparse
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "profile_projection_gemm_perf.py"


def load_module():
    spec = importlib.util.spec_from_file_location("profile_projection_gemm_perf", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ProfileProjectionGemmPerfTests(unittest.TestCase):
    def test_benchmark_command_defaults_to_single_long(self):
        perf = load_module()
        args = argparse.Namespace(
            model_dir=Path("assets"),
            scenario="single/long",
            warmup=1,
            iters=1,
            threads=1,
        )

        command = perf.benchmark_command(args, Path("target/release/benchmark_ltembed"))

        self.assertEqual(
            command,
            [
                "target/release/benchmark_ltembed",
                "--mode",
                "warm",
                "--scenario",
                "single/long",
                "--model-dir",
                "assets",
                "--warmup",
                "1",
                "--iters",
                "1",
                "--threads",
                "1",
            ],
        )

    def test_perf_record_command_uses_requested_sampling_settings(self):
        perf = load_module()
        args = argparse.Namespace(
            perf_freq=999,
            perf_event="cpu-clock",
            call_graph="dwarf",
            model_dir=Path("assets"),
            scenario="single/long",
            warmup=1,
            iters=1,
            threads=1,
        )

        command = perf.perf_record_command(
            args,
            Path("target/release/benchmark_ltembed"),
            Path("perf-results/run/perf.data"),
        )

        self.assertEqual(
            command[:12],
            [
                "perf",
                "record",
                "-F",
                "999",
                "-e",
                "cpu-clock",
                "-g",
                "--call-graph",
                "dwarf",
                "--output",
                "perf-results/run/perf.data",
                "--",
            ],
        )
        self.assertIn("single/long", command)

    def test_format_command_failure_includes_stderr(self):
        perf = load_module()

        message = perf.format_command_failure(
            ["perf", "record", "--call-graph", "dwarf"],
            "perf_event_open(..., PERF_FLAG_FD_CLOEXEC) failed with unexpected error 1",
        )

        self.assertIn("perf record --call-graph dwarf", message)
        self.assertIn("perf_event_open", message)

    def test_extract_matrixmultiply_symbols_ignores_non_gemm_symbols(self):
        perf = load_module()
        report_text = """
  35.00%  benchmark_ltembed  benchmark_ltembed  [.] matrixmultiply::sgemm_kernel::kernel_target_neon
   9.00%  benchmark_ltembed  benchmark_ltembed  [.] matrixmultiply::gemm::gemm_loop
   7.00%  benchmark_ltembed  benchmark_ltembed  [.] ltembed::models::bert::masked_softmax
   5.00%  benchmark_ltembed  benchmark_ltembed  [.] matrixmultiply::sgemm_kernel::kernel_target_neon
"""

        symbols = perf.extract_matrixmultiply_symbols(report_text, limit=3)

        self.assertEqual(
            symbols,
            [
                "matrixmultiply::sgemm_kernel::kernel_target_neon",
                "matrixmultiply::gemm::gemm_loop",
            ],
        )

    def test_extract_matrixmultiply_symbols_strips_perf_ipc_columns(self):
        perf = load_module()
        report_text = """
    59.74%  [.] matrixmultiply::sgemm_kernel::kernel_target_neon                                                                                                                                                                                                               -      -
     4.43%  [.] matrixmultiply::gemm::gemm_loop                                                                                                                                                                                                                                -      -
"""

        symbols = perf.extract_matrixmultiply_symbols(report_text, limit=3)

        self.assertEqual(
            symbols,
            [
                "matrixmultiply::sgemm_kernel::kernel_target_neon",
                "matrixmultiply::gemm::gemm_loop",
            ],
        )


if __name__ == "__main__":
    unittest.main()
