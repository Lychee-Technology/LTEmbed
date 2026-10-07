# Script Agent Guidance

These instructions apply to Python utilities, benchmark orchestration, reports, and reference tooling.

Also follow `/AGENTS.md`.

## Script Roles

Important tools include:

```text
generate_fixtures.py
generate_token_ids.py
bench_pytorch.py
run_embedding_benchmarks.py
compare_benchmarks.py
render_benchmark_report.py
write_benchmark_metadata.py
```

Understand whether a script is a correctness reference, fixture generator, benchmark orchestrator, benchmark comparison tool, or report generator. Do not blur these roles.

## Benchmark Reproducibility

Preserve:

- deterministic benchmark inputs where intended;
- warm vs cold distinction;
- requested and actual thread configuration;
- implementation identity;
- model/quant metadata;
- output dimension;
- normalization configuration.

Never improve benchmark numbers by changing workload semantics.

## Correctness vs Performance

Keep correctness/parity results separate from latency/throughput results.

Quantization comparisons must not silently redefine the reference baseline.

The PyTorch/F32 golden remains independent of the GGUF implementation.

## Historical Backend Terminology

Some scripts or comments may still contain ONNX-era terminology.

Do not assume such wording reflects the current runtime. When touching related current code, verify terminology against the present llama.cpp/GGUF implementation and clean up stale descriptions where appropriate.

Historical measurement data itself should not be rewritten.

## Python Tests

When modifying a Python utility that has corresponding tests under `tests/`, run those focused tests in addition to repository-wide Rust validation when applicable.
