# matrixmultiply NEON 8x12 Opt-In Implementation Plan

> For agentic workers: execute this plan with TDD where practical. Keep the default `matrixmultiply` AArch64 NEON behavior unchanged unless the opt-in feature is enabled.

## Goal

Add an opt-in AArch64 NEON 8x12 SGEMM kernel to `matrixmultiply`, keep the existing 8x8 kernel as the default, and benchmark the LTEmbed-shaped GEMMs locally against the baseline.

## Architecture

The implementation adds a second explicit AArch64 f32 kernel, `KernelNeon12`, with `MR=8` and `NR=12`. Runtime dispatch remains unchanged by default: on `aarch64 + neon`, `matrixmultiply` still selects the existing 8x8 `KernelNeon`. When a new cargo feature is enabled, dispatch selects the 8x12 kernel instead.

The new kernel shares small helper macros and structural logic with the existing 8x8 NEON path, but remains a separate kernel implementation rather than a fully generic `const NR` template. That keeps the register-blocking structure legible and minimizes risk while still avoiding unnecessary duplication.

## Scope

### In scope

- Add a cargo feature gating the 8x12 kernel
- Add the `U12` const-width type needed by the kernel trait
- Implement `KernelNeon12` in `matrixmultiply` for AArch64 NEON f32 SGEMM
- Keep default behavior on AArch64 unchanged unless the feature is enabled
- Add tests for the new kernel path
- Run local correctness tests and LTEmbed-shaped benchmarks
- Compare baseline 8x8 vs opt-in 8x12, with optional block-size sweeps

### Out of scope

- Changing the default AArch64 kernel selection
- Adding runtime CPU-model-specific dispatch
- Adding a shape-specialized LTEmbed-only kernel
- Modifying LTEmbed code
- Upstream issue/PR preparation

## Files To Modify

- `matrixmultiply/Cargo.toml`
  - Add the opt-in cargo feature, tentatively `neon-8x12`
- `matrixmultiply/src/kernel.rs`
  - Add `U12` and its `ConstNum` implementation
- `matrixmultiply/src/sgemm_kernel.rs`
  - Add `KernelNeon12`
  - Gate dispatch on the new feature
  - Add kernel tests for the 8x12 path
  - Factor only the obvious shared NEON helper logic
- `matrixmultiply/src/lib.rs`
  - Document the experimental opt-in feature in crate docs
- `matrixmultiply/benches/ltembed_shapes.rs`
  - Reuse as-is unless a tiny labeling tweak helps benchmark output readability

## Design Decisions

### 1. Feature-gated opt-in

Default behavior stays on the current 8x8 kernel.

- Without `neon-8x12`: `detect()` chooses `KernelNeon`
- With `neon-8x12`: `detect()` chooses `KernelNeon12`

This keeps the existing portability and performance profile untouched for normal users while allowing direct A/B benchmarking.

### 2. Explicit 8x12 kernel, not a fully generic template

The 8x8 kernel has a `2 x 2` block layout of 4x4 accumulators. The new 8x12 kernel uses a `2 x 3` block layout. Because accumulator shape, beta handling, and C load/store structure all change together, the first version should remain explicit.

Shared pieces may still be extracted locally:

- 4-lane outer-product accumulate macro
- 4x4 load/store helpers for strided `C`
- alpha/beta combine helpers where it improves readability

### 3. Dispatch policy

Do not add per-microarchitecture runtime dispatch in this iteration. The crate currently detects only ISA features, not CPU model. A feature flag is the simplest safe switch for experimentation.

## Implementation Tasks

### Task 1: Add feature plumbing

- Add `neon-8x12 = []` to `matrixmultiply/Cargo.toml`
- Add a short docs note in `matrixmultiply/src/lib.rs`
- Add `U12` to `matrixmultiply/src/kernel.rs`

Verification:

- `cargo test --no-run`
- `cargo test --features neon-8x12 --no-run`

### Task 2: Add a failing test for the new kernel type

Before implementing `KernelNeon12`, add a kernel self-test entry in `matrixmultiply/src/sgemm_kernel.rs` for the feature-gated path.

Expected red phase:

- `cargo test --features neon-8x12 neon8x12 -- --nocapture`
- It should fail because `KernelNeon12` does not exist yet

### Task 3: Implement `KernelNeon12`

Add a new AArch64 NEON SGEMM kernel with:

- `MR = 8`
- `NR = 12`
- 24 accumulator registers arranged as 6 groups of `float32x4_t`
- 5 temporary vector registers: `a1`, `a2`, `b1`, `b2`, `b3`

Inner loop structure per `k` step:

- load `a1`, `a2`
- load `b1`, `b2`, `b3`
- update `ab11`, `ab12`, `ab13`, `ab21`, `ab22`, `ab23`
- advance packed A/B pointers by `MR` / `NR`

Preserve the current transpose-friendly behavior when `rsc == 1`.

Verification:

- `cargo test --features neon-8x12 neon8x12 -- --nocapture`
- `cargo test --features neon-8x12`

### Task 4: Gate dispatch

Update `detect()` in `matrixmultiply/src/sgemm_kernel.rs`:

- default: `KernelNeon`
- with `feature = "neon-8x12"`: `KernelNeon12`

Verification:

- `cargo test`
- `cargo test --features neon-8x12`

### Task 5: Benchmark locally

Run the LTEmbed-shaped benchmark on the local machine.

Baseline:

- `cargo bench --bench ltembed_shapes`

Opt-in 8x12:

- `cargo bench --features neon-8x12 --bench ltembed_shapes`

Optional block-size sweeps, if `constconf` is enabled:

- `cargo bench --features "constconf neon-8x12" --bench ltembed_shapes`
- `MATMUL_SGEMM_KC=128 cargo bench --features "constconf neon-8x12" --bench ltembed_shapes`
- `MATMUL_SGEMM_KC=384 cargo bench --features "constconf neon-8x12" --bench ltembed_shapes`
- `MATMUL_SGEMM_MC=128 cargo bench --features "constconf neon-8x12" --bench ltembed_shapes`
- `MATMUL_SGEMM_KC=384 MATMUL_SGEMM_MC=128 cargo bench --features "constconf neon-8x12" --bench ltembed_shapes`

Record at least:

- `proj_384_384`
- `ffn_up`
- `ffn_down`
- whether `attn_qk` / `attn_sv` changed materially

## Success Criteria

- Default AArch64 NEON builds still use the existing 8x8 kernel
- `neon-8x12` builds compile and pass tests
- Local benchmarks complete successfully
- Benchmark output clearly shows baseline vs opt-in results
- If present, block-size sweeps are reported separately from the kernel comparison

## Risks And Mitigations

- 8x12 may help some ARM cores more than others
  - Mitigation: feature-gated opt-in, default unchanged
- General-stride `C` handling is easy to get wrong with the wider tile
  - Mitigation: reuse existing kernel structure and run full `cargo test`
- Over-abstracting the kernel can make debugging harder
  - Mitigation: keep the main 8x12 kernel explicit, share only small helpers

## Benchmark Reporting Format

When execution finishes, report:

- test commands run and whether they passed
- benchmark commands run
- the most important GFLOPS or timing deltas for `proj_384_384`, `ffn_up`, and `ffn_down`
- whether the optional block-size sweeps changed the conclusion
