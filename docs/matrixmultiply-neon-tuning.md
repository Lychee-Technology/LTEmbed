# matrixmultiply NEON Kernel Tuning Investigation

## Problem statement

LTEmbed on ARM64 reaches ~11.7 GFLOPS effective throughput on the projection/FFN
GEMMs (m=128, k=384, n=384 or 1536). PyTorch/OpenBLAS reaches ~18.4 GFLOPS for the
same shapes — a 57% gap that accounts for all the difference in benchmark results.
Profiling (`profile-arm64` workflow, 2026-03-23) confirmed:

- **single/long**: 70.9% of CPU time in `matrixmultiply::sgemm_kernel::kernel_target_neon`
- **IPC 2.24** (single/long), **1.57** (single/medium)
- **L1 miss rate 2.1%** (single/long) — packing is working, not memory-bound
- Conclusion: compute-bound on SGEMM kernel throughput, not cache misses or layout

---

## Current kernel: matrixmultiply 0.3.10

### Micro-kernel tile

```
MR = 8, NR = 8   (defined in sgemm_kernel.rs, KernelNeon)
```

The kernel computes an 8×8 f32 tile as four 4×4 quadrants using `vfmaq_laneq_f32`.

**Per k-step inner loop (src: `kernel_target_neon`):**

```
4 × vld1q_f32       (load a1, a2, b1, b2 — 4 float32x4_t)
16 × vfmaq_laneq_f32 (4 quadrants × 4 rows = 16 FMA ops)
2 × pointer bumps
```

**Register use:**
- 16 accumulator registers: `ab11[4]`, `ab12[4]`, `ab21[4]`, `ab22[4]`
- 4 temporary registers: a1, a2, b1, b2
- Total: 20/32 NEON registers — **12 registers idle**

### Blocking parameters (archparam_defaults.rs)

```
S_MC = 64    rows of A packed at once
S_KC = 256   depth (inner dimension) of each packed strip
S_NC = 1024  columns of B packed at once
```

For our main GEMM shape (m=128, k=384, n=384):
- A-pack covers full m=128 in 2 MC-strips (64 rows each)
- B-pack covers k=384 in 2 KC-strips (256 depth each), n=384 fits in 1 NC-strip
- A-pack size: 64 × 256 × 4B = 64 KB (fits in typical 256 KB L2)
- B-pack size: 256 × 384 × 4B = 384 KB (L2 miss on cores with ≤384 KB L2)

---

## IPC analysis

On Neoverse-N1 / Cortex-A78 (likely CI runner microarchitecture):

| Resource | Throughput |
|---|---|
| FMLA (128-bit) | 2 per cycle |
| Load | 2 per cycle |

Per k-step: 4 loads + 16 FMAs = 20 instructions.

- **FMA-bound rate**: 16 FMAs / 2 per cycle = 8 cycles/k-step
- **Expected IPC at FMA bound**: 20 insns / 8 cycles = **2.5**
- **Measured IPC**: 2.24 (single/long) → **90% of FMA-bound ceiling**

The kernel is already very close to the throughput limit of its own micro-kernel design.
The gap vs OpenBLAS is not execution efficiency within the kernel — it's that the kernel
computes fewer FLOPs per k-step due to the 12 idle registers.

---

## Why wider tile = more FLOPs/cycle

A wider micro-kernel amortizes the same loop overhead over more FMA work:

| Kernel | Accs (registers) | FMAs/k-step | Idle regs | FLOPs/k-step |
|---|---|---|---|---|
| matrixmultiply 8×8 | 16 | 16 | 12 | 16 |
| BLIS-style 8×12 | 24 | 24 | 4 | 24 |
| BLIS-style 12×8 | 24 | 24 | 4 | 24 |

An 8×12 kernel (8 rows × 12 cols in f32, = 8 rows × 3 groups of f32x4):
- Accumulator registers: 8 rows × 3 col-groups = 24
- Input registers: 2 a-vectors (8 f32 = 2 × float32x4) + 3 b-vectors = 5
- Total: 29/32 — fits with 3 registers spare
- FMAs per k-step: 8×3 = 24 — **50% more work per iteration vs current**

OpenBLAS uses an 8×12 micro-kernel for Neoverse-N1 (see `kernel/arm64/SGEMM_*`),
which is why it achieves ~1.5× the throughput.

---

## Knobs available without forking

### 1. Blocking parameter overrides (zero-cost experiment)

matrixmultiply reads `MATMUL_SGEMM_NC/KC/MC` at compile time via `option_env!`.
Set via `RUSTFLAGS` or a `.cargo/config.toml` `[env]` section:

```toml
# .cargo/config.toml
[env]
MATMUL_SGEMM_KC = "512"
MATMUL_SGEMM_MC = "128"
```

**Hypothesis**: the default KC=256 causes a B-pack of 256×384×4B=384 KB which may
spill from L2 on the CI runner. Increasing KC to 384 (= full k for our shapes) would
eliminate the KC loop entirely for single-k GEMMs, reducing loop overhead but increasing
B-pack size to 576 KB. Whether this wins or loses depends on the runner's L2/L3 sizes.

Alternatively, reducing KC to 128 might improve L2 hit rate if L2 ≤ 256 KB.

These can be benchmarked without any code change — just rerun `benchmark-arm64` with
different `RUSTFLAGS` values passed as a workflow input.

### 2. Upstream issue / PR to matrixmultiply

The 12-idle-register gap is a known limitation of the current 8×8 design. An 8×12
kernel would be a backward-compatible improvement for AArch64. Options:

- File an issue at `https://github.com/bluss/matrixmultiply` with this analysis
- Submit a PR adding a `KernelNeon12` (MR=8, NR=12) with the wider tile
- If upstream is slow, vendor a patched copy under `vendor/matrixmultiply/`

### 3. Replace with a local NEON SGEMM for the specific shapes we use

LTEmbed calls SGEMM with a small set of shapes:
- Projections: (m, 384, 384) and (m, 384, 1536) / (m, 1536, 384) for FFN, where m = seq_len × batch_size
- Attention: (seq_len, head_dim=32, seq_len) × num_heads=12

For these specific shapes, a hand-tuned inline kernel with MR=8, NR=12 or MR=6, NR=16
could be added alongside matrixmultiply and selected by a shape threshold. This avoids
the dependency on upstream but adds maintenance burden.

---

## Recommended path

**Step 1 (cheap, this week):** Experiment with blocking params via RUSTFLAGS in the
`benchmark-arm64` workflow. Add a `rustflags` workflow input and test KC=128, KC=384,
MC=128. Expected outcome: small (<5%) win or neutral — this fixes packing efficiency
but not register underutilization.

**Step 2 (medium effort):** File an upstream issue with this analysis. The data is
compelling: 12 idle registers, 90% FMA efficiency already, clear path to 8×12. If
upstream is receptive, a PR is straightforward since the macros in `aarch64/macros.rs`
already support this pattern.

**Step 3 (if upstream is slow, ~2–4 weeks):** Vendor a patched matrixmultiply with
an 8×12 NEON kernel behind a Cargo feature flag. Keep the crates.io version as default
so library users aren't affected without opt-in.

---

## What we will NOT do

- Flash attention: bottleneck is compute throughput, not DRAM bandwidth
- Contiguous Q/K/V layout: attention is ~5% of runtime at seq=128
- SIMD GELU/LayerNorm: not measurable in profiles
- OpenBLAS as default dependency: adds a C toolchain requirement that conflicts with
  the library's pure-Rust portability goal; keep as a feature flag if added at all
