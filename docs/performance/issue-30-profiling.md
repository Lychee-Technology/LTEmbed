# Issue 30 Profiling Notes

## Scope

This note captures the reproducible profiling and benchmark workflow used for issue `#30` on macOS ARM64, plus the current hotspot summary after adding the fused masked-softmax path.

## Representative Commands

### Warm benchmark, all scenarios

Baseline workspace:

```bash
cargo run --release --bin benchmark_ltembed -- \
  --mode warm \
  --model-dir assets \
  --warmup 10 \
  --iters 30
```

Worktree under active development:

```bash
cargo run --release --bin benchmark_ltembed -- \
  --mode warm \
  --model-dir /Users/ruoshi/code/github/LTEmbed/assets \
  --warmup 10 \
  --iters 30
```

### Warm benchmark, single scenario

```bash
cargo run --release --bin benchmark_ltembed -- \
  --mode warm \
  --scenario single/medium \
  --model-dir /Users/ruoshi/code/github/LTEmbed/assets \
  --warmup 10 \
  --iters 30
```

This single-scenario mode is the new profiling-oriented entrypoint added in this change.

### Text hotspot summary with `sample`

```bash
target/release/benchmark_ltembed \
  --mode warm \
  --scenario single/long \
  --model-dir /Users/ruoshi/code/github/LTEmbed/assets \
  --warmup 5 \
  --iters 200 >/tmp/issue30-bench.json &

pid=$!
sample "$pid" 5 -mayDie -file /tmp/issue30-sample.txt
wait "$pid"
```

The `sample` output is a good fallback when Instruments export is unstable inside the current machine configuration.

### macOS trace artifact attempt

Attempted:

```bash
xcrun xctrace record \
  --template 'Time Profiler' \
  --output /tmp/issue30-single-long.trace \
  --time-limit 8s \
  --launch -- \
  target/release/benchmark_ltembed \
    --mode warm \
    --scenario single/long \
    --model-dir /Users/ruoshi/code/github/LTEmbed/assets \
    --warmup 1 \
    --iters 30
```

Result on this machine: `xctrace` crashed in an Instruments plugin before writing a `.trace` bundle. Because of that toolchain instability, the fallback evidence for this run is:

- `sample` text profile at `/tmp/issue30-sample.txt`
- scenario benchmark JSON at `/tmp/issue30-bench.json`

## Current Benchmark Evidence

The main comparison from the full warm run:

| Scenario | Baseline mean ms | Current mean ms | Delta |
|---|---:|---:|---:|
| `single/long` | 296.349 | 290.976 | `-1.81%` |
| `single/medium` | 23.666 | 23.877 | `+0.89%` |
| `batch/medium/8` | 105.074 | 113.437 | noisy / regressed in this run |

Interpretation:

- the fused masked-softmax change shows a measurable win on the longer sequence path where attention score post-processing is amplified
- the shorter and batched scenarios were noisier on this machine and should be re-measured on a quieter runner before making stronger claims

## Hotspot Summary

`sample` on `single/long` still shows that the dominant time remains in BERT forward compute, with the biggest families being:

| Rank | Hotspot family | Evidence from `/tmp/issue30-sample.txt` | Interpretation |
|---|---|---|---|
| 1 | `matrixmultiply::sgemm_kernel::kernel_target_neon` | repeated under multiple `Bert::forward` subtrees such as `+6336`, `+6924`, `+5416`, `+3104`, `+2628` | dense projections and attention/value matmuls remain the largest compute bucket |
| 2 | `tanhf` | strongest scalar leaf under `Bert::forward +6684` | GELU remains a major scalar cost |
| 3 | `expf` in the attention score path | visible under `Bert::forward +4372`, `+4452`, `+4480`, `+4544`, `+4580`, `+4608` | masked softmax is still a real scalar hotspot, but now handled in one fused row pass |
| 4 | `ltembed::models::bert::layer_norm_rows` | visible under `Bert::forward +6116` and `+7624` | layer norm still appears as a remaining scalar cleanup target |

### Hotspot Chart

The chart below uses representative subtree sample counts from the `sample` report. These counts are not exclusive percentages; they are a quick way to rank hotspot families from the text profile.

```mermaid
xychart-beta
    title "Issue 30 Hotspot Families (sample, single/long)"
    x-axis ["sgemm", "tanhf", "softmax-expf", "layer-norm"]
    y-axis "Observed subtree sample count" 0 --> 1900
    bar [1870, 519, 272, 54]
```

## Code Change Tied To This Profiling Pass

The optimization landed in [`src/models/bert.rs`](/Users/ruoshi/code/github/LTEmbed/.worktrees/issue-30-scalar-hotspots/src/models/bert.rs):

- added `masked_softmax`
- removed the separate column-wise padding-mask write loop
- normalized each attention row while zeroing masked positions in the same helper
- reused the helper in both `Bert::forward` and `Bert::forward_batch`

This reduces passes over the attention score rows and is the code change associated with the `single/long` improvement above.
