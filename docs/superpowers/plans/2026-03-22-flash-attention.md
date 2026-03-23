# Flash Attention (CPU) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the materialized O(seq²) attention score matrix with a tiled flash attention kernel that keeps score tiles in L1 cache, eliminating ~270ms of DRAM traffic per forward pass at seq_len=128.

**Architecture:** New `src/flash_attn.rs` module containing `flash_attn_single_head` — a pure-safe-Rust tiled kernel using online softmax (Milakov & Gimelshein 2018). Tile size BR=BC=32; for head_dim=32 all tiles fit in L1. The kernel accepts raw pointers + stride so Q/K/V stay in their existing `[seq × hidden]` strided layout with no copy. `bert.rs` replaces the inner head loop with calls to this kernel for the contiguous-prefix-mask case; the existing sgemm fallback is preserved for non-contiguous masks.

**Tech Stack:** Rust, `matrixmultiply` (unchanged for projections), no new dependencies.

---

## File Map

| File | Change |
|------|--------|
| `src/flash_attn.rs` | **Create** — kernel + tests |
| `src/lib.rs` | **Modify** — add `pub(crate) mod flash_attn;` |
| `src/models/bert.rs` | **Modify** — replace head loop in `forward` and `forward_batch` |

---

## Background: Current Attention Code

In `src/models/bert.rs`, the attention inner loop (both `forward` lines 727–788 and `forward_batch` lines 964–1022) does for each head `h`:

```rust
// Step 1 — QK^T → scores [seq×seq], written to DRAM
matrixmultiply::sgemm(seq_len, head_dim, seq_len, scale,
    q_ptr.add(h * head_dim), hidden as isize, 1,   // Q row stride = hidden
    k_ptr.add(h * head_dim), 1, hidden as isize,   // K accessed transposed
    0.0, scores.as_mut_ptr(), seq_len as isize, 1);

// Step 2 — softmax per row, reads + writes scores from DRAM
for i in 0..seq_len { masked_softmax_active_prefix(&mut scores[i*seq_len..], active_len); }

// Step 3 — Attn×V → attn_out, reads scores from DRAM again
matrixmultiply::sgemm(seq_len, seq_len, head_dim, 1.0,
    scores.as_ptr(), seq_len as isize, 1,
    v_ptr.add(h * head_dim), hidden as isize, 1,
    0.0, attn_out_ptr.add(h * head_dim), hidden as isize, 1);
```

Score matrix size: seq=128 → 64KB/head; 12 heads × 12 layers × 3 passes = **27MB DRAM traffic per forward pass**. Flash attention eliminates this entirely.

---

## Task 1 — Create `src/flash_attn.rs` with failing tests

**Files:**
- Create: `src/flash_attn.rs`

### Flash attention algorithm summary (for implementer)

```
Constants: BR = 32  (row tile size)
           BC = 32  (col tile size)

fn flash_attn_single_head(q, k, v, o, stride, seq_len, head_dim, scale, active_len):
  // q/k/v/o: raw pointers into [seq × hidden] buffers
  // For token i, head h: element d is at ptr + i*stride + d
  // active_len: positions 0..active_len are unmasked; >= active_len are masked (-inf)

  Allocate (once per call, reused across tile iterations):
    q_tile [BR × head_dim]   — loaded contiguously from strided Q
    k_tile [BC × head_dim]   — loaded contiguously from strided K
    v_tile [BC × head_dim]   — loaded contiguously from strided V
    s_tile [BR × BC]         — score tile (never leaves L1)
    o_tile [BR × head_dim]   — output accumulator tile
    m_row  [BR]              — running row-max
    ell_row[BR]              — running row-sum of exp

  Zero output buffer o[0..seq_len*stride]

  For r_tile in 0..ceil(seq_len/BR):           // row tiles
    r_start = r_tile * BR
    r_end   = min(r_start + BR, seq_len)
    br_act  = r_end - r_start

    // Load Q tile: q_tile[i][d] = Q[r_start+i][d]  (copy from strided to contiguous)
    for i in 0..br_act:
      q_tile[i*head_dim..] = q[(r_start+i)*stride..(r_start+i)*stride+head_dim]

    m_row[..br_act]  = -inf
    ell_row[..br_act] = 0.0
    o_tile[..br_act*head_dim] = 0.0

    For c_tile in 0..ceil(seq_len/BC):          // col tiles
      c_start = c_tile * BC
      c_end   = min(c_start + BC, seq_len)
      bc_act  = c_end - c_start

      // Load K tile and V tile (contiguous copies from strided buffers)
      for j in 0..bc_act:
        k_tile[j*head_dim..] = k[(c_start+j)*stride..(c_start+j)*stride+head_dim]
        v_tile[j*head_dim..] = v[(c_start+j)*stride..(c_start+j)*stride+head_dim]

      // Compute score tile: s_tile[i][j] = dot(q_tile[i], k_tile[j]) * scale
      for i in 0..br_act:
        for d in 0..head_dim:                   // SAXPY-order: good cache reuse on s_tile[i]
          q_val = q_tile[i*head_dim + d]
          for j in 0..bc_act:
            s_tile[i*BC + j] += q_val * k_tile[j*head_dim + d]
        for j in 0..bc_act:
          s_tile[i*BC + j] *= scale
          if c_start + j >= active_len: s_tile[i*BC + j] = -inf   // mask

      // Online softmax update per row
      for i in 0..br_act:
        // New row max across this col tile
        m_tile_i = max over j in 0..bc_act of s_tile[i*BC + j]
        m_new = max(m_row[i], m_tile_i)

        // Rescale existing O and ell by exp(old_m - new_m)
        rescale = exp(m_row[i] - m_new)        // = 0.0 when m_row[i] = -inf (initial)
        o_tile[i*head_dim..][..head_dim] *= rescale
        ell_row[i] *= rescale

        // Accumulate P×V and update ell
        for j in 0..bc_act:
          p_j = if s_tile[i*BC+j] == -inf { 0.0 } else { exp(s_tile[i*BC+j] - m_new) }
          for d in 0..head_dim:
            o_tile[i*head_dim + d] += p_j * v_tile[j*head_dim + d]
          ell_row[i] += p_j

        m_row[i] = m_new

    // Write output row tile (divide by ell)
    for i in 0..br_act:
      inv_ell = if ell_row[i] > 0.0 { 1.0 / ell_row[i] } else { 0.0 }
      o[(r_start+i)*stride..][..head_dim] = o_tile[i*head_dim..][..head_dim] * inv_ell
```

**Important:** `-inf` detection uses `== f32::NEG_INFINITY` OR `<= SOFTMAX_EXP_CUTOFF`. Use `softmax_exp_cutoff(s - m_new)` where values below -12.0 are treated as 0 — matches the existing `softmax_exp` convention in bert.rs.

- [ ] **Step 1.1: Write the failing test**

  In `src/flash_attn.rs`, write `#[cfg(test)]` module. The test generates Q/K/V in the strided `[seq × hidden]` layout (stride = hidden), computes reference output using the existing sgemm + masked_softmax_active_prefix, then calls `flash_attn_single_head` and asserts epsilon-close:

  ```rust
  // src/flash_attn.rs
  pub(crate) const BR: usize = 32;
  pub(crate) const BC: usize = 32;

  /// Flash attention for one head.
  /// q, k, v, o: pointers into [seq × hidden] buffers at offset h*head_dim in each row.
  /// stride = hidden. active_len = unmasked prefix length (<= seq_len).
  pub(crate) unsafe fn flash_attn_single_head(
      q: *const f32,
      k: *const f32,
      v: *const f32,
      o: *mut f32,
      stride: usize,
      seq_len: usize,
      head_dim: usize,
      scale: f32,
      active_len: usize,
  ) {
      todo!()
  }

  #[cfg(test)]
  mod tests {
      use super::*;
      use approx::assert_relative_eq;

      fn patterned(n: usize) -> Vec<f32> {
          (0..n).map(|i| ((i as f32) * 0.1).sin()).collect()
      }

      /// Reference: sgemm QK^T + masked_softmax_active_prefix + sgemm Attn×V.
      fn reference_attn(
          q: &[f32], k: &[f32], v: &[f32],
          seq_len: usize, head_dim: usize, stride: usize,
          active_len: usize, h: usize, scale: f32,
      ) -> Vec<f32> {
          // Extract contiguous Q/K/V for head h
          let mut q_h = vec![0.0f32; seq_len * head_dim];
          let mut k_h = vec![0.0f32; seq_len * head_dim];
          let mut v_h = vec![0.0f32; seq_len * head_dim];
          for i in 0..seq_len {
              let src_off = i * stride + h * head_dim;
              q_h[i*head_dim..(i+1)*head_dim].copy_from_slice(&q[src_off..src_off+head_dim]);
              k_h[i*head_dim..(i+1)*head_dim].copy_from_slice(&k[src_off..src_off+head_dim]);
              v_h[i*head_dim..(i+1)*head_dim].copy_from_slice(&v[src_off..src_off+head_dim]);
          }

          let mut scores = vec![0.0f32; seq_len * seq_len];
          unsafe {
              matrixmultiply::sgemm(
                  seq_len, head_dim, seq_len, scale,
                  q_h.as_ptr(), head_dim as isize, 1,
                  k_h.as_ptr(), 1, head_dim as isize,
                  0.0, scores.as_mut_ptr(), seq_len as isize, 1,
              );
          }

          for row in 0..seq_len {
              let row_slice = &mut scores[row * seq_len..(row + 1) * seq_len];
              // mask suffix
              for j in active_len..seq_len {
                  row_slice[j] = f32::NEG_INFINITY;
              }
              // softmax
              let max = row_slice[..active_len]
                  .iter().cloned().fold(f32::NEG_INFINITY, f32::max);
              let mut sum = 0.0f32;
              for val in row_slice[..active_len].iter_mut() {
                  *val = (*val - max).exp();
                  sum += *val;
              }
              for val in row_slice[active_len..].iter_mut() { *val = 0.0; }
              if sum > 0.0 { for val in row_slice[..active_len].iter_mut() { *val /= sum; } }
          }

          let mut out = vec![0.0f32; seq_len * head_dim];
          unsafe {
              matrixmultiply::sgemm(
                  seq_len, seq_len, head_dim, 1.0,
                  scores.as_ptr(), seq_len as isize, 1,
                  v_h.as_ptr(), head_dim as isize, 1,
                  0.0, out.as_mut_ptr(), head_dim as isize, 1,
              );
          }
          out
      }

      fn run_flash_attn_test(seq_len: usize, num_heads: usize, head_dim: usize, active_len: usize) {
          let hidden = num_heads * head_dim;
          let q = patterned(seq_len * hidden);
          let k = patterned(seq_len * hidden);
          let v = patterned(seq_len * hidden);
          let scale = 1.0 / (head_dim as f32).sqrt();

          for h in 0..num_heads {
              let ref_out = reference_attn(&q, &k, &v, seq_len, head_dim, hidden, active_len, h, scale);

              let mut flash_out = vec![0.0f32; seq_len * hidden];
              unsafe {
                  flash_attn_single_head(
                      q.as_ptr().add(h * head_dim),
                      k.as_ptr().add(h * head_dim),
                      v.as_ptr().add(h * head_dim),
                      flash_out.as_mut_ptr().add(h * head_dim),
                      hidden, seq_len, head_dim, scale, active_len,
                  );
              }

              for i in 0..seq_len {
                  for d in 0..head_dim {
                      let got = flash_out[i * hidden + h * head_dim + d];
                      let exp = ref_out[i * head_dim + d];
                      assert_relative_eq!(got, exp, epsilon = 1e-4,
                          "Mismatch at seq={seq_len} h={h} i={i} d={d}");
                  }
              }
          }
      }

      #[test]
      fn test_flash_attn_e5_small_short() {
          // seq=7, fully unmasked (active_len=seq_len)
          run_flash_attn_test(7, 12, 32, 7);
      }

      #[test]
      fn test_flash_attn_e5_small_medium() {
          run_flash_attn_test(20, 12, 32, 20);
      }

      #[test]
      fn test_flash_attn_e5_small_medium_masked() {
          // 20 tokens, only 15 unmasked
          run_flash_attn_test(20, 12, 32, 15);
      }

      #[test]
      fn test_flash_attn_e5_small_long() {
          // seq=128 — exercises multiple tile rows and cols
          run_flash_attn_test(128, 12, 32, 128);
      }

      #[test]
      fn test_flash_attn_e5_small_long_masked() {
          run_flash_attn_test(128, 12, 32, 100);
      }

      #[test]
      fn test_flash_attn_seq_not_multiple_of_tile() {
          // seq=50: ceil(50/32)=2 row tiles, partial last tile
          run_flash_attn_test(50, 12, 32, 50);
      }

      #[test]
      fn test_flash_attn_seq_512() {
          run_flash_attn_test(512, 12, 32, 512);
      }
  }
  ```

- [ ] **Step 1.2: Run tests to verify they fail**

  ```bash
  cargo test flash_attn 2>&1 | head -30
  ```
  Expected: compile error (`todo!()` panics or function body missing).

---

## Task 2 — Implement `flash_attn_single_head`

**Files:**
- Modify: `src/flash_attn.rs`

- [ ] **Step 2.1: Add the `SOFTMAX_EXP_CUTOFF` constant and helper**

  Flash attention needs to match the numerical conventions in `bert.rs`. Add at the top of `src/flash_attn.rs`:

  ```rust
  // Matches bert.rs SOFTMAX_EXP_CUTOFF to avoid denormals
  const EXP_CUTOFF: f32 = -12.0;

  #[inline(always)]
  fn safe_exp(shifted: f32) -> f32 {
      if shifted <= EXP_CUTOFF { 0.0 } else { shifted.exp() }
  }
  ```

- [ ] **Step 2.2: Replace `todo!()` with the full implementation**

  ```rust
  pub(crate) unsafe fn flash_attn_single_head(
      q: *const f32,
      k: *const f32,
      v: *const f32,
      o: *mut f32,
      stride: usize,
      seq_len: usize,
      head_dim: usize,
      scale: f32,
      active_len: usize,
  ) {
      // Tile sizes (clamped to seq_len for short sequences)
      let br = BR.min(seq_len);
      let bc = BC.min(seq_len);

      // Scratch buffers — allocated once, reused across all tile iterations
      let mut q_tile = vec![0.0f32; br * head_dim];
      let mut k_tile = vec![0.0f32; bc * head_dim];
      let mut v_tile = vec![0.0f32; bc * head_dim];
      let mut s_tile = vec![0.0f32; br * bc];
      let mut o_tile = vec![0.0f32; br * head_dim];
      let mut m_row = vec![f32::NEG_INFINITY; br];
      let mut ell_row = vec![0.0f32; br];

      // Zero the output buffer for this head (only head_dim elements per row)
      for i in 0..seq_len {
          let o_row = std::slice::from_raw_parts_mut(o.add(i * stride), head_dim);
          o_row.fill(0.0);
      }

      let num_row_tiles = seq_len.div_ceil(br);
      for r_tile in 0..num_row_tiles {
          let r_start = r_tile * br;
          let br_act = (r_start + br).min(seq_len) - r_start;

          // Load Q tile (strided → contiguous)
          for i in 0..br_act {
              let src = std::slice::from_raw_parts(q.add((r_start + i) * stride), head_dim);
              q_tile[i * head_dim..(i + 1) * head_dim].copy_from_slice(src);
          }

          // Reset per-row-tile state
          m_row[..br_act].fill(f32::NEG_INFINITY);
          ell_row[..br_act].fill(0.0);
          o_tile[..br_act * head_dim].fill(0.0);

          let num_col_tiles = seq_len.div_ceil(bc);
          for c_tile in 0..num_col_tiles {
              let c_start = c_tile * bc;
              let bc_act = (c_start + bc).min(seq_len) - c_start;

              // Load K tile and V tile (strided → contiguous)
              for j in 0..bc_act {
                  let k_src = std::slice::from_raw_parts(k.add((c_start + j) * stride), head_dim);
                  k_tile[j * head_dim..(j + 1) * head_dim].copy_from_slice(k_src);
                  let v_src = std::slice::from_raw_parts(v.add((c_start + j) * stride), head_dim);
                  v_tile[j * head_dim..(j + 1) * head_dim].copy_from_slice(v_src);
              }

              // Compute score tile: SAXPY-order for cache reuse on s_tile row
              s_tile[..br_act * bc].fill(0.0);
              for i in 0..br_act {
                  for d in 0..head_dim {
                      let q_val = q_tile[i * head_dim + d];
                      for j in 0..bc_act {
                          s_tile[i * bc + j] += q_val * k_tile[j * head_dim + d];
                      }
                  }
                  // Scale and mask in one pass
                  for j in 0..bc_act {
                      let s = s_tile[i * bc + j] * scale;
                      s_tile[i * bc + j] = if c_start + j >= active_len {
                          f32::NEG_INFINITY
                      } else {
                          s
                      };
                  }
              }

              // Online softmax update per row i
              for i in 0..br_act {
                  // Row max for current tile
                  let m_tile_i = s_tile[i * bc..i * bc + bc_act]
                      .iter()
                      .cloned()
                      .fold(f32::NEG_INFINITY, f32::max);
                  let m_new = m_row[i].max(m_tile_i);

                  // Rescale previous accumulator
                  let rescale = safe_exp(m_row[i] - m_new);
                  if rescale != 1.0 {
                      for d in 0..head_dim {
                          o_tile[i * head_dim + d] *= rescale;
                      }
                      ell_row[i] *= rescale;
                  }

                  // Accumulate P × V
                  for j in 0..bc_act {
                      let p_j = safe_exp(s_tile[i * bc + j] - m_new);
                      if p_j > 0.0 {
                          for d in 0..head_dim {
                              o_tile[i * head_dim + d] += p_j * v_tile[j * head_dim + d];
                          }
                          ell_row[i] += p_j;
                      }
                  }

                  m_row[i] = m_new;
              }
          } // col tiles

          // Write normalized output for this row tile
          for i in 0..br_act {
              let inv_ell = if ell_row[i] > 0.0 { 1.0 / ell_row[i] } else { 0.0 };
              let o_row = std::slice::from_raw_parts_mut(o.add((r_start + i) * stride), head_dim);
              for d in 0..head_dim {
                  o_row[d] = o_tile[i * head_dim + d] * inv_ell;
              }
          }
      } // row tiles
  }
  ```

- [ ] **Step 2.3: Register the module in `src/lib.rs`**

  Add `pub(crate) mod flash_attn;` after `pub(crate) mod gemm;`:

  ```rust
  pub(crate) mod flash_attn;
  pub(crate) mod gemm;
  ```

- [ ] **Step 2.4: Run flash_attn tests**

  ```bash
  cargo test flash_attn -- --nocapture 2>&1
  ```
  Expected: all `test_flash_attn_*` tests pass.

- [ ] **Step 2.5: Commit**

  ```bash
  git add src/flash_attn.rs src/lib.rs
  git commit -m "feat: add flash_attn_single_head kernel with online softmax (closes #97 step 1)"
  ```

---

## Task 3 — Wire flash attention into `bert.rs` `forward`

**Files:**
- Modify: `src/models/bert.rs:727-788`

The head loop in `forward` currently lives inside a match on `attention_prefix_len`:
```
if let Some(active_len) = attention_prefix_len {
    if active_len == seq_len { ... } else { ... }
} else { ... /* full per-element mask */ }
```

Replace the `Some(active_len)` branch with `flash_attn_single_head` calls:

- [ ] **Step 3.1: Replace the entire `for h in 0..num_heads` block in `forward` (lines 727–789)**

  The actual code structure (verified in the file) has **one single `for h` loop** containing all three steps — QK^T sgemm, `if let Some / else` softmax dispatch, and Attn×V sgemm — all unconditionally in the same loop body. The `if let Some(active_len)` is inside the head loop, not wrapping it.

  Keep lines 723–725 (scale computation + the two `.fill(0.0)` calls) exactly as-is. Replace only lines 727–789 (the entire head loop) with:

  ```rust
  // lines 727–789 replaced:
  if let Some(active_len) = attention_prefix_len {
      // Flash attention: score tiles stay in L1, no DRAM traffic for score matrix
      for h in 0..num_heads {
          unsafe {
              crate::flash_attn::flash_attn_single_head(
                  sc.q.as_ptr().add(h * head_dim),
                  sc.k.as_ptr().add(h * head_dim),
                  sc.v.as_ptr().add(h * head_dim),
                  sc.attn_out.as_mut_ptr().add(h * head_dim),
                  hidden,
                  seq_len,
                  head_dim,
                  scale,
                  active_len,
              );
          }
      }
  } else {
      // Fallback: non-contiguous attention mask (rare in practice)
      for h in 0..num_heads {
          sc.scores[..seq_sq].fill(0.0);
          unsafe {
              matrixmultiply::sgemm(
                  seq_len, head_dim, seq_len, scale,
                  sc.q.as_ptr().add(h * head_dim), hidden as isize, 1,
                  sc.k.as_ptr().add(h * head_dim), 1, hidden as isize,
                  0.0f32,
                  sc.scores.as_mut_ptr(), seq_len as isize, 1,
              );
          }
          for i in 0..seq_len {
              masked_softmax(
                  &mut sc.scores[i * seq_len..(i + 1) * seq_len],
                  attention_mask,
              );
          }
          unsafe {
              matrixmultiply::sgemm(
                  seq_len, seq_len, head_dim, 1.0f32,
                  sc.scores.as_ptr(), seq_len as isize, 1,
                  sc.v.as_ptr().add(h * head_dim), hidden as isize, 1,
                  0.0f32,
                  sc.attn_out.as_mut_ptr().add(h * head_dim), hidden as isize, 1,
              );
          }
      }
  }
  ```

  `sc.scores` and `sc.attn_out` remain in `Scratch` unchanged — the flash path uses `sc.attn_out` for output (zeroed inside the kernel per token row), and the fallback still needs `sc.scores`.

- [ ] **Step 3.2: Verify line 724 is preserved**

  Line 724 (`sc.attn_out[..seq_hidden].fill(0.0)`) must remain in place. It is harmless for the flash path (the kernel overwrites) and required for the fallback (the Attn×V sgemm uses `beta=0.0` which overwrites per-head, but the fill is a cheap safety net).

- [ ] **Step 3.3: Run tests**

  ```bash
  cargo test 2>&1 | tail -20
  ```
  Expected: all tests pass, including `test_embed_returns_unit_vector`, `test_embed_dimension`.

- [ ] **Step 3.4: Commit**

  ```bash
  git add src/models/bert.rs
  git commit -m "perf: use flash attention in forward (single-sequence path)"
  ```

---

## Task 4 — Wire flash attention into `bert.rs` `forward_batch`

**Files:**
- Modify: `src/models/bert.rs:964-1023`

The batch path has a nearly identical head loop inside `for batch_idx in 0..batch_size`.

- [ ] **Step 4.1: Replace the head loop in `forward_batch`**

  Inside `for batch_idx in 0..batch_size { ... for h in 0..num_heads { ... } }`, replace the `Some(batch_mask_prefix_len)` inner head loop:

  ```rust
  // Before (~line 964):
  for h in 0..num_heads {
      scores.fill(0.0);
      unsafe { matrixmultiply::sgemm( /* QK^T */ ); }
      /* softmax branches */
      unsafe { matrixmultiply::sgemm( /* Attn×V */ ); }
  }

  // After:
  let batch_active_len = batch_mask_prefix_len.unwrap_or(seq_len);
  // Use flash attention when mask is a contiguous prefix (or fully unmasked)
  if batch_mask_prefix_len.is_some() {
      for h in 0..num_heads {
          unsafe {
              crate::flash_attn::flash_attn_single_head(
                  q.as_ptr().add(hidden_offset + h * head_dim),
                  k.as_ptr().add(hidden_offset + h * head_dim),
                  v.as_ptr().add(hidden_offset + h * head_dim),
                  attn_out.as_mut_ptr().add(hidden_offset + h * head_dim),
                  hidden,
                  seq_len,
                  head_dim,
                  scale,
                  batch_active_len,
              );
          }
      }
  } else {
      // Fallback: full per-element mask (non-contiguous, rare)
      for h in 0..num_heads {
          scores.fill(0.0);
          unsafe { matrixmultiply::sgemm( /* QK^T, unchanged */ ); }
          for i in 0..seq_len { masked_softmax(&mut scores[...], batch_mask); }
          unsafe { matrixmultiply::sgemm( /* Attn×V, unchanged */ ); }
      }
  }
  ```

  The `attn_out[hidden_offset..hidden_offset+seq_hidden].fill(0.0)` — flash attention zeros only the head's slice per call (the `o.fill(0.0)` inside the kernel). Add an explicit `attn_out[hidden_offset..hidden_offset+seq_hidden].fill(0.0)` before the head loop to match the existing behavior of the `attn_out.fill(0.0)` that was called outside the batch loop before.

  Specifically, the existing code has `attn_out.fill(0.0)` outside the `batch_idx` loop (~line 956). Change it to zero per-batch-item inside the loop instead (or verify the existing fill is sufficient — it zeros the whole buffer once per layer, which is fine).

- [ ] **Step 4.2: Run full test suite**

  ```bash
  cargo test 2>&1 | tail -25
  ```
  Expected: all 47+ tests pass, including `test_embed_batch_matches_individual` and `test_embed_batch_mixed_lengths_matches_individual`.

- [ ] **Step 4.3: Run clippy**

  ```bash
  cargo clippy -- -D warnings 2>&1
  ```
  Expected: no warnings.

- [ ] **Step 4.4: Commit**

  ```bash
  git add src/models/bert.rs
  git commit -m "perf: use flash attention in forward_batch (batch path)"
  ```

---

## Task 5 — Verify and push

- [ ] **Step 5.1: Run the full test suite one final time**

  ```bash
  cargo test 2>&1 | grep -E "^test result|FAILED"
  ```
  Expected: `test result: ok` for all suites, 0 failed.

- [ ] **Step 5.2: Build release binary (sanity check)**

  ```bash
  cargo build --release --bin benchmark_ltembed 2>&1 | tail -5
  ```
  Expected: `Compiling ltembed ... Finished`.

- [ ] **Step 5.3: Push**

  ```bash
  git push
  ```

---

## Correctness notes for implementer

- **Online softmax initialisation:** `m_row[i] = f32::NEG_INFINITY`, `ell_row[i] = 0.0`. On the first col tile, `rescale = exp(-inf - m_new) = 0.0`, so the previous empty o_tile and ell are correctly zeroed out.
- **Fully-masked row:** If all `seq_len` positions are masked (`active_len = 0`), all scores are -inf, `ell_row[i]` stays 0, `inv_ell = 0`, output row is zero. This is correct — padding positions get zero embeddings.
- **`safe_exp` vs `f32::exp`:** Use `safe_exp(shifted)` everywhere in the kernel to stay consistent with `softmax_exp` in bert.rs and avoid denormals.
- **Partial tiles:** `br_act = actual rows in this row tile ≤ BR`. All loops over tile elements use `br_act`/`bc_act`, not `BR`/`BC`. The `s_tile` buffer is still sized `BR × BC`; only `[..br_act × bc_act]` is valid.
- **The `None` fallback:** Non-contiguous attention masks (e.g. `[1,0,1]`) are theoretically possible but never produced by the BERT tokenizer for normal inputs. The fallback path is kept for correctness but is not performance-critical.
