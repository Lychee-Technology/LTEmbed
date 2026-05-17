# LTEmbed Backend Branch Split Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split LTEmbed into a `matrixmultiply` branch that preserves the legacy matrixmultiply backend line and a `main` branch that remains ONNXRuntime-only.

**Architecture:** Use `neon-8x12-ci` as the base for the new `matrixmultiply` branch because it is the only remaining LTEmbed branch that still contains the matrixmultiply backend code path. Keep `main` aligned with `origin/main` as the ORT-only line, and selectively cherry-pick only benchmark improvements that remain useful for matrixmultiply instead of merging unrelated ORT bundle work.

**Tech Stack:** Git branches/worktrees, Rust, GitHub Actions workflows, Python benchmark harnesses.

---

## Scope

### In scope
- Create a local `matrixmultiply` branch from `neon-8x12-ci`
- Selectively integrate matrixmultiply-relevant benchmark commits from `main-benchmark-fix`
- Preserve the current `main` line as ORT-only by realigning local `main` to `origin/main` with a backup pointer
- Verify the resulting split with git and content checks

### Out of scope
- Removing matrixmultiply code from every non-main branch
- Rewriting ORT benchmark workflows to support both backends in one branch
- Pushing or opening PRs unless requested later
- Full code cleanup of `main` beyond restoring it to the ORT-only remote state

## File / Branch Structure

### Branches involved
- Source branch for matrixmultiply line: `neon-8x12-ci`
- New branch to create: `matrixmultiply`
- Benchmark-history source for selective cherry-picks: `main-benchmark-fix`
- ORT-only canonical base: `origin/main`
- Local backup to create before moving `main`: `backup/local-main-9bb3dd3`

### Key code paths expected after split
- `matrixmultiply` branch should still contain:
  - `Cargo.toml` dependency on `matrixmultiply`
  - `src/gemm.rs`
  - `src/models/bert.rs`
  - matrixmultiply-specific benchmark and profiling harness support
- `main` should remain ORT-only and still contain:
  - `src/engine.rs`
  - ORT bundle workflows
  - ORT benchmark harness path

## Task 1: Create isolated worktree for branch split work

**Files:**
- Modify: none

- [ ] **Step 1: Create a dedicated LTEmbed worktree from the current repo**

Run:
```bash
git worktree add ".worktrees/matrixmultiply-split" -b "matrixmultiply-split"
```

Expected: new worktree at `LTEmbed/.worktrees/matrixmultiply-split`

- [ ] **Step 2: Verify the worktree starts clean**

Run:
```bash
git -C ".worktrees/matrixmultiply-split" status --short
```

Expected: no output

## Task 2: Create the `matrixmultiply` branch from the correct base

**Files:**
- Modify: branch refs only

- [ ] **Step 1: Write the failing structural check**

Run:
```bash
git -C ".worktrees/matrixmultiply-split" rev-parse --verify matrixmultiply
```

Expected: FAIL because the branch does not exist yet

- [ ] **Step 2: Create `matrixmultiply` from `neon-8x12-ci`**

Run:
```bash
git -C ".worktrees/matrixmultiply-split" checkout -b matrixmultiply neon-8x12-ci
```

Expected: branch `matrixmultiply` now points at the same commit as `neon-8x12-ci`

- [ ] **Step 3: Verify backend-specific files are present**

Run:
```bash
git -C ".worktrees/matrixmultiply-split" grep -n "matrixmultiply" -- Cargo.toml src/gemm.rs src/models/bert.rs scripts/run_embedding_benchmarks.py .github/workflows/benchmark-arm64.yml
```

Expected: matches in all of the listed matrixmultiply-specific paths

## Task 3: Selectively absorb matrixmultiply-useful benchmark improvements

**Files:**
- Modify: selected files from cherry-picked commits, likely among:
  - `.github/workflows/benchmark-arm64.yml`
  - `scripts/run_embedding_benchmarks.py`
  - `scripts/README.md`
  - `src/bin/benchmark_ltembed.rs`
  - `tests/test_benchmark_orchestrator.py`
  - `tests/data/jane-austen_pride-and-prejudice-retrieval_eval.json`

- [ ] **Step 1: Cherry-pick the benchmark workflow fix commit**

Run:
```bash
git -C ".worktrees/matrixmultiply-split" cherry-pick fadf7a4cf94b384af53c5c60c4626f996af5da89
```

Expected: clean cherry-pick or a small, resolvable conflict limited to benchmark workflow files

- [ ] **Step 2: Cherry-pick benchmark flag alignment fix**

Run:
```bash
git -C ".worktrees/matrixmultiply-split" cherry-pick 5fdba36f298ec5369bfcb1d6c211dbe4a5dd3369
```

Expected: benchmark harness flags become consistent

- [ ] **Step 3: Cherry-pick retrieval evaluation additions in chronological order**

Run:
```bash
git -C ".worktrees/matrixmultiply-split" cherry-pick f8eaa8e4307d37930176e35a96e7bc4ff95f9e53 0c7af6fef2b5e14753ce500204e99c95b8b4e6e8 206fe417ed832bb2dcc95c10138be188c07f3a83
```

Expected: retrieval-eval cases and related benchmark harness updates land on `matrixmultiply`

- [ ] **Step 4: Stop if cherry-picks conflict with ORT-only bundle assumptions**

If conflicts touch files that are clearly ORT-bundle-only, inspect with:
```bash
git -C ".worktrees/matrixmultiply-split" status --short
git -C ".worktrees/matrixmultiply-split" diff --name-only --diff-filter=U
```

Resolve only the matrixmultiply-useful portions; do not bring in ORT bundle asset assumptions by accident.

## Task 4: Preserve the local-only `main` commit and realign `main`

**Files:**
- Modify: branch refs only

- [ ] **Step 1: Verify local `main` still has one unique commit**

Run:
```bash
git rev-list --left-right --count main...origin/main
git log --oneline origin/main..main
```

Expected:
- first command prints `1 5`
- second command shows `9bb3dd3 feat: implement benchmark builder bundle with ORT support`

- [ ] **Step 2: Create a backup pointer for the local-only `main` commit**

Run:
```bash
git branch backup/local-main-9bb3dd3 main
```

Expected: backup branch now points at the old local `main`

- [ ] **Step 3: Move local `main` to the remote ORT-only line**

Run:
```bash
git branch -f main origin/main
```

Expected: local `main` now matches `origin/main`

- [ ] **Step 4: Verify the move preserved the old commit on the backup branch**

Run:
```bash
git log --oneline --decorate -1 main
git log --oneline --decorate -1 backup/local-main-9bb3dd3
```

Expected:
- `main` points to `0d40f3a`
- backup branch points to `9bb3dd3`

## Task 5: Verify the backend split

**Files:**
- Modify: none

- [ ] **Step 1: Verify `matrixmultiply` branch still exposes matrixmultiply paths**

Run:
```bash
git -C ".worktrees/matrixmultiply-split" grep -n "matrixmultiply" -- Cargo.toml src .github scripts tests
```

Expected: matches remain in code, workflow, or benchmark harness files tied to the matrixmultiply backend

- [ ] **Step 2: Verify local `main` is ORT-only with respect to code paths**

Run:
```bash
git grep -n "matrixmultiply" main -- Cargo.toml src .github scripts tests || true
git grep -n "ort::\|model.ort\|ORT_BUNDLE_DIR" main -- Cargo.toml src .github scripts tests
```

Expected:
- first command prints no code-path hits that represent active matrixmultiply backend support in `main`
- second command prints ORT-related matches in runtime and workflow files

- [ ] **Step 3: Verify branch pointers and worktrees**

Run:
```bash
git branch --format='%(refname:short)|%(objectname:short)|%(subject)'
git worktree list
```

Expected:
- `matrixmultiply` exists
- `main` points at the ORT-only remote lineage
- worktree list shows the dedicated `matrixmultiply-split` worktree if kept

## Success Criteria

- A local `matrixmultiply` branch exists and is based on `neon-8x12-ci`
- `matrixmultiply` still contains the matrixmultiply backend files and benchmark harness path
- selected benchmark improvements from `main-benchmark-fix` are absorbed without dragging in unrelated ORT bundle-only behavior
- local `main` is realigned to `origin/main`
- the previous local-only `main` commit is preserved on `backup/local-main-9bb3dd3`

## Risks and Mitigations

- Cherry-picked benchmark commits may assume ORT bundles
  - Mitigation: stop on conflicts and keep only matrixmultiply-relevant pieces
- Repointing `main` could orphan the local-only commit
  - Mitigation: create `backup/local-main-9bb3dd3` first
- Current root worktree is dirty
  - Mitigation: perform branch-creation work in a separate worktree and avoid switching the main working directory branch
