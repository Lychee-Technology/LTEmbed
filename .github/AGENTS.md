# GitHub Automation Agent Guidance

These instructions apply to CI, GitHub Actions, repository automation, artifact fetching, and releases.

Also follow `/AGENTS.md`.

## CI Is an Operational Source of Truth

Current workflow files are authoritative for:

- runner architecture;
- Rust toolchain setup;
- static llama.cpp artifact acquisition;
- artifact checksum verification;
- model-backed test bundle construction;
- actual CI test commands.

Do not copy artifact versions or hashes from old documentation when the workflow provides current values.

## Static llama.cpp Artifacts

The release pin is set in more than one workflow, and the asset name and artifact contract are also checked outside the workflows. `docs/development.md` § "Static llama.cpp artifacts" lists every location and what must change together when bumping the release; follow it rather than editing one copy.

Preserve repository-pinned artifact identity, SHA verification, and artifact-contract validation.

Do not weaken integrity checks to make setup easier.

## Model-Backed CI

The CI Test job sets:

```text
LTEMBED_REQUIRE_TEST_BUNDLE=1
```

so bundle-gated integration tests fail instead of silently skipping if `LTEMBED_TEST_BUNDLE_DIR` or required files are missing.

Preserve that fail-closed behavior when changing bundle assembly or test invocation.

`docs/testing.md` § "What CI runs" and `docs/development.md` § "Checks before pushing" mirror the CI test commands, and `/AGENTS.md` uses the latter as the baseline for local validation. Update both when you add, remove, or change a CI check.

## CI Changes

When modifying a workflow:

- consider cache behavior;
- keep ARM64 requirements explicit;
- preserve reproducibility;
- avoid silently skipping model-backed validation;
- verify shell commands with `set -euo pipefail` semantics where applicable.

## Release Bundles

Treat the current release workflow as authoritative for distributable bundle assembly.

Keep release metadata consistent with the runtime GGUF contract.

When bundle structure changes, review the runtime loader, CI bundle generation, release bundle generation, tests, and documentation together.

## Deployment Claims

Distinguish between a bundle fitting a deployment size limit, release artifacts being generated, and a fully supported deployment workflow.

Do not claim the third merely because the first two are true.
