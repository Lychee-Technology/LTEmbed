# Documentation Agent Guidance

These instructions apply to current project documentation.

Also follow `/AGENTS.md`.

## Primary Rule

Documentation outside `docs/history/` should describe the current `main` branch unless explicitly labeled otherwise.

Implementation and current CI behavior take precedence over older prose.

## Current Documentation

When behavior changes, inspect the relevant current documents, including as applicable:

```text
README.md
docs/design.md
docs/development.md
docs/integ-test.md
scripts/README.md
```

Avoid duplicating the same contract in many places. Prefer one authoritative detailed document and link to it from higher-level guides.

## Examples and Commands

Current documentation must use real public APIs, binaries/examples, environment variables, Cargo commands, and repository paths.

Do not invent commands to make a workflow appear complete.

Prefer examples derived from `examples/api_usage.rs`.

## Backend Terminology

The current `main` backend is llama.cpp/GGUF.

References such as these require scrutiny outside historical material:

```text
OnnxEngine
OnnxEngineConfig
model.ort
ort_bundle
libonnxruntime.so
ORT_DYLIB_PATH
matrixmultiply
spike_llama
--features llama
```

They may describe history, but must not accidentally present an obsolete path as current usage.

## Documentation Organization

Use progressive disclosure for human readers too:

- root README: project entry point and minimum working path;
- current `docs/`: detailed operational/reference documentation;
- `docs/performance/`: current relevant performance analysis;
- `docs/history/`: superseded plans, migrations, and experiments.

Do not leave superseded implementation plans beside current getting-started material without clear status.

## Link Validation

When moving or renaming documentation, inspect and repair relative links, including references from the root README, other docs, scripts README, and relevant code comments.
