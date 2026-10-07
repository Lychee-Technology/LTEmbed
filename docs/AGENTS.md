# Documentation Agent Guidance

These instructions apply to current project documentation.

Also follow `/AGENTS.md`.

## Primary Rule

Documentation outside `docs/history/` should describe the current `main` branch unless explicitly labeled otherwise.

Implementation and current CI behavior take precedence over older prose.

## Current Documentation

`docs/README.md` is the per-task index of current documentation. When behavior changes, use it, together with the root `README.md`, to find the documents that describe that behavior, and update them in the same change.

When adding, renaming, or removing a document, update `docs/README.md` as well.

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

- root `README.md`: project entry point and minimum working path;
- `docs/README.md`: per-task index of current documentation;
- other files in `docs/`: detailed operational and reference documentation;
- `docs/history/<category>/`: superseded plans, migrations, experiments, and their measurements.

Do not leave superseded implementation plans beside current getting-started material without clear status.

## Link Validation

When moving, renaming, or removing documentation, find and repair every reference to the old path: links in the root `README.md`, `docs/README.md`, other docs, `scripts/README.md`, `AGENTS.md` files, and code comments. `git grep -n '<old file name>'` finds them.

Relative links resolve from the directory of the file that contains them, so check each one from its own location.
