# Historical Documentation Agent Guidance

These instructions apply to historical decision records, migration documents, superseded plans, and branch-specific material.

Also follow `/AGENTS.md` and `/docs/AGENTS.md`, except where this file intentionally narrows documentation rules.

## Preserve Historical Context

Historical documents are records of what was believed, tested, or planned at a particular time.

Do not rewrite them so they appear to have originally described today's architecture.

For example, an ONNX migration document may legitimately contain `OnnxEngine`, `model.ort`, or `ORT_DYLIB_PATH` when those terms are part of the historical record.

## Add Status, Don't Rewrite History

Every historical document starts with a status banner. Match the banners already in `docs/history/`:

```markdown
> [!NOTE]
> **Status:** Historical (superseded)\
> **Applies to:** the former ONNX Runtime backend, kept on the `ort` branch\
> **Current implementation:** llama.cpp/GGUF on `main`\
> **Current documentation:** [docs/architecture.md](../../architecture.md)
```

The trailing backslashes keep the fields on separate lines. The link is relative to the document: `../../architecture.md` is correct for `docs/history/<category>/<name>.md`, the layout all historical documents use; adjust the depth for any other location.

Below the banner, preserve the original technical narrative where practical.

When adding a historical document, put it in a category directory under `docs/history/` and list it in `docs/README.md` § "History".

## Obsolete Commands

Commands that only worked for a past branch or throwaway spike may remain when historically relevant, but must not be presented as current reproduction instructions.

Label them clearly.

## Plans

For completed or abandoned implementation plans, mark status explicitly as `Completed`, `Superseded`, `Abandoned`, or `Branch-specific`.

Do not leave an old unchecked task list looking like current project work.
