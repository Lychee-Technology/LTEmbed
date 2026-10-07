# Historical Documentation Agent Guidance

These instructions apply to historical decision records, migration documents, superseded plans, and branch-specific material.

Also follow `/AGENTS.md` and `/docs/AGENTS.md`, except where this file intentionally narrows documentation rules.

## Preserve Historical Context

Historical documents are records of what was believed, tested, or planned at a particular time.

Do not rewrite them so they appear to have originally described today's architecture.

For example, an ONNX migration document may legitimately contain `OnnxEngine`, `model.ort`, or `ORT_DYLIB_PATH` when those terms are part of the historical record.

## Add Status, Don't Rewrite History

Prefer adding a short status banner such as:

```markdown
> [!NOTE]
> **Status:** Historical / Superseded
> **Applies to:** former ONNX Runtime backend
> **Current implementation:** llama.cpp/GGUF on `main`
> **Current documentation:** ../../architecture.md
```

Then preserve the original technical narrative where practical.

## Obsolete Commands

Commands that only worked for a past branch or throwaway spike may remain when historically relevant, but must not be presented as current reproduction instructions.

Label them clearly.

## Plans

For completed or abandoned implementation plans, mark status explicitly as `Completed`, `Superseded`, `Abandoned`, or `Branch-specific`.

Do not leave an old unchecked task list looking like current project work.
