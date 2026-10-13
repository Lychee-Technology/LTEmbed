# llama.cpp and static artifact attribution

LTEmbed links the prebuilt `libllama.a`, `libggml.a`, `libggml-cpu.a` and
`libggml-base.a` archives through raw FFI. The current CI pin is
`Lychee-Technology/static-llama-cpp-rs-builder` release **`v0.1.159-1`**, compiling
**llama.cpp `v0.6.0`** at commit
**`d81235049384534c167caea52b85a694f6103d14`**. These are separate third-party
materials, not relicensed under LTEmbed's `MIT OR Apache-2.0` source license.

## llama.cpp and ggml

- Upstream: <https://github.com/ggml-org/llama.cpp>
- Pinned source: <https://github.com/ggml-org/llama.cpp/tree/d81235049384534c167caea52b85a694f6103d14>
- License source: <https://raw.githubusercontent.com/ggml-org/llama.cpp/d81235049384534c167caea52b85a694f6103d14/LICENSE>
- Preserved license: [LICENSE-MIT.txt](./LICENSE-MIT.txt)
- Supplied copyright: **Copyright (c) 2023-2026 The ggml authors**.

The upstream project license is MIT. The static artifact's contract identifies
this license as covering its llama.cpp and ggml components; the artifact ships
the same text as `LICENSES/llama.cpp.LICENSE`. The copy here preserves that full
copyright, permission notice and warranty disclaimer without changes.

## Static artifact builder

The artifact also supplies a separate MIT license for builder materials:

- Repository: <https://github.com/Lychee-Technology/static-llama-cpp-rs-builder>
- Release: <https://github.com/Lychee-Technology/static-llama-cpp-rs-builder/releases/tag/v0.1.159-1>
- Builder commit recorded in both variants' `build-info.json`:
  **`e2a4568909524d14f70149c2132ea9064b54fcdc`**.
- License source: <https://raw.githubusercontent.com/Lychee-Technology/static-llama-cpp-rs-builder/e2a4568909524d14f70149c2132ea9064b54fcdc/LICENSE>
- Preserved license: [LICENSE-BUILDER-MIT.txt](./LICENSE-BUILDER-MIT.txt)
- Supplied copyright: **Copyright (c) 2026 Lychee Technology Inc.**

This is the unmodified text shipped as
`LICENSES/static-llama-cpp-rs-builder.LICENSE`. Keep it when redistributing the
artifact or its builder materials, including the supplied build-script material.

## Redistribution

MIT permits commercial use and redistribution; its copyright and permission
notice must accompany copies or substantial portions of the covered software.
**Static linking does not remove this requirement:** distributing an LTEmbed
executable that contains llama.cpp/ggml code must retain the upstream license and
copyright notice in the accompanying distribution. Keep this directory with
binary packages and preserve the artifact's supplied `LICENSES/` when distributing
the static artifact itself. Do not replace upstream notices with LTEmbed's own
MIT copyright notice.

The current release workflow creates a source-plus-model archive, not a compiled
LTEmbed executable or a bundle of the native static archives. This change records
their licenses in the source tree; it does not implement binary-release packaging,
alter artifact pins, or modify already-published assets. Binary distributors must
include the applicable notices explicitly and review any additional components
they ship.

These copies are not an exhaustive audit of every vendored source, Rust dependency
or system library; preserve any additional applicable third-party notices. On an
artifact upgrade, check the actual compiled llama.cpp commit, builder provenance
and supplied license files, and update these copies and attribution together.
Attribution does not imply upstream endorsement. The MIT terms do not remove the
separate NonCommercial restriction on the Jina model or resolve tokenizer rights.
