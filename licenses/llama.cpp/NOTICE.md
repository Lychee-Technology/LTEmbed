# llama.cpp and static artifact attribution

LTEmbed links the prebuilt `libllama.a`, `libggml.a`, `libggml-cpu.a` and
`libggml-base.a` archives through raw FFI. The current CI pin is
`Lychee-Technology/static-llama-cpp-rs-builder` release **`v0.1.159-2`**, compiling
**llama.cpp `b11577`** (embedded version `0.6.0-dev`) at commit
**`cbe28bb3cb0758f0c566a9a3561766821aeb0748`**, as both variants' `build-info.json`
records. These are separate third-party materials, not relicensed under LTEmbed's
`MIT OR Apache-2.0` source license.

## llama.cpp and ggml

- Upstream: <https://github.com/ggml-org/llama.cpp>
- Pinned source: <https://github.com/ggml-org/llama.cpp/tree/cbe28bb3cb0758f0c566a9a3561766821aeb0748>
- License source: <https://raw.githubusercontent.com/ggml-org/llama.cpp/cbe28bb3cb0758f0c566a9a3561766821aeb0748/LICENSE>
- Preserved license: [LICENSE-MIT.txt](./LICENSE-MIT.txt), sha256
  `94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d`
- Supplied copyright: **Copyright (c) 2023-2026 The ggml authors**.

The upstream project license is MIT. The static artifact's contract identifies
this license as covering its llama.cpp and ggml components; the artifact ships
the same text as `LICENSES/llama.cpp.LICENSE`. The copy here preserves that full
copyright, permission notice and warranty disclaimer without changes.

## Static artifact builder

The artifact also supplies a separate MIT license for builder materials:

- Repository: <https://github.com/Lychee-Technology/static-llama-cpp-rs-builder>
- Release: <https://github.com/Lychee-Technology/static-llama-cpp-rs-builder/releases/tag/v0.1.159-2>
- Builder commit recorded in both variants' `build-info.json`:
  **`cafd3d62998f50464ee25497593d3592ce13a8db`**.
- License source: <https://raw.githubusercontent.com/Lychee-Technology/static-llama-cpp-rs-builder/cafd3d62998f50464ee25497593d3592ce13a8db/LICENSE>
- Preserved license: [LICENSE-BUILDER-MIT.txt](./LICENSE-BUILDER-MIT.txt), sha256
  `02df995a98a425ea888076c5c62189c2e67a63e3b3197a93375a87a2367ed12d`
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

The release workflow builds a source-plus-model archive from `git archive`, not a
compiled LTEmbed executable or a bundle of the native static archives, so this
directory reaches release tarballs as part of the source tree. Binary distributors
must include the applicable notices explicitly and review any additional components
they ship.

These copies are not an exhaustive audit of every vendored source, Rust dependency
or system library; preserve any additional applicable third-party notices. On an
artifact upgrade, check the actual compiled llama.cpp commit, builder provenance
and supplied license files, and update these copies and attribution together
([docs/development.md](../../docs/development.md#static-llamacpp-artifacts)).
`tests/test_reference_generators.py` fails while this notice names another release
or llama.cpp commit than `.github/workflows/ci.yml`, or while a copy's sha256 differs
from the one recorded above. Attribution does not imply upstream endorsement. These
MIT terms cover llama.cpp/ggml and the builder materials only; the model's terms are
in [../embeddinggemma-2/NOTICE.md](../embeddinggemma-2/NOTICE.md).
