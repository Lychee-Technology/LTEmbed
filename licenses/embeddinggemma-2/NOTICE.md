# EmbeddingGemma 2: license and attribution

The release tarball's `gguf_bundle/model.gguf` is EmbeddingGemma 2, a model by Google
DeepMind, as converted to GGUF and quantized by Unsloth. This directory holds the copy of
the model's license that the license requires a redistributor to give recipients, and the
model's attribution. It covers the model only, not LTEmbed's source code or anything else
in the tarball.

## The model file

- Model: [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2), by
  Google DeepMind.
- GGUF repository: [`unsloth/embeddinggemma-2-GGUF`](https://huggingface.co/unsloth/embeddinggemma-2-GGUF/tree/031f0d4b35536f69ab3509d4893c923264fcf253),
  revision `031f0d4b35536f69ab3509d4893c923264fcf253`, whose model card names
  `google/embeddinggemma-2` as its base model.
- File: `embeddinggemma-2-UD-Q5_K_XL.gguf`, sha256
  `a9d7a3b7eb421113f05e964e10882676f10fff83811b7876d5bb983db7c7ac41`.

LTEmbed does not modify the model. `scripts/assemble_gguf_bundle.py` downloads that file
from that revision, checks its sha256 and stores it unchanged as `model.gguf`. The bundle's
other files, `build-info.json`, `SHA256SUMS` and `provenance.json`, are LTEmbed's own,
written by the same script.

## License

The model is licensed under the Apache License, Version 2.0.
[LICENSE-APACHE-2.0.txt](./LICENSE-APACHE-2.0.txt) is the license text, unmodified, from
<https://www.apache.org/licenses/LICENSE-2.0.txt> (sha256
`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`).

What this rests on, as checked on 2026-10-14:

- The model cards of `unsloth/embeddinggemma-2-GGUF` at the revision above,
  `unsloth/embeddinggemma-2` at `c8f89cc7ecb4ba5122cd15fdd487422b2b487daf` and
  `google/embeddinggemma-2` at `914f7f89142e33e77833254d9c9b90c3cef7303b` declare
  `license: apache-2.0`.
- Their license link, <https://ai.google.dev/gemma/docs/gemma_4_license>, shows the Apache
  License 2.0. Its terms, sections 1 to 9, are word for word those of the file here; the
  page leaves out the appendix on how to apply the license.
- None of the three repositories has a `LICENSE` or `NOTICE` file at those revisions, so
  there is no upstream NOTICE file to pass on under section 4(d).

The model card also says: "Deployments must adhere to the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)."

## Redistributing

Section 4(a) of the license requires giving every recipient of the model a copy of the
license. The release tarball contains this directory, and `release-bundles.yml` fails if
the archive lacks either of its files. If you pass `gguf_bundle/` on without the rest of
the tarball, pass this directory on with it.

Naming Google DeepMind and Unsloth credits their work. It does not mean either endorses
LTEmbed.

## When the pinned model changes

When `scripts/assemble_gguf_bundle.py` pins another revision or quant for the release,
check the license of the new files and update this notice.
`tests/test_assemble_gguf_bundle.py` fails while this notice names another file, revision
or sha256 than the release quant's.
