# Jina model attribution and redistribution notice

> [!NOTE]
> **LTEmbed no longer ships this model.** Since [#179](https://github.com/Lychee-Technology/LTEmbed/pull/179), `main` and the releases built from
> it use EmbeddingGemma 2, under Apache-2.0 ([its license and attribution](../embeddinggemma-2/NOTICE.md)).
> This notice applies to the Jina model in the `v0.1.0` release asset described below,
> and to anyone who redistributes a copy of that asset. The `v0.1.0` GitHub release and
> its asset have since been deleted; the `v0.1.0` tag remains and holds source code only.
> Later release tarballs contain this directory only because they contain the source
> tree; they do not contain the Jina model.

This directory preserves upstream terms and attribution for
**`jinaai/jina-embeddings-v5-text-nano-retrieval`**, published by **Jina AI**.
It is not a new license grant, a guarantee of redistribution clearance, or a
license for LTEmbed source code or its other dependencies. No LTEmbed software
license is selected by these files; LTEmbed's original source code is separately
dual-licensed under MIT OR Apache-2.0 as described in the project README and root
`LICENSE-MIT` and `LICENSE-APACHE` files. Third-party materials retain their
applicable upstream terms.

## Jina model license and attribution

Jina declares the model licensed under **Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**:
<https://creativecommons.org/licenses/by-nc/4.0/>. The complete official legal
text is preserved in `LICENSE-CC-BY-NC-4.0.txt`, downloaded from
<https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt>.

The public grant permits sharing and adaptations for **NonCommercial purposes
only**, subject to its conditions and limited to rights the licensor can grant.
NonCommercial means "not primarily intended for or directed towards commercial
advantage or monetary compensation"; a free GitHub download is not by itself
proof of compliance. Commercial use is not granted by this public model license.
For commercial permission, contact **[sales@jina.ai](mailto:sales@jina.ai)** and
confirm coverage of the intended public redistribution and downstream use, not
only internal inference.

`MODEL-CARD.md` is an unmodified copy of Jina's retrieval model card at revision
**`ac5d898c8d382b17167c33e5c8af644a3519b47d`**. It retains the supplied license,
provenance and citation identifying Mohammad Kalim Akram, Saba Sturua, Nastia
Havriushenko, Quentin Herreros, Michael Günther, Maximilian Werk and Han Xiao,
*jina-embeddings-v5-text: Task-Targeted Embedding Distillation* (2026),
<https://arxiv.org/abs/2602.15547>.

The card is upstream documentation, not LTEmbed API instructions. Its original
links and examples are preserved byte-for-byte; any original relative links
resolve against the pinned upstream repository, not this local directory.
Direct upstream sources:

- Repository: <https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval>
- Pinned repository: <https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval/tree/ac5d898c8d382b17167c33e5c8af644a3519b47d>
- Original card: <https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval/blob/ac5d898c8d382b17167c33e5c8af644a3519b47d/README.md>
- Exact card download: <https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval/raw/ac5d898c8d382b17167c33e5c8af644a3519b47d/README.md>
- Official Q5_K_M GGUF: <https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval/resolve/ac5d898c8d382b17167c33e5c8af644a3519b47d/v5-nano-retrieval-Q5_K_M.gguf>
- Tokenizer: <https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval/resolve/ac5d898c8d382b17167c33e5c8af644a3519b47d/tokenizer.json>
- Upstream file and hash records: <https://huggingface.co/api/models/jinaai/jina-embeddings-v5-text-nano-retrieval/tree/ac5d898c8d382b17167c33e5c8af644a3519b47d?recursive=true&expand=false>

## Exact audited bundle and packaging changes

The following unchanged-content statement applies **only to the audited LTEmbed
`v0.1.0` Q5_K_M bundle**, published until the release's deletion as
`https://github.com/Lychee-Technology/LTEmbed/releases/download/v0.1.0/ltembed-0.1.0-jinaai__jina-embeddings-v5-text-nano-retrieval-Q5_K_M-linux-arm64.tar.gz`.
The recorded asset SHA-256 is
`34998a20b0cb068cc12c8950f90b1bfa075be7e1e9fa41d0cff01b9fd7eab326`.
The completed audit found these model/tokenizer bytes identical to the pinned
official files:

| Upstream file | Bundled file | Bytes | SHA-256 |
|---|---|---:|---|
| `v5-nano-retrieval-Q5_K_M.gguf` | `model.gguf` | 168,910,912 | `46fbc0423862cb6a5d4ff776d885f349d2a87c36d821dd5630f9fa184c9b4b92` |
| `tokenizer.json` | `tokenizer.json` | 17,210,235 | `98d4a1d32152d6cedf85b5e88f3b205106dca1fe72aaab34e0ac13c238421069` |

For that audited bundle, LTEmbed did not train, convert or requantize the model:
it redistributed Jina's official Q5_K_M GGUF unchanged in content, renamed it to
`model.gguf`, kept `tokenizer.json` unchanged, and added `build-info.json`,
packaging and checksums. Jina's preceding task-specific model development and
published GGUF conversion/quantization are upstream changes, not LTEmbed work;
the upstream card retains their description and base-model relationship.

**Every future redistribution must verify these same hashes before reusing this
unchanged-content statement.** If bytes or revisions differ, review their terms
and provenance and update the notice to describe actual changes. This directory
does not amend that archive, which shipped without these files, and the release
workflow does not package the Jina model.

## EuroBERT base and retained Apache attribution

The documented base chain is retrieval model →
`jinaai/jina-embeddings-v5-text-nano` → `EuroBERT/EuroBERT-210m`.
The reviewed source snapshots below document that relationship; they are **not
asserted to be the exact historical training or conversion input revisions**:

- Jina nano card: <https://huggingface.co/jinaai/jina-embeddings-v5-text-nano/raw/8a7f00aac812071b69403df470f1038ec85f8925/README.md>
- EuroBERT-210m card: <https://huggingface.co/EuroBERT/EuroBERT-210m/raw/39b51e15dd1f1a06f58b5cbf6a8a188cec60bd0e/README.md>
- EuroBERT training repository: <https://github.com/Nicolas-BZRD/EuroBERT/tree/55a8722aec57142a35631124e5de44304ffdcf17>
- Exact EuroBERT license source: <https://raw.githubusercontent.com/Nicolas-BZRD/EuroBERT/55a8722aec57142a35631124e5de44304ffdcf17/LICENSE.TXT>

EuroBERT declares its model architectures, weights and training codebase under
Apache-2.0. `LICENSE-EUROBERT-APACHE-2.0.txt` preserves the upstream `LICENSE.TXT`
verbatim, including its metadata and **`Copyright [2025] [Nicolas BOIZARD]`**.
This copy preserves terms for inherited EuroBERT material to which Apache-2.0
applies; **it does not relicense Jina's finished model as Apache-2.0, remove its
NonCommercial restriction, or establish a tokenizer-specific grant**.

The EuroBERT card's citation credits Nicolas Boizard, Hippolyte
Gisserot-Boukhlef, Duarte M. Alves, André Martins, Ayoub Hammal, Caio Corro,
Céline Hudelot, Emmanuel Malherbe, Etienne Malaboeuf, Fanny Jourdan, Gabriel
Hautreux, João Alves, Kevin El-Haddad, Manuel Faysse, Maxime Peyrard, Nuno M.
Guerreiro, Patrick Fernandes, Ricardo Rei and Pierre Colombo;
<https://arxiv.org/abs/2503.05500>.

Jina's upstream EuroBERT Python files also retain this supplied notice:
"Copyright 2025 Nicolas Boizard, Duarte M. Alves, Hippolyte Gisserot-Boukhlef
and the EuroBert team. All rights reserved."
Those Python files are not components of the audited runtime bundle:

- <https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval/raw/ac5d898c8d382b17167c33e5c8af644a3519b47d/configuration_eurobert.py>
- <https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval/raw/ac5d898c8d382b17167c33e5c8af644a3519b47d/modeling_eurobert.py>

## Tokenizer provenance remains unresolved

The pinned Jina and EuroBERT tokenizer records share the SHA-256 above.
EuroBERT's official tutorial describes using a Llama tokenizer:
<https://raw.githubusercontent.com/Nicolas-BZRD/EuroBERT/55a8722aec57142a35631124e5de44304ffdcf17/examples/continuous_pretraining.ipynb>.
The exact original Meta version/revision, file-specific redistribution grant and
any supplemental authorization have not been established. Do not infer that the
tokenizer is conclusively CC-only or Apache-only, or substitute guessed
version-specific Meta notices. Obtain Jina/EuroBERT clarification for this exact
hash and preserve any confirmed applicable terms before claiming clearance.

## Disclaimers and no endorsement

The CC legal text, Section 5, and Apache text, Sections 7–8, retain their
applicable as-is warranty disclaimers and limitations of liability, including
title and non-infringement disclaimers. This notice does not change those terms
or grant additional rights. Attribution does not imply sponsorship, endorsement
or official status from Jina AI, EuroBERT, Meta or any other upstream party.

## Checklist before redistribution

- [ ] Record the actual release purpose and whether it meets NonCommercial.
  Where commercial or uncertain, obtain written Jina permission covering public
  redistribution, the GGUF/tokenizer, any modifications and intended downstream
  use. These companion files alone do not clear a release.
- [ ] Obtain upstream clarification of the exact tokenizer hash's original
  source/version, file-specific rights, supplemental permissions and required
  notices; add any confirmed applicable license/notice materials.
- [ ] Download model, tokenizer and card from the pinned revision, verify model
  and tokenizer against the hashes above, and verify the three preserved
  upstream texts against their official sources. Review any changed inputs and
  record actual modifications before reusing this notice.
- [ ] Copy this **whole directory beside the model** when redistributing, for
  example as `gguf_bundle/licenses/jina-embeddings-v5-text-nano-retrieval/`, so
  copying the bundle retains all four companions. Include all companions in the
  package's checksum manifest and verify the extracted archive and checksums.
- [ ] Keep model/tokenizer terms separate from LTEmbed source licensing and
  other component obligations; do not advertise the whole package as
  commercially permissive or imply upstream endorsement.
