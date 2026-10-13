#!/usr/bin/env python3
"""Check that relative Markdown links and AGENTS.md path references resolve.

Runs offline over the files Git tracks and exits 1 if any reference is broken.

1. Links. The target of every inline link, image and reference definition in a tracked
   `*.md` file must be a tracked file or directory, unless it has a URL scheme (`https:`,
   `mailto:`). A target that starts with `/` resolves from the repository root, as on
   GitHub; any other target resolves from the directory of the file that contains it, and
   a target with no path (`#anchor`, or an empty `()`) is that file itself. A `#fragment`
   on a link to a Markdown file must match one of that file's headings, slugged the way
   GitHub slugs them. Links with a URL scheme are skipped, never fetched. Link syntax in a
   code span or code block, fenced or indented, is code, not a link. HTML `<a href>` links
   are not checked, and HTML `<a id>` anchors are not headings.
2. AGENTS.md paths. Every path-like token in a code span or code block of an `AGENTS.md`
   file must be a tracked file or directory, and every `file` § "Section" reference must
   match a heading of that file. A token is path-like if it contains `/` or ends in a file
   extension. Tokens in NOT_REPO_PATHS are runtime or historical names, not repository
   paths, and are skipped.

Check 2 covers only `AGENTS.md` files, so the historical documents under `docs/history/`,
which may name files that no longer exist, get only the link check.
"""

import argparse
import fnmatch
import os
import posixpath
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]

# Tokens that AGENTS.md files put in backticks but that are not repository paths.
NOT_REPO_PATHS = frozenset(
    {
        # The runtime GGUF bundle (docs/bundle-format.md) and the model's tokenizer.json, which
        # the tokenizer tests compare its vocabulary with: built locally or by CI, never
        # committed.
        "gguf_bundle/",
        "model.gguf",
        "tokenizer.json",
        "build-info.json",
        # ONNX-era file names, listed in docs/AGENTS.md as terminology to scrutinize.
        "model.ort",
        "libonnxruntime.so",
        # The history status banner's link in docs/history/AGENTS.md. It is relative to
        # docs/history/<category>/<name>.md, not to the AGENTS.md that shows it.
        "../../architecture.md",
        # A container platform.
        "linux/arm64",
        # Prose in code fences: the engine flow diagram in src/engine/AGENTS.md and the
        # status banner template in docs/history/AGENTS.md.
        "query/document",
        "llama.cpp/GGUF",
        # Hugging Face repositories of the current and the previous model (root AGENTS.md,
        # src/engine/AGENTS.md), and the floating revision URL segment .github/AGENTS.md
        # forbids.
        "unsloth/embeddinggemma-2-GGUF",
        "unsloth/embeddinggemma-2",
        "jinaai/jina-embeddings-v5-text-nano-retrieval",
        "resolve/main",
    }
)

FENCE_OPEN_RE = re.compile(r"^[ \t]*(?:>[ \t]?)*(`{3,}|~{3,})(.*)$")
FENCE_CLOSE_RE = re.compile(r"^[ \t]*(?:>[ \t]?)*(`{3,}|~{3,})[ \t]*$")
ATX_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
SETEXT_UNDERLINE_RE = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
THEMATIC_BREAK_RE = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
# A list item marker, matched after the line's indentation, and the spaces after it.
LIST_ITEM_RE = re.compile(r"([-*+]|\d{1,9}[.)])(?: +|$)")
# A first paragraph line that cannot be the text of a setext heading: a list item, block
# quote, table row, HTML block, or a line indented into a list item.
NOT_SETEXT_TEXT_RE = re.compile(r"^(?: {4}|\t| {0,3}(?:[-*+>|<]|\d+[.)])(?:[ \t]|$))")
CODE_SPAN_RE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.DOTALL)
REFERENCE_DEFINITION_RE = re.compile(r"^ {0,3}\[[^\]]+\]:[ \t]*(<[^>\n]*>|\S+)", re.MULTILINE)
LINK_TAIL_RE = re.compile(r"""[ \t\n]*(?:(?:"[^"]*"|'[^']*'|\([^)]*\))[ \t\n]*)?\)""")
SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
SECTION_RE = re.compile(r"§[ \t\n]*[\"“]([^\"”\n]+)[\"”]")
TOKEN_RE = re.compile(r"[^\s`'\"()\[\]{},;|=]+")
EXTENSION_RE = re.compile(r"\.[A-Za-z][A-Za-z0-9]*$")
# Tokens holding these are placeholders or shell expansions, such as
# `docs/history/<category>/` or `$PWD/gguf_bundle`.
PLACEHOLDER_CHARS = frozenset("<>$…")


@dataclass(frozen=True)
class Block:
    """A fenced code block, an indented code line, a heading, or a run of other non-blank lines."""

    line: int  # 1-based number of the first line
    text: str
    code: bool = False


@dataclass(frozen=True)
class Document:
    blocks: tuple[Block, ...]
    headings: tuple[str, ...]  # plain text, in document order


@dataclass(frozen=True, order=True)
class Problem:
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


class Repo:
    """The tracked files of a checkout, and parsed Markdown documents on demand."""

    def __init__(self, root: Path, files: Iterable[str]):
        self.root = root
        self.files = frozenset(files)
        dirs = {"."}
        for name in self.files:
            parent = posixpath.dirname(name)
            while parent and parent not in dirs:
                dirs.add(parent)
                parent = posixpath.dirname(parent)
        self.dirs = frozenset(dirs)
        self._documents: dict[str, Document] = {}

    @classmethod
    def from_git(cls, root: Path) -> "Repo":
        listing = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True
        ).stdout
        names = [os.fsdecode(name) for name in listing.split(b"\0") if name]
        # A tracked file deleted from the working tree is not there to link to.
        return cls(root, [name for name in names if (root / name).is_file()])

    def markdown_files(self) -> list[str]:
        return sorted(name for name in self.files if is_markdown(name))

    def document(self, path: str) -> Document:
        if path not in self._documents:
            text = (self.root / path).read_text(encoding="utf-8")
            self._documents[path] = parse_markdown(text)
        return self._documents[path]

    def anchors(self, path: str) -> set[str]:
        return heading_anchors(self.document(path).headings)

    def missing(self, path: str, want_dir: bool = False) -> str | None:
        """Why `path` (normalized, repo-relative) is not a tracked file or directory."""
        if path in self.dirs:
            return None
        if path in self.files:
            return f"{path} is a file, not a directory" if want_dir else None
        if (self.root / path).exists():
            return f"{path} exists but is not tracked by Git"
        return f"{path} does not exist"


def is_markdown(path: str) -> bool:
    return path.lower().endswith(".md")


def parse_markdown(text: str) -> Document:
    blocks: list[Block] = []
    headings: list[str] = []
    paragraph: list[str] = []
    paragraph_line = 0
    fence: tuple[str, int] | None = None
    fence_lines: list[str] = []
    fence_line = 0
    # Content columns of the open list items, innermost last. Indentation up to the
    # innermost one puts a line inside that item rather than making it code.
    items: list[int] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(Block(paragraph_line, "\n".join(paragraph)))
            paragraph.clear()

    for number, line in enumerate(text.splitlines(), start=1):
        if fence is not None:
            close = FENCE_CLOSE_RE.match(line)
            if close and close.group(1)[0] == fence[0] and len(close.group(1)) >= fence[1]:
                blocks.append(Block(fence_line, "\n".join(fence_lines), code=True))
                fence = None
            else:
                fence_lines.append(line)
            continue
        if not line.strip():
            flush_paragraph()
            continue
        expanded = line.expandtabs(4)
        indent = len(expanded) - len(expanded.lstrip(" "))
        item = LIST_ITEM_RE.match(expanded, indent)
        if item or not paragraph:
            # A line that does not continue a paragraph closes the list items it is not
            # indented into.
            while items and indent < items[-1]:
                items.pop()
        if indent - (items[-1] if items else 0) >= 4:
            # Four columns past the innermost item's content: a paragraph's continuation,
            # or else indented code. Neither is markup, not even a fence.
            if paragraph:
                paragraph.append(line)
            else:
                blocks.append(Block(number, line, code=True))
            continue
        opening = FENCE_OPEN_RE.match(line)
        if opening and not (opening.group(1)[0] == "`" and "`" in opening.group(2)):
            flush_paragraph()
            fence = (opening.group(1)[0], len(opening.group(1)))
            fence_lines = []
            fence_line = number + 1
            continue
        heading = ATX_HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            blocks.append(Block(number, line))
            headings.append(plain_text(heading.group(2) or ""))
            continue
        underline = SETEXT_UNDERLINE_RE.match(line)
        if underline or THEMATIC_BREAK_RE.match(line):
            # Under a paragraph, `---` or `===` turns it into a heading, as GitHub renders it.
            if underline and paragraph and not NOT_SETEXT_TEXT_RE.match(paragraph[0]):
                headings.append(plain_text(" ".join(part.strip() for part in paragraph)))
            flush_paragraph()
            continue
        if item:
            # The item's content starts after the 1 to 4 spaces that follow the marker. With
            # more, or none, it starts one column after the marker.
            gap = item.end() - item.end(1)
            items.append(item.end(1) + (gap if gap <= 4 and item.end() < len(expanded) else 1))
        if not paragraph:
            paragraph_line = number
        paragraph.append(line)
    flush_paragraph()
    if fence is not None:  # An unclosed fence runs to the end of the document.
        blocks.append(Block(fence_line, "\n".join(fence_lines), code=True))
    return Document(tuple(blocks), tuple(headings))


def plain_text(markdown: str) -> str:
    """The text a heading renders as: code spans literal, other inline markup removed."""
    parts = []
    position = 0
    for span in CODE_SPAN_RE.finditer(markdown):
        parts.append(_strip_inline_markup(markdown[position : span.start()]))
        parts.append(span.group(2))
        position = span.end()
    parts.append(_strip_inline_markup(markdown[position:]))
    return "".join(parts).strip()


def _strip_inline_markup(text: str) -> str:
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links and images: keep text
    text = re.sub(r"<[^>]+>", "", text)  # HTML tags
    text = re.sub(r"\*+|~~|(?<!\w)_+|_+(?!\w)", "", text)  # emphasis, not snake_case
    return re.sub(r"\\(.)", r"\1", text)  # backslash escapes


def github_slug(heading: str) -> str:
    """GitHub's heading anchor: lowercase, punctuation dropped, spaces to hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")


def heading_anchors(headings: Iterable[str]) -> set[str]:
    """Anchors for headings in document order.

    As on GitHub, a heading whose slug is taken gets the first free -1, -2, ... suffix, and
    a suffixed anchor is taken like any other: `Foo`, `Foo`, `Foo-1` are #foo, #foo-1 and
    #foo-1-1.
    """
    anchors: set[str] = set()
    for heading in headings:
        slug = anchor = github_slug(heading)
        count = 0
        while anchor in anchors:
            count += 1
            anchor = f"{slug}-{count}"
        anchors.add(anchor)
    return anchors


def mask_code_spans(text: str) -> str:
    """Blank out code spans, keeping offsets and newlines, so their contents never parse."""
    return CODE_SPAN_RE.sub(lambda span: re.sub(r"[^\n]", "x", span.group(0)), text)


def link_targets(text: str) -> list[tuple[int, str]]:
    """(offset, target) for inline links, images and reference definitions in `text`.

    `text` must already have its code spans masked.
    """
    targets = []
    for opening in re.finditer(r"(?<!\\)\]\(", text):
        target = _parse_destination(text, opening.end())
        if target is not None:
            targets.append((opening.start(), target))
    for definition in REFERENCE_DEFINITION_RE.finditer(text):
        target = definition.group(1)
        if target.startswith("<"):
            target = target[1:-1]
        targets.append((definition.start(), target))
    return sorted(targets)


def _parse_destination(text: str, position: int) -> str | None:
    while position < len(text) and text[position] in " \t\n":
        position += 1
    if text.startswith("<", position):
        end = text.find(">", position)
        if end == -1 or "\n" in text[position:end]:
            return None
        target = text[position + 1 : end]
        position = end + 1
    else:
        start, depth = position, 0
        while position < len(text):
            char = text[position]
            if char == "\\":
                position += 2
                continue
            if char.isspace() or (char == ")" and depth == 0):
                break
            depth += {"(": 1, ")": -1}.get(char, 0)
            position += 1
        target = text[start:position]
    return target if LINK_TAIL_RE.match(text, position) else None


def normalize(path: str) -> str | None:
    """Repo-relative normalized path, or None if it leaves the repository."""
    normalized = posixpath.normpath(path) if path else "."
    if normalized == ".." or normalized.startswith(("../", "/")):
        return None
    return normalized


def resolve_link_path(source: str, path: str) -> str | None:
    """Where a link to `path` in file `source` points; `/` is the repository root."""
    base = "" if path.startswith("/") else posixpath.dirname(source)
    return normalize(posixpath.join(base, path.lstrip("/")))


def check_link(repo: Repo, source: str, target: str) -> str | None:
    if SCHEME_RE.match(target) or target.startswith("//"):
        return None  # An external URL: never fetched.
    raw_path, _, fragment = target.partition("#")
    path = unquote(raw_path.split("?", 1)[0])
    # No path, as in `#anchor` or an empty `()`, is the linking file itself.
    resolved = resolve_link_path(source, path) if path else source
    if resolved is None:
        return f"link `{target}` points outside the repository"
    problem = repo.missing(resolved, want_dir=path.endswith("/"))
    if problem:
        return f"link `{target}`: {problem}"
    anchor = unquote(fragment)
    if anchor and is_markdown(resolved) and resolved in repo.files:
        if anchor not in repo.anchors(resolved):
            return f"link `{target}`: {resolved} has no heading with anchor #{anchor}"
    return None


def check_links(repo: Repo, path: str, document: Document) -> list[Problem]:
    problems = []
    for block in document.blocks:
        if block.code:
            continue
        text = mask_code_spans(block.text)
        for offset, target in link_targets(text):
            message = check_link(repo, path, target)
            if message:
                problems.append(Problem(path, block.line + text.count("\n", 0, offset), message))
    return problems


def path_tokens(text: str) -> list[str]:
    """Words of a code span or fence line, without trailing sentence punctuation."""
    return [token.rstrip(".,:;") for token in TOKEN_RE.findall(text)]


def is_path_like(token: str) -> bool:
    if "://" in token or "::" in token or token.startswith("-"):
        return False
    return "/" in token or EXTENSION_RE.search(token) is not None


def check_path_token(repo: Repo, source: str, token: str) -> tuple[str | None, str | None]:
    """Check a backticked token from AGENTS.md file `source`.

    Returns (problem, resolved path); both are None for a token that is not a repository
    path. A token starting with `/` resolves from the repository root; one starting with
    `./` or `../` from the directory of `source`; any other from the repository root or,
    failing that, from the directory of `source` (`bundle.rs` in src/engine/AGENTS.md is
    src/engine/bundle.rs).
    """
    if not token or not is_path_like(token) or token in NOT_REPO_PATHS:
        return None, None
    if PLACEHOLDER_CHARS.intersection(token):
        return None, None
    here = posixpath.dirname(source)
    if token.startswith("/"):
        candidates = [token.lstrip("/")]
    elif token.startswith(("./", "../")):
        candidates = [posixpath.join(here, token)]
    else:
        candidates = [token, posixpath.join(here, token)]
    resolved = [path for path in map(normalize, candidates) if path is not None]
    if not resolved:
        return f"path `{token}` points outside the repository", None
    if "*" in token or "?" in token:
        names = repo.files | repo.dirs
        if any(fnmatch.filter(names, pattern) for pattern in resolved):
            return None, None
        return f"path `{token}` matches no tracked file", None
    want_dir = token.endswith("/")
    for path in resolved:
        if repo.missing(path, want_dir) is None:
            return None, path
    return f"path `{token}`: {repo.missing(resolved[0], want_dir)}", None


def check_agents_references(repo: Repo, path: str, document: Document) -> list[Problem]:
    problems = []
    for block in document.blocks:
        if block.code:
            for offset, line in enumerate(block.text.split("\n")):
                for token in path_tokens(line):
                    message, _ = check_path_token(repo, path, token)
                    if message:
                        problems.append(Problem(path, block.line + offset, message))
            continue
        # Walk code spans, Markdown links and § "Section" references in order. A section
        # belongs to the closest Markdown file named before it in the same block, or to
        # this file if none is. A section is not checked if that Markdown path is broken,
        # external or a placeholder, or if it directly follows a code span that names no
        # Markdown file, as in the notation `file` § "Section".
        masked = mask_code_spans(block.text)
        events = [
            (span.start(), span.end(), "span", span.group(2))
            for span in CODE_SPAN_RE.finditer(block.text)
        ]
        events += [(offset, offset, "link", target) for offset, target in link_targets(masked)]
        events += [
            (match.start(), match.end(), "section", match.group(1))
            for match in SECTION_RE.finditer(masked)
        ]
        section_file: str | None = path
        other_span_end: int | None = None
        for start, end, kind, value in sorted(events):
            line = block.line + block.text.count("\n", 0, start)
            if kind == "span":
                names_markdown = False
                for token in path_tokens(value):
                    message, resolved = check_path_token(repo, path, token)
                    if message:
                        problems.append(Problem(path, line, message))
                    if is_markdown(token):
                        section_file = resolved if resolved in repo.files else None
                        names_markdown = True
                other_span_end = None if names_markdown else end
            elif kind == "link":
                target = unquote(value.partition("#")[0])
                if is_markdown(target):
                    resolved = None if SCHEME_RE.match(target) else resolve_link_path(path, target)
                    section_file = resolved if resolved in repo.files else None
            elif other_span_end is not None and not masked[other_span_end:start].strip():
                continue
            elif section_file is not None and value not in repo.document(section_file).headings:
                problems.append(Problem(path, line, f'{section_file} has no heading "{value}"'))
    return problems


def check_repository(repo: Repo) -> list[Problem]:
    problems = []
    for path in repo.markdown_files():
        document = repo.document(path)
        problems += check_links(repo, path, document)
        if posixpath.basename(path) == "AGENTS.md":
            problems += check_agents_references(repo, path, document)
    return sorted(problems)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check that relative Markdown links and AGENTS.md path references resolve."
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="repository root (default: this checkout)"
    )
    args = parser.parse_args(argv)
    repo = Repo.from_git(args.root.resolve())
    problems = check_repository(repo)
    for problem in problems:
        print(problem)
    checked = len(repo.markdown_files())
    if problems:
        print(f"{len(problems)} broken reference(s) in {checked} Markdown files.")
        return 1
    print(f"Checked {checked} Markdown files: all relative links and AGENTS.md paths resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
