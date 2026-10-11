import importlib.util
import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "check_doc_references.py"


def load_module():
    spec = importlib.util.spec_from_file_location("check_doc_references", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CheckDocReferencesTests(unittest.TestCase):
    def setUp(self):
        self.check = load_module()
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)

    def write(self, files):
        for name, text in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    def problems(self, files):
        """Problems found when exactly `files` are tracked."""
        self.write(files)
        repo = self.check.Repo(self.root, files)
        return [str(problem) for problem in self.check.check_repository(repo)]

    # Links

    def test_relative_links_resolve_from_the_linking_file(self):
        files = {
            "README.md": "",
            "docs/a.md": "[b](./b.md) [b](b.md) [up](../README.md) [src](../src/) [img](../x.png)\n",
            "docs/b.md": "",
            "src/lib.rs": "",
            "x.png": "",
        }
        self.assertEqual(self.problems(files), [])

    def test_link_to_missing_file_fails(self):
        files = {"docs/a.md": "Intro.\n\nSee [the design](./design.md).\n"}
        self.assertEqual(
            self.problems(files),
            ["docs/a.md:3: link `./design.md`: docs/design.md does not exist"],
        )

    def test_root_relative_link_resolves_from_the_repository_root(self):
        files = {
            "docs/history/old.md": "[ok](/src/lib.rs)\n\n- trace: [`/tmp/a.trace`](/tmp/a.trace)\n",
            "src/lib.rs": "",
        }
        self.assertEqual(
            self.problems(files),
            ["docs/history/old.md:3: link `/tmp/a.trace`: tmp/a.trace does not exist"],
        )

    def test_link_outside_the_repository_fails(self):
        files = {"README.md": "[x](../other/README.md)\n"}
        self.assertEqual(
            self.problems(files),
            ["README.md:1: link `../other/README.md` points outside the repository"],
        )

    def test_directory_link_must_be_a_directory(self):
        files = {"README.md": "[docs](docs/) [not a dir](docs/a.md/)\n", "docs/a.md": ""}
        self.assertEqual(
            self.problems(files),
            ["README.md:1: link `docs/a.md/`: docs/a.md is a file, not a directory"],
        )

    def test_anchor_must_match_a_github_heading_slug(self):
        target = "\n".join(
            [
                "# Testing",
                "## Golden fixture: `tests/fixtures/test_fixtures.json`",
                "## 4. Step 1 — Prepare the ONNX Model",
                "## Why Q5_K_M",
                "## Static llama.cpp artifacts ##",
                "## Repeat",
                "## Repeat",
                "",
                "A paragraph underlined",
                "---",
                "",
                "* a list item, not a heading",
                "---",
                "",
                "```bash",
                "# a comment, not a heading",
                "```",
            ]
        )
        links = [
            "golden-fixture-testsfixturestest_fixturesjson",
            "4-step-1--prepare-the-onnx-model",
            "why-q5_k_m",
            "static-llamacpp-artifacts",
            "repeat",
            "repeat-1",
            "a-paragraph-underlined",
            "repeat-2",
            "a-list-item-not-a-heading",
            "a-comment-not-a-heading",
            "Testing",
        ]
        source = "\n".join(f"[x](./testing.md#{anchor})" for anchor in links)
        source += "\n[same file](#local) [other](#elsewhere)\n\n## Local\n"
        files = {"docs/a.md": source, "docs/testing.md": target}
        self.assertEqual(
            self.problems(files),
            [
                f"docs/a.md:{line}: link `./testing.md#{anchor}`: "
                f"docs/testing.md has no heading with anchor #{anchor}"
                for line, anchor in [
                    (8, "repeat-2"),
                    (9, "a-list-item-not-a-heading"),
                    (10, "a-comment-not-a-heading"),
                    (11, "Testing"),
                ]
            ]
            + ["docs/a.md:12: link `#elsewhere`: docs/a.md has no heading with anchor #elsewhere"],
        )

    def test_repeated_heading_takes_the_first_free_suffix(self):
        # Headings and anchors from github-slugger's test/fixtures.json, which records what
        # github.com renders: a suffixed anchor is taken like any other.
        headings_and_anchors = [
            ("echo", "echo"),
            ("echo", "echo-1"),
            ("echo 1", "echo-1-1"),
            ("echo-1", "echo-1-2"),
            ("echo", "echo-2"),
            ("foxtrot-1", "foxtrot-1"),
            ("foxtrot", "foxtrot"),
            ("foxtrot", "foxtrot-2"),
        ]
        target = "\n\n".join(f"## {heading}" for heading, _ in headings_and_anchors)
        anchors = [anchor for _, anchor in headings_and_anchors] + ["echo-3", "foxtrot-1-1"]
        files = {
            "a.md": "\n".join(f"[x](b.md#{anchor})" for anchor in anchors),
            "b.md": target,
        }
        self.assertEqual(
            self.problems(files),
            [
                f"a.md:{line}: link `b.md#{anchor}`: b.md has no heading with anchor #{anchor}"
                for line, anchor in [(9, "echo-3"), (10, "foxtrot-1-1")]
            ],
        )

    def test_link_with_no_path_is_the_linking_file(self):
        files = {
            "docs/a.md": (
                "[self]() [self](<>) [top](#) [ref][self] [heading](#heading) [gone](#gone)\n"
                "\n"
                "[self]: <>\n"
                "\n"
                "## Heading\n"
            )
        }
        self.assertEqual(
            self.problems(files),
            ["docs/a.md:1: link `#gone`: docs/a.md has no heading with anchor #gone"],
        )

    def test_fragment_on_a_non_markdown_target_is_not_checked(self):
        files = {"README.md": "[line](src/lib.rs#L10)\n", "src/lib.rs": ""}
        self.assertEqual(self.problems(files), [])

    def test_links_inside_code_are_not_checked(self):
        files = {
            "README.md": "\n".join(
                [
                    "```markdown",
                    "[x](missing.md)",
                    "```",
                    "",
                    "~~~",
                    "[x](missing.md)",
                    "~~~",
                    "",
                    "Write links as `[x](missing.md)`.",
                ]
            )
        }
        self.assertEqual(self.problems(files), [])

    def test_reference_definitions_are_checked(self):
        files = {
            "README.md": "[a][ok] and [b][bad]\n\n[ok]: ./a.md\n[bad]: <./missing.md>\n",
            "a.md": "",
        }
        self.assertEqual(
            self.problems(files),
            ["README.md:4: link `./missing.md`: missing.md does not exist"],
        )

    def test_external_urls_are_skipped_and_never_fetched(self):
        files = {
            "README.md": (
                "[a](https://example.invalid/missing.md) [b](http://example.invalid)\n"
                "[c](mailto:nobody@example.invalid) [d](//example.invalid/x)\n"
                "<https://example.invalid/autolink>\n"
            )
        }
        network = AssertionError("the check must not use the network")
        with (
            mock.patch("socket.socket", side_effect=network),
            mock.patch("socket.getaddrinfo", side_effect=network),
            mock.patch("socket.create_connection", side_effect=network),
        ):
            self.assertEqual(self.problems(files), [])

    # AGENTS.md paths

    def test_agents_path_that_does_not_exist_fails(self):
        files = {
            "docs/AGENTS.md": "Use `docs/README.md` and `docs/design.md`.\n",
            "docs/README.md": "",
        }
        self.assertEqual(
            self.problems(files),
            ["docs/AGENTS.md:1: path `docs/design.md`: docs/design.md does not exist"],
        )

    def test_agents_paths_resolve_from_the_root_or_the_agents_directory(self):
        files = {
            "AGENTS.md": "",
            "Cargo.toml": "",
            "src/engine/bundle.rs": "",
            "src/engine/llama/mod.rs": "",
            "tests/test_a.py": "",
            "src/engine/AGENTS.md": (
                "Also follow `/AGENTS.md`. See `Cargo.toml`, `bundle.rs`, `./bundle.rs`,\n"
                "`../../Cargo.toml`, `src/engine/llama/` and `tests/test_*.py`.\n"
                "\n"
                "Broken: `src/engine/bundle.rs/`, `tests/test_*.sh`, `../../../x.md`.\n"
            ),
        }
        self.assertEqual(
            self.problems(files),
            [
                "src/engine/AGENTS.md:4: path `../../../x.md` points outside the repository",
                "src/engine/AGENTS.md:4: path `src/engine/bundle.rs/`: "
                "src/engine/bundle.rs is a file, not a directory",
                "src/engine/AGENTS.md:4: path `tests/test_*.sh` matches no tracked file",
            ],
        )

    def test_agents_code_fences_are_checked_and_non_paths_are_skipped(self):
        files = {
            "scripts/generate_fixtures.py": "",
            "tests/AGENTS.md": "\n".join(
                [
                    "```text",
                    "scripts/generate_fixtures.py",
                    "scripts/gone.py",
                    "gguf_bundle/",
                    "├── model.gguf",
                    "```",
                    "",
                    "Not paths: `EmbeddingInput::query(...)`, `LTEMBED_TEST_BUNDLE_DIR=1`,",
                    "`--all-targets`, `1.94.0`, `docs/history/<category>/`, `$PWD/gguf_bundle`,",
                    "`https://example.invalid/a/b.md`, `tokenizer.json`, `linux/arm64`.",
                ]
            ),
        }
        self.assertEqual(
            self.problems(files),
            ["tests/AGENTS.md:3: path `scripts/gone.py`: scripts/gone.py does not exist"],
        )

    def test_agents_section_references_must_match_headings(self):
        files = {
            "docs/development.md": "# Development\n\n## Checks before pushing\n\n## Native builds\n",
            "AGENTS.md": "\n".join(
                [
                    '`docs/development.md` § "Checks before pushing" and § "Checks Before Pushing".',
                    "",
                    '`docs/development.md` is the authority: § "Native builds", § "Container builds".',
                    "",
                    'This file: § "Scope". A broken path: `docs/gone.md` § "Anything".',
                    "",
                    'A link: [development](docs/development.md) § "Native builds".',
                    "",
                    'The notation `file` § "Section", after `docs/development.md`, names no file.',
                    "",
                    "## Scope",
                ]
            ),
        }
        self.assertEqual(
            self.problems(files),
            [
                'AGENTS.md:1: docs/development.md has no heading "Checks Before Pushing"',
                'AGENTS.md:3: docs/development.md has no heading "Container builds"',
                "AGENTS.md:5: path `docs/gone.md`: docs/gone.md does not exist",
            ],
        )

    def test_history_documents_get_only_the_link_check(self):
        files = {
            "docs/architecture.md": "",
            "docs/history/ort/old.md": (
                "Used `src/models/bert.rs` and `model.ort`.\n"
                "\n"
                "[Current](../../architecture.md) [Gone](../../design.md)\n"
            ),
        }
        self.assertEqual(
            self.problems(files),
            ["docs/history/ort/old.md:3: link `../../design.md`: docs/design.md does not exist"],
        )

    def test_readding_docs_design_md_to_docs_agents_md_fails(self):
        repo = self.check.Repo.from_git(ROOT)
        text = (ROOT / "docs" / "AGENTS.md").read_text(encoding="utf-8")
        document = self.check.parse_markdown(text + "\nInspect `docs/design.md` first.\n")
        messages = [
            problem.message
            for problem in self.check.check_agents_references(repo, "docs/AGENTS.md", document)
        ]
        self.assertIn("path `docs/design.md`: docs/design.md does not exist", messages)

    # Command line

    def test_main_lists_problems_and_exits_nonzero(self):
        self.write(
            {
                "README.md": "[new](docs/new.md) [gone](docs/gone.md)\n",
                "AGENTS.md": "See `README.md`.\n",
                "docs/new.md": "",
            }
        )
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "README.md", "AGENTS.md"], check=True)

        output = io.StringIO()
        with redirect_stdout(output):
            status = self.check.main(["--root", str(self.root)])
        self.assertEqual(status, 1)
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "README.md:1: link `docs/gone.md`: docs/gone.md does not exist",
                "README.md:1: link `docs/new.md`: docs/new.md exists but is not tracked by Git",
                "2 broken reference(s) in 2 Markdown files.",
            ],
        )

        (self.root / "README.md").write_text("[new](docs/new.md)\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.root), "add", "docs/new.md"], check=True)
        output = io.StringIO()
        with redirect_stdout(output):
            status = self.check.main(["--root", str(self.root)])
        self.assertEqual(status, 0, output.getvalue())


if __name__ == "__main__":
    unittest.main()
