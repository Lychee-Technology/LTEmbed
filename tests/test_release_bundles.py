"""Tests for .github/workflows/release-bundles.yml: its version check and release notes.

`.github/scripts/release-version.sh` runs against scratch Cargo.toml files, with RELEASE_TAG
set as each trigger sets it. The workflow is read as text, as the other workflow tests read
theirs. Two steps' own `run` scripts are also executed, in a scratch directory, under the
shell GitHub runs a step with when the step names none: the version step with a copy of the
script, and the step that creates the release with a stub `gh` first on PATH.
"""

import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release-bundles.yml"
RELEASE_VERSION = ROOT / ".github" / "scripts" / "release-version.sh"
VERSION_STEP = "Check the release tag against Cargo.toml"
CREATE_STEP = "Ensure release exists for tag"
NOTES_DIR = ROOT / "docs" / "release-notes"
REPOSITORY = "Lychee-Technology/LTEmbed"
# GitHub's command for a `run` step without `shell:` on Linux, with the script as {0}.
DEFAULT_STEP_SHELL = ["bash", "-e"]


def cargo_toml(version):
    return f'[package]\nname = "ltembed"\nversion = "{version}"\nedition = "2021"\n'


def workflow_steps():
    """The package job's steps in order, as {name: text}; the text is the step's YAML."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    _, _, steps = workflow.partition("\n    steps:\n")
    chunks = re.split(r"^      - name: ", steps, flags=re.MULTILINE)
    return {chunk.splitlines()[0]: chunk for chunk in chunks[1:]}


def run_script(step):
    """A step's `run: |` block, dedented as the runner writes it to a script file."""
    lines = step.splitlines()
    start = lines.index("        run: |") + 1
    body = []
    for line in lines[start:]:
        if line.strip() and not line.startswith(" " * 10):
            break
        body.append(line)
    return textwrap.dedent("\n".join(body)).strip() + "\n"


def render(script, **expressions):
    """Substitute a script's `${{ }}` expressions, as the runner does before running it."""
    for name, value in expressions.items():
        script = script.replace("${{ " + name + " }}", value)
    if "${{" in script:
        raise AssertionError(f"unrendered expression in:\n{script}")
    return script


def environment(release_tag):
    """The test's environment with RELEASE_TAG set, or removed when it is None."""
    env = dict(os.environ)
    env.pop("RELEASE_TAG", None)
    if release_tag is not None:
        env["RELEASE_TAG"] = release_tag
    return env


class ReleaseVersionTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.tmp = Path(temp_dir.name)

    def run_check(self, version, release_tag):
        manifest = self.tmp / "Cargo.toml"
        manifest.write_text(cargo_toml(version), encoding="utf-8")
        return subprocess.run(
            [str(RELEASE_VERSION), str(manifest)],
            env=environment(release_tag), capture_output=True, text=True,
        )

    def assert_passes(self, version, release_tag):
        result = self.run_check(version, release_tag)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, f"{version}\n")
        self.assertEqual(result.stderr, "")

    def assert_fails(self, version, release_tag):
        result = self.run_check(version, release_tag)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "", "no version for the caller to name a tarball with")
        self.assertIn(
            f"::error::release tag {release_tag} does not match Cargo.toml version {version};",
            result.stderr,
        )

    def test_workflow_dispatch_has_no_tag_to_check(self):
        self.assert_passes("0.2.0", None)
        self.assert_passes("0.2.0", "")

    def test_accepts_the_tag_of_the_version(self):
        self.assert_passes("0.2.0", "v0.2.0")
        self.assert_passes("0.3.0-rc.1", "v0.3.0-rc.1")

    def test_fails_on_the_tag_of_another_version(self):
        self.assert_fails("0.2.0", "v0.1.0")
        self.assert_fails("0.1.0", "v0.2.0")

    def test_compares_the_whole_tag(self):
        self.assert_fails("0.2.0", "0.2.0")
        self.assert_fails("0.2.0", "v0.2")
        self.assert_fails("0.2.0", "v0.2.0-rc.1")
        self.assert_fails("0.2.0-rc.1", "v0.2.0")

    def test_fails_without_a_package_version(self):
        manifest = self.tmp / "Cargo.toml"
        manifest.write_text('[workspace]\nmembers = ["ltembed"]\n', encoding="utf-8")
        result = subprocess.run(
            [str(RELEASE_VERSION), str(manifest)],
            env=environment(None), capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_reads_the_repository_cargo_toml_by_default(self):
        with open(ROOT / "Cargo.toml", "rb") as f:
            version = tomllib.load(f)["package"]["version"]
        result = subprocess.run(
            [str(RELEASE_VERSION)], cwd=ROOT,
            env=environment(f"v{version}"), capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, f"{version}\n")


class WorkflowTests(unittest.TestCase):
    def test_release_tag_is_the_tag_of_a_push_or_a_published_release(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(
            "RELEASE_TAG: ${{ github.event_name == 'push' && github.ref_name"
            " || github.event_name == 'release' && github.event.release.tag_name || '' }}",
            workflow,
        )

    def test_checks_the_version_before_downloading_or_publishing(self):
        names = list(workflow_steps())
        self.assertEqual(names[:2], ["Checkout", VERSION_STEP])
        for later in ("Assemble GGUF bundle", "Build LTEmbed release tarball",
                      "Upload workflow artifact", "Ensure release exists for tag",
                      "Upload release asset"):
            self.assertGreater(names.index(later), 1, later)

    def test_version_step_runs_on_every_trigger_and_gates_the_job(self):
        step = workflow_steps()[VERSION_STEP]
        self.assertIn("        id: version\n", step)
        self.assertNotIn("        if:", step)
        self.assertNotIn("continue-on-error", step)
        self.assertNotIn("        shell:", step)
        self.assertIn(".github/scripts/release-version.sh", run_script(step))

    def test_tarball_is_named_with_the_checked_version(self):
        step = workflow_steps()["Build LTEmbed release tarball"]
        self.assertIn("          VERSION: ${{ steps.version.outputs.version }}\n", step)
        self.assertNotIn("Cargo.toml", step)
        self.assertNotIn("::warning::", step)
        # The name docs/releases.md documents: version, model and quant, and no platform.
        self.assertIn('ROOT_NAME="ltembed-${VERSION}-${MODEL_SLUG}-${QUANT}"\n', step)


class VersionStepTests(unittest.TestCase):
    """The version step's `run` script, as the runner executes it."""

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.repo = Path(temp_dir.name)
        (self.repo / ".github" / "scripts").mkdir(parents=True)
        shutil.copy2(RELEASE_VERSION, self.repo / ".github" / "scripts" / "release-version.sh")
        (self.repo / "Cargo.toml").write_text(cargo_toml("0.2.0"), encoding="utf-8")
        self.script = self.repo / "step.sh"
        self.script.write_text(run_script(workflow_steps()[VERSION_STEP]), encoding="utf-8")
        self.output = self.repo / "github-output"
        self.output.touch()

    def run_step(self, release_tag):
        env = environment(release_tag)
        env["GITHUB_OUTPUT"] = str(self.output)
        return subprocess.run(
            [*DEFAULT_STEP_SHELL, str(self.script)],
            cwd=self.repo, env=env, capture_output=True, text=True,
        )

    def test_a_mismatched_tag_fails_the_step(self):
        result = self.run_step("v0.1.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("::error::release tag v0.1.0 does not match Cargo.toml version 0.2.0;",
                      result.stderr)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "")

    def test_the_step_outputs_the_version(self):
        for release_tag in ("v0.2.0", "", None):
            with self.subTest(release_tag=release_tag):
                self.output.write_text("", encoding="utf-8")
                result = self.run_step(release_tag)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.output.read_text(encoding="utf-8"), "version=0.2.0\n")



# Logs each call to $STUB_LOG; `release view` exits $STUB_VIEW_STATUS (default 1: no release).
STUB_GH = """#!/bin/sh
echo "gh $*" >> "$STUB_LOG"
case "$1 $2" in
  "release view") exit "${STUB_VIEW_STATUS:-1}" ;;
  "release create") exit 0 ;;
esac
exit 2
"""


class ReleaseCreationStepTests(unittest.TestCase):
    """The step that creates the release on a tag push, as the runner executes it."""

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.tmp = Path(temp_dir.name)
        self.repo = self.tmp / "repo"
        self.notes_dir = self.repo / "docs" / "release-notes"
        self.notes_dir.mkdir(parents=True)
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        gh = bin_dir / "gh"
        gh.write_text(STUB_GH, encoding="utf-8")
        gh.chmod(0o755)
        self.path = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
        self.log = self.tmp / "calls.log"
        self.script = self.tmp / "step.sh"
        self.script.write_text(
            render(run_script(workflow_steps()[CREATE_STEP]), **{"github.repository": REPOSITORY}),
            encoding="utf-8",
        )

    def run_step(self, release_tag="v0.2.0", view_status=1):
        env = environment(release_tag)
        env.update(PATH=self.path, STUB_LOG=str(self.log), STUB_VIEW_STATUS=str(view_status))
        result = subprocess.run(
            [*DEFAULT_STEP_SHELL, str(self.script)],
            cwd=self.repo, env=env, capture_output=True, text=True,
        )
        calls = self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []
        return result, calls

    def test_creates_the_release_with_the_notes_of_its_tag(self):
        (self.notes_dir / "v0.2.0.md").write_text("# LTEmbed v0.2.0\n", encoding="utf-8")
        result, calls = self.run_step()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [
            f"gh release view v0.2.0 --repo {REPOSITORY}",
            f"gh release create v0.2.0 --repo {REPOSITORY} --verify-tag --title v0.2.0"
            " --notes-file docs/release-notes/v0.2.0.md",
        ])

    def test_fails_without_the_notes_of_its_tag(self):
        (self.notes_dir / "v0.1.0.md").write_text("# LTEmbed v0.1.0\n", encoding="utf-8")
        result, calls = self.run_step()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("::error::docs/release-notes/v0.2.0.md is missing;", result.stdout)
        self.assertEqual(calls, [f"gh release view v0.2.0 --repo {REPOSITORY}"])

    def test_keeps_an_existing_release_and_its_notes(self):
        result, calls = self.run_step(view_status=0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [f"gh release view v0.2.0 --repo {REPOSITORY}"])

    def test_only_a_tag_push_creates_a_release(self):
        self.assertIn("        if: github.event_name == 'push'\n", workflow_steps()[CREATE_STEP])
        self.assertNotIn("--generate-notes", WORKFLOW.read_text(encoding="utf-8"))



class ReleaseNotesTests(unittest.TestCase):
    """docs/release-notes/v<version>.md, which a tag push publishes as the release's notes."""

    def test_the_cargo_toml_version_has_its_release_notes(self):
        with open(ROOT / "Cargo.toml", "rb") as f:
            version = tomllib.load(f)["package"]["version"]
        notes = NOTES_DIR / f"v{version}.md"
        self.assertTrue(notes.is_file(),
                        f"{notes.relative_to(ROOT)} is missing: write it with the version bump")
        self.assertEqual(notes.read_text(encoding="utf-8").splitlines()[0],
                         f"# LTEmbed v{version}")

    def test_release_notes_link_with_absolute_urls(self):
        # The release page renders the notes, and a relative link does not resolve there.
        for notes in sorted(NOTES_DIR.glob("*.md")):
            text = notes.read_text(encoding="utf-8")
            targets = (re.findall(r"\]\(([^)\s]+)", text)
                       + re.findall(r"^ {0,3}\[[^\]]+\]:\s*(\S+)", text, flags=re.MULTILINE))
            self.assertTrue(targets, notes.name)
            for target in targets:
                with self.subTest(notes=notes.name, target=target):
                    self.assertRegex(target, r"^https://")


if __name__ == "__main__":
    unittest.main()
