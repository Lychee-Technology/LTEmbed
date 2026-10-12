"""Tests for .githooks/pre-push and the x86-64-v3 check it shares with CI.

The hook runs in a scratch Git repository with stub `uname`, `cargo` and `docker` commands
first on PATH, so each test chooses the host it simulates and nothing is compiled or run in a
container. The repository's copy of `.github/scripts/x86-64-v3.sh` is a stub too, so the
branch the hook takes does not depend on the CPU running the tests; `X86_64V3CheckTests`
covers the real check against a fake dynamic loader and `/proc/cpuinfo`.
"""

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".githooks" / "pre-push"
X86_64_V3_CHECK = ROOT / ".github" / "scripts" / "x86-64-v3.sh"

AARCH64_TRIPLE = "aarch64-unknown-linux-gnu"
X86_64_TRIPLE = "x86_64-unknown-linux-gnu"
RELEASE_FILES = (
    "lib/libllama.a",
    "lib/libggml.a",
    "lib/libggml-cpu.a",
    "lib/libggml-base.a",
    "bindings.rs",
)
NATIVE_CLIPPY = "cargo clippy --all-targets -- -D warnings"
CONTAINER_CLIPPY = "rust:1.94.0 cargo clippy --all-targets -- -D warnings"

# A command line per stub call, appended to $STUB_LOG.
STUB = """#!/bin/sh
echo "{name} $*" >> "$STUB_LOG"
{body}
"""
STUBS = {
    "uname": 'echo "$STUB_UNAME"',
    "cargo": 'exit "${STUB_CARGO_STATUS:-0}"',
    "docker": 'exit "${STUB_DOCKER_STATUS:-0}"',
}
# Stands in for the real check: fails with $STUB_V3_UNSUPPORTED as the reason when it is set.
STUB_X86_64_V3_CHECK = """x86_64_v3_check() {
  echo "x86_64_v3_check" >> "$STUB_LOG"
  if [ -n "${STUB_V3_UNSUPPORTED:-}" ]; then
    echo "$STUB_V3_UNSUPPORTED"
    return 1
  fi
}
"""


def write_executable(path, text):
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def write_release(directory, triple, contract="4", omit=()):
    """An extracted static llama.cpp release: empty files and a build-info.json."""
    for name in RELEASE_FILES:
        if name not in omit:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    info = {"artifact_contract_version": contract, "target_triple": triple}
    (directory / "build-info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return directory


class PrePushHookTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        # Resolved, as the hook's `git rev-parse --show-toplevel` is.
        self.tmp = Path(temp_dir.name).resolve()

        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True, env=self.git_env())
        (self.repo / ".githooks").mkdir()
        shutil.copy2(HOOK, self.repo / ".githooks" / "pre-push")
        (self.repo / ".github" / "scripts").mkdir(parents=True)
        (self.repo / ".github" / "scripts" / "x86-64-v3.sh").write_text(
            STUB_X86_64_V3_CHECK, encoding="utf-8"
        )

        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        for name, body in STUBS.items():
            write_executable(self.bin / name, STUB.format(name=name, body=body))
        self.log = self.tmp / "calls.log"

        # The release the container branch mounts.
        self.container_release = self.repo / ".llama-artifacts" / "extracted"

    @staticmethod
    def git_env():
        """The environment without Git's per-repository variables, which a hook inherits."""
        return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}

    def run_hook(self, uname, static_llama_dir=None, **stub_env):
        env = self.git_env()
        env.pop("STATIC_LLAMA_DIR", None)
        env["PATH"] = f"{self.bin}{os.pathsep}{env.get('PATH', '')}"
        env["STUB_LOG"] = str(self.log)
        env["STUB_UNAME"] = uname
        if static_llama_dir is not None:
            env["STATIC_LLAMA_DIR"] = str(static_llama_dir)
        env.update(stub_env)
        self.log.write_text("", encoding="utf-8")
        result = subprocess.run(
            ["bash", ".githooks/pre-push", "origin", "git@example.com:repo.git"],
            cwd=self.repo,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
        calls = self.log.read_text(encoding="utf-8").splitlines()
        return result, calls

    def assert_native(self, result, calls):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(NATIVE_CLIPPY, calls)
        self.assertFalse([c for c in calls if c.startswith("docker ")], calls)

    def assert_container(self, result, calls):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse([c for c in calls if c.startswith("cargo ")], calls)
        self.assertIn("docker info", calls)
        [run] = [c for c in calls if c.startswith("docker run ")]
        self.assertIn("--platform linux/arm64", run)
        self.assertIn(f"-v {self.container_release}:/opt/static-llama", run)
        self.assertTrue(run.endswith(CONTAINER_CLIPPY), run)

    # x86_64 Linux.

    def test_x86_64_v3_runs_clippy_natively_without_docker_or_container_release(self):
        release = write_release(self.tmp / "extracted-x86_64-v3", X86_64_TRIPLE)
        result, calls = self.run_hook("Linux x86_64", release, STUB_DOCKER_STATUS="1")
        self.assert_native(result, calls)
        self.assertIn("x86_64_v3_check", calls)
        self.assertFalse(self.container_release.exists())

    def test_x86_64_below_v3_uses_container(self):
        release = write_release(self.tmp / "extracted-x86_64-v3", X86_64_TRIPLE)
        write_release(self.container_release, AARCH64_TRIPLE)
        result, calls = self.run_hook(
            "Linux x86_64", release, STUB_V3_UNSUPPORTED="no x86-64-v3 here"
        )
        self.assert_container(result, calls)
        self.assertIn("not running clippy natively: no x86-64-v3 here", result.stdout)

    def test_x86_64_without_static_llama_dir_uses_container(self):
        write_release(self.container_release, AARCH64_TRIPLE)
        for static_llama_dir in (None, ""):
            with self.subTest(STATIC_LLAMA_DIR=static_llama_dir):
                result, calls = self.run_hook("Linux x86_64", static_llama_dir)
                self.assert_container(result, calls)
                self.assertNotIn("x86_64_v3_check", calls)

    # aarch64 Linux and macOS.

    def test_aarch64_runs_clippy_natively_without_the_x86_64_check(self):
        release = write_release(self.tmp / "extracted", AARCH64_TRIPLE)
        result, calls = self.run_hook(
            "Linux aarch64", release, STUB_DOCKER_STATUS="1", STUB_V3_UNSUPPORTED="unused"
        )
        self.assert_native(result, calls)
        self.assertNotIn("x86_64_v3_check", calls)

    def test_aarch64_without_static_llama_dir_uses_container(self):
        write_release(self.container_release, AARCH64_TRIPLE)
        result, calls = self.run_hook("Linux aarch64")
        self.assert_container(result, calls)

    def test_macos_uses_container_even_with_static_llama_dir(self):
        write_release(self.container_release, AARCH64_TRIPLE)
        release = write_release(self.tmp / "extracted-x86_64-v3", X86_64_TRIPLE)
        for uname in ("Darwin arm64", "Darwin x86_64"):
            with self.subTest(uname=uname):
                result, calls = self.run_hook(uname, release)
                self.assert_container(result, calls)
                self.assertNotIn("x86_64_v3_check", calls)

    # The checks before clippy runs.

    def test_native_branches_reject_the_other_architectures_variant(self):
        cases = (
            ("Linux x86_64", AARCH64_TRIPLE, X86_64_TRIPLE, "x86_64-v3"),
            ("Linux aarch64", X86_64_TRIPLE, AARCH64_TRIPLE, "aarch64-graviton2"),
        )
        for uname, actual, expected, variant in cases:
            with self.subTest(uname=uname):
                release = write_release(self.tmp / uname.replace(" ", "-"), actual)
                result, calls = self.run_hook(uname, release)
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"unexpected target_triple '{actual}'", result.stderr)
                self.assertIn(f"(expected {expected})", result.stderr)
                self.assertIn(f"STATIC_LLAMA_DIR must hold the {variant} variant", result.stderr)
                self.assertFalse([c for c in calls if c.startswith(("cargo ", "docker "))])

    def test_native_branches_reject_another_contract_or_an_incomplete_release(self):
        bad_contract = "unexpected artifact_contract_version '{}' in {}/build-info.json"
        defects = {
            "contract 3": (dict(contract="3"), bad_contract.format("3", "{}")),
            "no contract": (dict(contract=""), bad_contract.format("<missing>", "{}")),
            "no libggml.a": (dict(omit=("lib/libggml.a",)), "{}/lib/libggml.a not found"),
        }
        for uname, triple in (("Linux x86_64", X86_64_TRIPLE), ("Linux aarch64", AARCH64_TRIPLE)):
            for name, (release_args, message) in defects.items():
                with self.subTest(uname=uname, defect=name):
                    release = write_release(
                        self.tmp / f"{uname}-{name}".replace(" ", "-"), triple, **release_args
                    )
                    result, calls = self.run_hook(uname, release)
                    self.assertEqual(result.returncode, 1)
                    self.assertIn(message.format(release), result.stderr)
                    self.assertFalse([c for c in calls if c.startswith(("cargo ", "docker "))])

    def test_container_branch_checks_its_release_and_docker(self):
        with self.subTest("missing release"):
            result, calls = self.run_hook("Darwin arm64")
            self.assertEqual(result.returncode, 1)
            self.assertIn("extracted/lib/libllama.a not found", result.stderr)
            self.assertFalse([c for c in calls if c.startswith("docker ")])

        with self.subTest("x86_64 variant in .llama-artifacts/extracted/"):
            write_release(self.container_release, X86_64_TRIPLE)
            result, calls = self.run_hook("Darwin arm64")
            self.assertEqual(result.returncode, 1)
            self.assertIn(
                ".llama-artifacts/extracted/ must hold the aarch64-graviton2 variant",
                result.stderr,
            )
            self.assertFalse([c for c in calls if c.startswith("docker ")])

        with self.subTest("docker unreachable"):
            write_release(self.container_release, AARCH64_TRIPLE)
            result, calls = self.run_hook("Darwin arm64", STUB_DOCKER_STATUS="1")
            self.assertEqual(result.returncode, 1)
            self.assertIn("docker daemon not reachable", result.stderr)
            self.assertEqual([c for c in calls if c.startswith("docker ")], ["docker info"])

    def test_native_clippy_failure_fails_the_push(self):
        release = write_release(self.tmp / "extracted-x86_64-v3", X86_64_TRIPLE)
        result, calls = self.run_hook("Linux x86_64", release, STUB_CARGO_STATUS="101")
        self.assertEqual(result.returncode, 101)
        self.assertIn(NATIVE_CLIPPY, calls)


# glibc's `ld.so --help` hwcaps listing on a CPU that supports up to x86-64-v3.
HWCAPS_V3 = """\
Subdirectories of glibc-hwcaps directories, in priority order:
  x86-64-v4
  x86-64-v3 (supported, searched)
  x86-64-v2 (supported, searched)
"""
HWCAPS_V2 = """\
Subdirectories of glibc-hwcaps directories, in priority order:
  x86-64-v4
  x86-64-v3
  x86-64-v2 (supported, searched)
"""
V3_FLAGS = (
    "cx16 lahf_lm popcnt pni sse4_1 sse4_2 ssse3 avx avx2 bmi1 bmi2 f16c fma abm movbe xsave"
).split()


class X86_64V3CheckTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.tmp = Path(temp_dir.name)

    def check(self, hwcaps, flags):
        """Run x86_64_v3_check under the callers' shell options; return (status, output)."""
        ldso = self.tmp / "ld.so"
        (self.tmp / "hwcaps").write_text(hwcaps, encoding="utf-8")
        write_executable(ldso, f'#!/bin/sh\ncat "{self.tmp / "hwcaps"}"\n')
        cpuinfo = self.tmp / "cpuinfo"
        lines = ["processor\t: 0", "model name\t: Test CPU"]
        if flags is not None:
            lines.append("flags\t\t: " + " ".join(flags))
        cpuinfo.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = subprocess.run(
            [
                "bash",
                "-c",
                'set -euo pipefail; source "$0"; x86_64_v3_check "$1" "$2"',
                str(X86_64_V3_CHECK),
                str(ldso),
                str(cpuinfo),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stderr, "")
        return result.returncode, result.stdout.strip()

    def test_passes_silently_when_glibc_and_the_kernel_report_v3(self):
        flags = ["fpu", "sse2", *V3_FLAGS, "avx512f"]
        self.assertEqual(self.check(HWCAPS_V3, flags), (0, ""))

    def test_fails_when_glibc_does_not_report_v3_as_supported(self):
        status, output = self.check(HWCAPS_V2, V3_FLAGS)
        self.assertEqual(status, 1)
        self.assertEqual(output, f"{self.tmp / 'ld.so'} does not report x86-64-v3 as supported")

    def test_fails_when_the_dynamic_loader_is_missing(self):
        result = subprocess.run(
            [
                "bash",
                "-c",
                'set -euo pipefail; source "$0"; x86_64_v3_check "$1" /proc/cpuinfo',
                str(X86_64_V3_CHECK),
                str(self.tmp / "no-such-ld.so"),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("does not report x86-64-v3 as supported", result.stdout)

    def test_fails_naming_the_flags_the_kernel_does_not_report(self):
        # `avx2` and `avx512f` must not count as `avx`.
        flags = [f for f in V3_FLAGS if f not in ("avx", "movbe")] + ["avx512f"]
        status, output = self.check(HWCAPS_V3, flags)
        self.assertEqual(status, 1)
        self.assertEqual(output, f"{self.tmp / 'cpuinfo'} lacks x86-64-v3 features: avx movbe")

    def test_fails_without_a_flags_line(self):
        status, output = self.check(HWCAPS_V3, None)
        self.assertEqual(status, 1)
        missing = " ".join(V3_FLAGS)
        self.assertEqual(output, f"{self.tmp / 'cpuinfo'} lacks x86-64-v3 features: {missing}")


if __name__ == "__main__":
    unittest.main()
