# shellcheck shell=bash
# The CPU check for the x86_64-v3 static llama.cpp archives. Sourced, not run, by
# .github/scripts/fetch-static-llama.sh, which fails the CI job when the check fails, and by
# .githooks/pre-push, which fails too rather than run the linux/arm64 container under
# emulation. The archives execute x86-64-v3 instructions unconditionally (no runtime
# dispatch), so a CPU below that level faults with SIGILL.

# x86_64_v3_check [LDSO] [CPUINFO]
#
# Succeeds, printing nothing, if glibc and the kernel both report every x86-64-v3 feature.
# Otherwise prints the reason on one line and fails. glibc lists x86-64-v3 as "supported"
# only if CPUID reports the features and XGETBV shows the OS enabled AVX state;
# /proc/cpuinfo is checked as well, as in the builder's own gate. The arguments stand in for
# the dynamic loader and /proc/cpuinfo in tests; callers pass none.
x86_64_v3_check() {
  local ldso="${1:-/lib64/ld-linux-x86-64.so.2}" cpuinfo="${2:-/proc/cpuinfo}"
  local hwcaps flags missing=() f
  hwcaps="$("$ldso" --help 2>/dev/null || true)"
  if ! grep -Eq '^[[:space:]]*x86-64-v3 \(supported' <<<"$hwcaps"; then
    echo "$ldso does not report x86-64-v3 as supported"
    return 1
  fi
  flags=" $(grep -m1 '^flags' "$cpuinfo" 2>/dev/null | cut -d: -f2 || true) "
  # x86-64-v2 + v3 feature set. Linux reports SSE3 as `pni` and LZCNT as `abm`.
  for f in cx16 lahf_lm popcnt pni sse4_1 sse4_2 ssse3 avx avx2 bmi1 bmi2 f16c fma abm movbe xsave; do
    [[ "$flags" == *" $f "* ]] || missing+=("$f")
  done
  if [ "${#missing[@]}" -ne 0 ]; then
    echo "$cpuinfo lacks x86-64-v3 features: ${missing[*]}"
    return 1
  fi
}
