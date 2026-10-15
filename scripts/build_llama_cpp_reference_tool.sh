#!/usr/bin/env bash
# Build upstream llama.cpp's embedding tool (examples/embedding/embedding.cpp) for
# scripts/generate_llama_cpp_reference.py, linked against a static-llama-cpp-rs-builder
# release's own archives.
#
# The tool then runs the release's kernels (libllama.a, libggml*.a) but none of LTEmbed's
# code. Only llama.cpp's `common/` helpers and the example itself are compiled here, from
# the llama.cpp commit the release was built from; the script refuses any other commit.
#
# Usage:
#   scripts/build_llama_cpp_reference_tool.sh [--static] LLAMA_CPP_SRC STATIC_LLAMA_DIR OUT_DIR
#
#   LLAMA_CPP_SRC     a llama.cpp checkout at the release's llama_cpp.commit, e.g.
#                       git init llama.cpp && cd llama.cpp
#                       git fetch --depth 1 https://github.com/ggml-org/llama.cpp <commit>
#                       git checkout FETCH_HEAD
#   STATIC_LLAMA_DIR  the extracted, verified release for this machine's architecture
#                     (docs/development.md § "Static llama.cpp artifacts")
#   OUT_DIR           where the objects and OUT_DIR/llama-embedding go
#   --static          link statically, so that a binary built in the linux/arm64 container
#                     on an ARM64 host also runs on the host outside the container
#
# Needs g++ (C++17), git, python3 and the release's architecture, natively: never run it
# under emulation (docs/development.md § "Container builds").
set -euo pipefail

static=()
if [[ "${1:-}" == "--static" ]]; then
  static=(-static)
  shift
fi
if [[ $# -ne 3 ]]; then
  sed -n '2,25p' "$0" >&2
  exit 2
fi
src="$(cd "$1" && pwd)"
release="$(cd "$2" && pwd)"
mkdir -p "$3"
out="$(cd "$3" && pwd)"

release_field() {
  python3 -c 'import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2:]:
    value = value[key]
print(value)' "$release/build-info.json" "$@"
}
commit="$(release_field llama_cpp commit)"
tag="$(release_field llama_cpp describe)"
architecture="$(release_field architecture)"
if [[ "$(git -C "$src" rev-parse HEAD)" != "$commit" ]]; then
  echo "error: $src is not at llama.cpp $commit, the commit the release was built from" >&2
  exit 1
fi
if [[ "$(uname -m)" != "$architecture" ]]; then
  echo "error: the release is for $architecture, this machine is $(uname -m)" >&2
  exit 1
fi

# common/build-info.cpp, as llama.cpp's CMake writes it. `--version` prints the commit,
# which generate_llama_cpp_reference.py checks.
sed -e "s/@LLAMA_BUILD_NUMBER@/${tag#b}/" \
    -e "s/@LLAMA_BUILD_COMMIT@/${commit:0:7}/" \
    -e "s/@BUILD_COMPILER@/$(g++ -dumpfullversion | sed 's/^/GNU /')/" \
    -e "s/@BUILD_TARGET@/$(uname -s) $(uname -m)/" \
    "$src/common/build-info.cpp.in" > "$out/build-info.cpp"

mkdir -p "$out/obj"
flags=(-O2 -DNDEBUG -std=c++17 -w)
includes=(-I"$src/common" -I"$src/vendor" -I"$src/include" -I"$src/ggml/include")
llama_defines=(-DGGML_USE_CPU -DLLAMA_SUBPROCESS)
httplib_defines=(
  -DCPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=1048576 -DCPPHTTPLIB_LISTEN_BACKLOG=512
  -DCPPHTTPLIB_REQUEST_URI_MAX_LENGTH=32768 -DCPPHTTPLIB_TCP_NODELAY=1
)

jobs="$out/jobs.txt"
: > "$jobs"
for source in "$src"/common/*.cpp "$src"/common/jinja/*.cpp "$src"/common/parsers/*.cpp \
              "$src/examples/embedding/embedding.cpp"; do
  object="$out/obj/$(realpath --relative-to="$src" "$source" | tr / _).o"
  echo "$source $object ${llama_defines[*]} ${includes[*]}" >> "$jobs"
done
echo "$out/build-info.cpp $out/obj/build-info.o -I$src/common" >> "$jobs"
echo "$src/vendor/cpp-httplib/httplib.cpp $out/obj/httplib.o -fPIC ${httplib_defines[*]}" \
  >> "$jobs"

compile() {
  local source="$1" object="$2"
  shift 2
  [[ -f "$object" ]] || g++ "${flags[@]}" "$@" -c "$source" -o "$object"
}
export -f compile
export flags_string="${flags[*]}"
# shellcheck disable=SC2016 # expanded by the inner bash
xargs -P "$(nproc)" -L 1 bash -c 'read -ra flags <<< "$flags_string"; compile "$@"' _ < "$jobs"

lib="$release/lib"
main="$out/obj/examples_embedding_embedding.cpp.o"
mapfile -t objects < <(find "$out/obj" -name '*.o' ! -path "$main" | sort)
g++ "${static[@]}" "$main" "${objects[@]}" \
  "$lib/libllama.a" "$lib/libggml.a" "$lib/libggml-cpu.a" "$lib/libggml-base.a" \
  -lm -lpthread -ldl -o "$out/llama-embedding"
"$out/llama-embedding" --version 2>&1 | grep -F "commit ${commit:0:7}"
echo "built $out/llama-embedding"
