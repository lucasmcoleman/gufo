#!/bin/sh
# Builds the CPU-only fork libllama (once) and the comparison tool. No GPU.
# Usage: build.sh   (then run, with the dump from laguna.dflash_real_smoke:)
#   GUFO_LAGUNA_DFLASH_REAL=1 GUFO_LAGUNA_DFLASH_DUMP=$D laguna_dflash_real_smoke_test
#   $FORK/build-cpu/dflash_fork_cmp $D $TARGET.gguf $DRAFTER.gguf
set -eu
FORK=/server/staging/laguna/dflash-ref
SRC=$FORK/poolside-llama.cpp   # pinned: 06f8cebd7
HERE=$(cd "$(dirname "$0")" && pwd)
if [ ! -f "$FORK/build-cpu/bin/libllama.so" ]; then
  cmake -S "$SRC" -B "$FORK/build-cpu" -DGGML_HIP=OFF -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF -DGGML_BLAS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$FORK/build-cpu" -j16 --target llama
fi
g++ -std=c++17 -O1 -I"$SRC/include" -I"$SRC/src" -I"$SRC/ggml/include" \
  "$HERE/dflash_fork_cmp.cpp" -o "$FORK/build-cpu/dflash_fork_cmp" \
  -L"$FORK/build-cpu/bin" -lllama -lggml -lggml-base -lggml-cpu \
  -Wl,-rpath,"$FORK/build-cpu/bin"
echo "built $FORK/build-cpu/dflash_fork_cmp"
