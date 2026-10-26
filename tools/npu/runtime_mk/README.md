# Runtime-M and runtime-K matmul program

One XCLBIN serves every token count (M) and every reduction length (K) of the
N = 1024 block-float GEMM. Neither is compiled into the cores. A new model that
uses the same tile (128 x 64 x 128) and 1024-column slices needs **no new XCLBIN**,
only small instruction files.

## How it works

Each core has a two-word `rtp` buffer. The instruction stream writes

- word 0: `K / 64`, the number of 64-wide B tiles per output tile, and
- word 1: `(M / 128) * (N / 128) / 32`, the output tiles per core (32 cores),

then releases the core's lock; the core loops over both counts at run time. The
instruction stream also holds the DMA descriptors for the A, B and C streams, which
is where M, K and N appear. So the XCLBIN is built once and an (M, K) pair costs
one instruction file of 8-25 KB.

The XCLBINs of different (M, K) builds are the same program: they differ in about
72 bytes of metadata (checked with `cmp -l`). `final_4096xKx1024_128x64x128.xclbin`
here was built at M = 4096, K = 2048 and is used for all of them.

The older `runtime_k/` XCLBIN has K at run time only (its cores process a fixed
4096 tokens); its instruction files are not compatible with this one because they
write only word 0.

## Files

- `final_4096xKx1024_128x64x128.xclbin`: the program.
- `insts_<M>x<K>x1024_128x64x128.txt`: instruction streams. Present: K = 2048 for
  M = 1024, 1536, ..., 4096 (steps of 512) and K = 4096 for M = 4096.
- `n32_core_placed.py`, `mm_bfp_mixed.cc`: the generator and the AIE microkernel.

## Supported shapes

- N = 1024 (the host runs one GEMM per 1024-row slice of the weights).
- M a multiple of 512 (one group of four 128-row tiles). 1024 to 4096 are built and
  were run; 512 and above 4096 should work but are untested.
- K a multiple of 64 (K = 2048, 2560, 4096, 5120, 6144 and 17408 have been built
  for this tile).
- Activations, weights and outputs are block-float (`v8bfp16ebs8`, 9 bytes per 8
  values, tile-ordered); see the NPU encode and decode kernels in
  `src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp`.

## Adding a shape (new model, new K or M)

Do not rebuild the XCLBIN. Generate the instruction file, about 4-5 minutes each
(three can run at once):

```sh
source ~/npu-work/ex_env.sh
cd ~/npu-work/ex/gemm_asymmetric_tile_buffering/cfg5_MKdyn_N1024
make use_placed=1 M=<M> K=<K> N=1024 build/final_<M>x<K>x1024_128x64x128.xclbin
cp build/insts_<M>x<K>x1024_128x64x128.txt <repo>/tools/npu/runtime_mk/
```

(`build/` also receives a new XCLBIN; discard it. Calling `aiecc` with only
`--aie-generate-npu-insts` and without `--unified` compiles every core separately
and took over 12 minutes, so use the `make` recipe.) Then register the stream:

```cpp
const int program = npu->AddProgram(".../insts_<M>x<K>x1024_128x64x128.txt", &error);
npu->Submit(stream, a_id, b_ids, c_ids, count, program);  // program 0 = the one given to Create
```

Buffers for the largest M are reusable for smaller M: A and C are tile-ordered
row-major, so a smaller M is a prefix. Decode C with the same `tokens` as the
program's M.

## When the XCLBIN must be rebuilt

Only when something compiled into the cores changes: the tile sizes (`-m -k -n`),
the number of columns or rows used, the N = 1024 slice width, or `mm_bfp_mixed.cc`.

## Safety

The NPU only runs from a binary named `gufo*` with the GPU fabric clock held at its
top level; otherwise the machine can hang. Run new shapes first in `gufo prompt` or
`gufo bench`, never from a stand-alone probe. Compare final-token logits against the
GPU-only path (Flint: KL of 1e-8 for the first runtime-M build).
