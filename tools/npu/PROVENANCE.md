# NPU programs: provenance and rebuilding

Everything in this directory comes from Mike Alani's fork
[mikealanni/gufo](https://github.com/mikealanni/gufo) at `fork-enhancements`
`2e499c39e4cd2d9250da2f44f529435f04995c84`. The AIE sources
(`mm_bfp_mixed.cc`, `n32_core_placed.py`) carry AMD's
`Apache-2.0 WITH LLVM-exception` notice ([licenses/mlir-aie.txt](../../licenses/mlir-aie.txt));
the runtime copies record Mike's changes in their headers. The XCLBIN and
instruction files are binaries he built with mlir-aie and Peano. None was
rebuilt here.

| Directory | Program | XCLBIN SHA-256 | Added in |
| --- | --- | --- | --- |
| `.` | fixed M = 4096, K = 2560 (PR #568, the live build's program) | `d31ceb12b0b62dcbb2e32d1ae73f2125b3fe15871b8570a1c89372f3c3fb410e` | `736a9f4c` |
| `runtime_k/` | K at run time, M = 4096 | `d229c66ab90641bd78a72c7e25711323dbe40053e38b8f73c0d48d2c48140b13` | `23c2d158` |
| `runtime_mk/` | M and K at run time | `f25931ff8851fd78aca963934b60a5d600dabc4b3350aa239f094f0caadabae0` | `ccfb7747` |

Instruction streams (little-endian 32-bit words):

| File | SHA-256 | Added in |
| --- | --- | --- |
| `insts_4096x2560x1024_128x64x128.txt` | `bfec14665c6bc0bda9c8abd2536172b9459a3e234c7890faa96b6c9ecbda8a90` | `736a9f4c` |
| `runtime_k/insts_4096x2048x1024_128x64x128.txt` | `fbdddaef3e8e4815e376571c239de3e9fc8f89a65450982517fecd713318b864` | `23c2d158` |
| `runtime_k/insts_4096x2560x1024_128x64x128.txt` | `dbab4045ea834a2e7b857e7553766c7918061951eee7b6165e174f123562b68b` | `23c2d158` |
| `runtime_k/insts_4096x4096x1024_128x64x128.txt` | `00a656fedeb4661cc66a01c4d05adeafe05399617d8168ba5245904d7c17d43b` | `1c00f953` |
| `runtime_k/insts_4096x5120x1024_128x64x128.txt` | `d75bb96e882c0107187bc55d8c835a8b7b146d2f7b730b7be1f4c3931cf4dcfe` | `23c2d158` |
| `runtime_k/insts_4096x6144x1024_128x64x128.txt` | `5a81a842062ce79fceb5f85eee1feb21e185a4e5885b59d9fecba1176ca35bd0` | `e7774b7c` |
| `runtime_k/insts_4096x17408x1024_128x64x128.txt` | `f45374b0556a6fb87c60d89f983a4278382a852809efb329c63c477050f3bbe8` | `e7774b7c` |
| `runtime_mk/insts_1024x2048x1024_128x64x128.txt` | `6b43c725934eb5fc29e2bb4a014713d7bbaad7c459a633e09812731b9c8b9862` | `ccfb7747` |
| `runtime_mk/insts_1536x2048x1024_128x64x128.txt` | `cbfa27f352bdd5d97c969c93c720e6ad868ae61a59c346f46d394961311fd07a` | `ccfb7747` |
| `runtime_mk/insts_2048x2048x1024_128x64x128.txt` | `5af43789a7f279de946cf9082a6dc446fc08f014f7d0b194994e8e3a9022347b` | `ccfb7747` |
| `runtime_mk/insts_2560x2048x1024_128x64x128.txt` | `3024362adc9fef4fd4f2f874d6879f7f6cad1671856ee6f857a1445a1c65d45b` | `ccfb7747` |
| `runtime_mk/insts_3072x2048x1024_128x64x128.txt` | `21e5344b00297a05cc5e9c0a9d842dda177d1dbef6fe372c0155fe7b6cfbde99` | `ccfb7747` |
| `runtime_mk/insts_3584x2048x1024_128x64x128.txt` | `a239e6f6e26e800f6ee110b13f3aea61b19c3abc29b0cb4cb9b8c42feee0e557` | `ccfb7747` |
| `runtime_mk/insts_4096x2048x1024_128x64x128.txt` | `5afe6d8db947f9d8454950fb6d50df5e257879ce8c801d19b0603ceaa5efd686` | `ccfb7747` |
| `runtime_mk/insts_4096x4096x1024_128x64x128.txt` | `db90b5578c8d42c77dbefc25577f73d544f05c5484198c67c01db06f30d2bd37` | `ccfb7747` |

## What was checked here (CPU only, 2026-10-26)

- `tests/models/qwen38_flash_next/npu_plan_test.cpp` reads the streams as
  data. Every runtime-K stream writes K / 64 into the rtp word of all 32
  cores; every runtime-M/K stream writes K / 64 and M / 512. At K = 2560 the
  runtime-K stream is the fixed stream's DMA program (5,504 identical words)
  behind a 384-word rtp prologue, so both programs move the same data.
- `xclbinutil --info` (XRT 2.21, reads the file only): all three XCLBINs
  expose one `MLIR_AIE` kernel with the same arguments (`instr`, `bo0`-`bo4`)
  and differ in their PDI (core programs) and UUID.
- `mm_bfp_mixed.cc` (fixed and runtime variants) compiles for `aie2p` with the
  Peano clang 21 shipped in the Ryzen AI 1.8 venv on MasterServer
  (`-DDIM_M=128 -DDIM_K=64 -DDIM_N=128`, an empty `aie_kernel_utils.h`). The
  compiled `.text` does not occur in the PDIs, so the prebuilt core code could
  not be matched byte for byte; the compiler build and flags Mike used are
  unknown.
- Not checked: that the XCLBINs were built from exactly these sources. That
  needs mlir-aie, which is not installed here.

## Rebuilding (not done here; needs mlir-aie, which is not on this box)

The fork records only a partial recipe (`runtime_mk/README.md`), run from
Mike's local copy of an mlir-aie GEMM example
(`~/npu-work/ex/gemm_asymmetric_tile_buffering/cfg5_MKdyn_N1024`):

```sh
source ~/npu-work/ex_env.sh          # mlir-aie + Peano environment
make use_placed=1 M=<M> K=<K> N=1024 build/final_<M>x<K>x1024_128x64x128.xclbin
```

Reconstructed from the sources, the steps that recipe runs are:

1. Install mlir-aie (its Python package provides `aie.*` and `aiecc.py`) and
   Peano (llvm-aie). Both are published by AMD/Xilinx on GitHub; check that
   the download needs no login.
2. Compile the microkernel for AIE2P:
   `clang++ --target=aie2p-none-unknown-elf -O2 -std=c++20 -DDIM_M=128 -DDIM_K=64 -DDIM_N=128 -c mm_bfp_mixed.cc -o mm_128x64x128.o`
   (with mlir-aie's `aie_kernels` include directory for `aie_kernel_utils.h`).
3. Emit the placed design:
   `python3 n32_core_placed.py -M <M> -K <K> -N 1024 -m 128 -k 64 -n 128 > aie.mlir`
   (the generator links `mm_128x64x128.o` by name).
4. Compile: `aiecc.py --aie-generate-xclbin --xclbin-name=final_<M>x<K>x1024_128x64x128.xclbin --aie-generate-npu-insts --npu-insts-name=insts_<M>x<K>x1024_128x64x128.txt --no-compile-host aie.mlir`
   (exact flags depend on the mlir-aie release; the fork notes that
   `aiecc` without `--unified` took over 12 minutes).

For a runtime program only the instruction file is new per shape; the fork
keeps one XCLBIN and discards the rebuilt one. Compare a rebuilt instruction
stream word for word with the shipped one
(`/server/staging/npu-borrow/scripts/insts_compare.py`), and the XCLBIN by
`xclbinutil --info` and its PDI; the fork reports that XCLBINs of different
(M, K) builds differ in about 72 bytes of metadata.

Run any new program first through `gufo prompt` or `gufo bench` under the NPU
prerequisites (fabric clock held), never from a stand-alone probe.
