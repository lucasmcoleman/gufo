# Qwen3.8 Flash-Next on the XDNA2 NPU

The XDNA2 NPU path is optional. It is compiled in with `-DGUFO_ENABLE_NPU=ON`
(see [Build from source](../../../README.md#build-from-source)) and switched on at run time by pointing
`GUFO_NPU_DIR` at `tools/npu`. Without `GUFO_NPU_DIR` the same binary runs
GPU-only.

## Prerequisites at run time

| Requirement | Command |
| --- | --- |
| GPU fabric clock held at its top level | `sudo tools/npu/fabric-clock.sh hold` (`status`, `release`) |
| NPU power mode `balanced` or `performance` (not `turbo`) | `sudo xrt-smi configure --pmode performance` |
| NPU kept out of runtime suspend | `echo on \| sudo tee /sys/bus/pci/devices/<npu pci id>/power/control` |
| XRT libraries on the loader path | `LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH` |

The fabric clock changing speed while the NPU is busy can hang the machine, so
hold it before starting. The clock and power mode reset at boot.

## Environment variables

### NPU on and off

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_NPU_DIR` | unset (NPU off) | Directory with the NPU kernel files (`tools/npu`). Setting it turns the NPU on. |
| `GUFO_NPU_PROGRAM` | `fixed` | Which NPU program to load from `GUFO_NPU_DIR`: `fixed` is the M = 4096, K = 2560 pair in the directory itself; `runtime_k` loads `runtime_k/` (one XCLBIN for every K, K / 64 written by the instruction stream; its K = 2560 stream moves the same data as the fixed one); `runtime_mk` loads `runtime_mk/`, which has no K = 2560 stream yet, so the NPU stays off until one is generated. Experimental, awaiting the hardware A/B. See [tools/npu/PROVENANCE.md](../../../tools/npu/PROVENANCE.md). |

### What the NPU computes

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_NPU_SLICES` | `6` | 1024-row slices of the linear-attention `ssm_in` projection (z rows) computed on the NPU, 1 to 6. |
| `GUFO_NPU_GATE_SLICES` | `8` | 1024-row slices of the attention `attn_qkv` projection (query and gate rows, two heads per slice) on the NPU, 0 to 12. `0` turns this part off. |
| `GUFO_NPU_SHEXP` | on | `0` keeps the shared-expert gate and up projections on the GPU. |
| `GUFO_NPU_LAYERS` | all | Limit how many linear-attention layers use the NPU. |

### Two-bank prefill and the F16 residual (experimental, default off)

Ported from mikealanni/gufo (MIT, Mike Alani). All four switches stay off
until the hardware A/B; none changes a default path.

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_PREFILL_TILE` | `4096` | Prefill chunk in tokens (128 to 65536). The two-bank prefill needs at least 8192; the fork runs 16384. Scratch grows with it (about 9 GiB at 16384, plus about 4.5 GiB for bank B). |
| `GUFO_TWO_BANK` | unset | `1` splits a prefill chunk of 8192 tokens or more (whole 4096-token groups) into two banks that advance layer by layer. Without `GUFO_NPU_WHOLE` the banks run the ordinary layer code in lock-step (expected to match the one-bank result). `2` also accepts chunks that are not a multiple of 4096. Checkpoint passes and image prompts keep the one-bank path. |
| `GUFO_NPU_WHOLE` | unset | With `GUFO_TWO_BANK=1`: the complete `ssm_in` (16 slices) and `attn_qkv` (13 slices) projections run on the NPU one bank ahead, and the GPU reads them straight from the NPU streams (convolution, gated norm, query/gate preparation). Not bit-identical (block-float projections). About 3.3 GiB more NPU weight buffers. |
| `GUFO_NPU_WHOLE_CHUNKED` | on within `GUFO_NPU_WHOLE` | The whole-projection linear layers use F16 convolution rows and the chunked (WY-form) gated delta rule, about 6e-4 relative against the row-split scan. `0` keeps F32 rows and the row-split scan, to separate its cost in an A/B. |
| `GUFO_F16_RESIDUAL` | unset | `1` keeps the prefill residual in F16 between the combines of large wide-prefill chunks (1024 rows or more), with or without the NPU. The fork measured +0.5% perplexity and +1.5% prefill with the NPU. Independent of `GUFO_TWO_BANK`. |

The fork's two-bank code restored the primary scratch only on success, and its
draft block skipped the pair (last row of bank A, first token of bank B),
shifting bank B's draft-cache positions by one; both are fixed here.

### Reliability and diagnostics

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_NPU_TIMEOUT_MS` | `8000` | Time a single NPU run may take before it is treated as failed. |
| `GUFO_NPU_KEEPALIVE` | on | `0` stops the 1 s heartbeat that keeps the NPU awake. |
| `GUFO_NPU_WARMUP_MS` | `0` | Keep the NPU out of the first N ms of each prefill burst so the GPU's idle-to-full step happens alone. |
| `GUFO_NPU_ISOLATE` | unset | `1` gives the NPU private buffers and copies through the CPU. Slower; for debugging. |
| `GUFO_NPU_STATS` | unset | Print wall-clock accounting of the NPU path. |
| `GUFO_NPU_TRACE` | unset | Path of a synchronous event log, one line per NPU event. |

## Examples

NPU on, benchmark:

```sh
GUFO_NPU_DIR=tools/npu build/release/gufo bench -m <model.gguf> -p 16384 -n 1 -r 3
```

Same, GPU-only: leave `GUFO_NPU_DIR` unset.

Only the z rows on the NPU:

```sh
GUFO_NPU_DIR=tools/npu GUFO_NPU_GATE_SLICES=0 GUFO_NPU_SHEXP=0 build/release/gufo serve llm ...
```
