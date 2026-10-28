# GLM-5.3-Flash experiments

| Date | Experiment | Decision | Evidence |
| --- | --- | --- | --- |
| 2026-10-19 | Capture position-independent one-token KDA half-layers with the shared HIP graph executor; MLA/indexer and FFN stay eager. | Rejected and removed. No gain at pp2048/tg128; short-prompt ranges overlap. | [Timing runs](artifacts/kda-graph-20261019.json), [quality checks](QUALITY.md). |

The prototype warmed one real token, recorded on the next token and replayed
once per token. Reset destroyed graphs before clearing recurrence state.
Review caught initialization ordering on the nonblocking stream and recording
failure fallback; both were fixed before the passing GPU checks. Claudems
performed the initial review; a Codex subagent checked the revisions.

The baseline harness initially mixed an old Executor object with a new header
layout in engine.cpp. That attempt was stopped and excluded. The qualified
baseline rebuilt both executor.cpp and engine.cpp against the old header;
all other production objects and the benchmark object were shared.

Evidence on MasterServer:
`/home/lucas/agentchattr-workspace/outputs/gufo-kda-graph` contains the rejected
patch, raw timing logs, binary hashes, build commands and review notes.
The initial invalid timing log is explicitly named and excluded.

## Shared gate/up input quantization, 2026-10-19

Rejected. Quantizing once for eligible dense/shared gate and up projections
preserved saved full-model logits bitwise, but pp2048 median increased from
10.2768 s to 10.3043 s. The prototype and its test were removed; the patch
remains in `outputs/gufo-prefill-20261019/paired-rejected.patch` on MasterServer.
The separate Q8_0 tail-read fix remains: partial macro tiles could load whole
activation tiles past the allocated scratch. Missing tiles now read tile zero
with a zero scale, as the K-quant implementation already does.

## Prefill chunk size, 2026-10-20

The existing 512-token option reduced measured pp2048 time by 5.29%, while
1024 reduced it by 4.87%. Both preserved the greedy sequence but changed
logits. Keep the default at 256 pending broader quality qualification;
retain the measurements as evidence for the existing option.

## Compact expert tiles, 2026-10-20

Retained after a 3.53% pp2048 gain with bitwise-identical saved logits.
Build a device list of nonempty expert column tiles, then run the original
16-column MMQ arithmetic. The path is limited to gfx1151 routed raw IQ
matrices without fused epilogues or stream-K. The existing rectangular
launch remains for other shapes. No environment switch was added.
This differs from the earlier rejected grid-striding experiment: the new
list removes empty expert/column combinations from the launch bound.

## Ideas ported from mikealanni/gufo, 2026-10-26

Source round (job 989, 2026-10-26), then a GPU round (2026-10-28, below)
that decided each switch by an unprofiled A/B plus the quality gate.

| Switch | What it does | Numerics |
| --- | --- | --- |
| `GUFO_GLM53_EXPERT_VEC=1` (`gateup`, `down`) | Few-token routed expert matvec (tokens <= 8): one lane per 256-weight block, gate and up in one launch over one activation quantization; IQ2_XXS, IQ2_S, IQ3_XXS (`kernels/expert_vec.*`) | mmvq's integer math, different F32 order; F32 activation scales |
| `GUFO_GLM53_DECODE_GRAPH=1` | HIP graphs over runs of position-independent half-layers (KDA mixer, capture-safe FFN) for passes up to max_verify_tokens; MLA attention halves stay eager | bitwise vs eager |
| `GUFO_GLM53_MLA_F16_QABS=1` | Dense prefill attention takes its scaled fp16 q from the k_b head kernel | bitwise |
| `GUFO_GLM53_KDA_LOWRANK=1` | Prefill f_b / g_b (Q8_0, 128 columns) through an LDS-tiled kernel | F32, different order |
| `GUFO_GLM53_MLA_HEADS_ROWS=4` (or 2) | k_b / v_b head kernel with 4 rows per wave | expected bitwise |
| `GUFO_GLM53_ROUTER_GEMM=sgemm` | Prefill router logits as one hipBLAS SGEMM | plain F32: quality gate |
| `GUFO_GLM53_PREFILL_CHUNK=N` | Overrides the prefill chunk (1..8192) for the chunk x MoE tile sweep; the head projects at most 256 rows at a time | chunk != 256 changes logits |

### GPU round result, 2026-10-28 (job 989)

Unprofiled, counterbalanced runs on gfx1151 with Maya-S-v2 IQ2_XXS, one
GPU job at a time. Logs and tables: /server/staging/glm53-borrow/gpu-round
(RESULT.md). NLL deltas are paired, 95% CI.

| Switch | Result | Verdict |
| --- | --- | --- |
| `GUFO_GLM53_EXPERT_VEC` | decode +5.2% plain, +6.4% MTP; NLL -0.0043 [-0.0191, +0.0097] (64 x 8-token passes) | on by default (`0` turns it off) |
| `GUFO_GLM53_KDA_LOWRANK` | prefill -3.8%; NLL -0.0063 [-0.0210, +0.0076] | on by default (`0` off) |
| `GUFO_GLM53_ROUTER_GEMM` | prefill -2.5%; NLL -0.0090 [-0.0309, +0.0060] | `sgemm` by default (`wave` off) |
| `GUFO_GLM53_MLA_HEADS_ROWS` | prefill -1.5%, bitwise | 4 by default (`1` off) |
| `prefill_chunk` | 2048-token prompt 13.4 s (256, MoE tile 16) -> 10.2 s (2048, tile 16); NLL on one 2048-token text -0.0068 [-0.0165, +0.0008] | 2048 by default |
| `GUFO_GLM53_DECODE_GRAPH` | bitwise on the real model, 0.0% / -0.1% | dropped (stays off) |
| `GUFO_GLM53_MLA_F16_QABS` | bitwise, -0.4% (noise) | dropped (stays off) |

All five keepers together vs the old defaults (same session, rounds 2-3):
decode plain 16.75 -> 17.62 tok/s (+5.2%), MTP 21.69 -> 24.37 tok/s
(+12.3%); 2048-token prefill 13.69 -> 9.15 s (-33%); NLL on two 2048-token
texts +0.0025 [-0.0070, +0.0106]. Prefill with MoE tile 64 at chunk 2048
was another -5% but is not the default (wider quality CI, untested stacked).
Not measured: prompts between 256 and 2048 tokens, and serve under load.

Ideas are from Mike Alani's fork (MIT), commits c338778f, 5a394170,
8808f1a0, 5b505adf, e5490fee, 3ec04be3, 2a9e953c, 3ba034e2, 92f05df6 and
a9ea8e67; the code is GLM-local, not his patches. His attend-rescale skip
(3ec04be3) has no counterpart here: our latent attention materialises the
scores and normalises once, with no running-max accumulator to rescale.
