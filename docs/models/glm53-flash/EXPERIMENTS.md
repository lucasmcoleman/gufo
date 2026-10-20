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
