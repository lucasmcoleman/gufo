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
