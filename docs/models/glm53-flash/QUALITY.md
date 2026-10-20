# GLM-5.3-Flash quality evidence

The rejected KDA graph experiment was checked on 2026-10-19 against `98daaee`.
All three focused GPU binaries passed: `kda_executor`, `executor_loop`, and
`generator_gpu`. The production release build passed.

The KDA test compared capture/replay with an independent CPU recurrence oracle
for F32 and Q8_0 weights, changing inputs and resetting for a second cycle.
The existing 1e-5 tolerance was unchanged. The Executor test compared five
single-token steps against eager execution and repeated them after reset;
finite logits matched bitwise. Its log contained 144 graph-hit events.

The real-model benchmark saved 768 full-vocabulary rows across six cases.
Both 475,791,360-byte files had SHA-256
`62a6a5f65753dce0b21d4dd3a04c86441419ea88880718eac8730faaf95a9070`.
Each case checks the prefill output and first 127 decode outputs for finiteness.
The final decode call is timed, but its output is not saved or checked.
This is regression evidence against Gufo, with an independent operator oracle;
no new full-model upstream-teacher qualification was performed.

The experimental code and tests were removed after the performance rejection.
The patch, test logs, harness and build commands remain at
`/home/lucas/agentchattr-workspace/outputs/gufo-kda-graph` on MasterServer.

## Q8_0 activation tail fix, 2026-10-19

The new mixed Q6_K/Q8_0 projection fixture exposed an invalid Q8_0 read at
batch 65. The failure also occurred with the original separate projections,
a fresh stream and dedicated device scratch. The shared Q8_0 WMMA kernel
formed activation pointers beyond the allocated tile count. It now clamps
missing tiles to tile zero and clears their scale; live-tile arithmetic is
unchanged. No synchronization was added to production.

The Qwen prefill operator test passes with exactly sized activation scratch,
including new batches 65 and 129. GLM FFN tests pass in default and exact
modes, and the independent GLM prefill GEMM oracle passes. The production
release build passes. Raw logs and the failure diagnosis are under
`/home/lucas/agentchattr-workspace/outputs/gufo-prefill-20261019`.

Both matched performance binaries included this safety fix. The rejected
paired-quantization candidate matched all saved logits byte for byte, SHA-256
`62a6a5f65753dce0b21d4dd3a04c86441419ea88880718eac8730faaf95a9070`.
The harness saved prefill plus the first 127 decode outputs per run; it timed
but did not save the final decode output. This comparison isolates paired
quantization, not the safety fix. No new upstream-teacher qualification was run.

Claudems reviewed the tail fix with no blockers. The wave64 follow-up was
checked at source: small cases decline that route; eligible ragged cases use
`WKQuantA8BlockedWmmaGEMMKernel`, which already clamps missing activation
tiles and zeros their scale. The failing 65-by-256-by-256 case reaches the
fixed W8 kernel.
