# GLM-5.3-Flash benchmarks

Measured 2026-10-19 on MasterServer (gfx1151), Maya-S-v2-IQ2_XXS, baseline
`98daaee`. Same production compiler flags and dependencies for both variants.
A standalone Model-session harness times synchronous `Forward` calls directly;
`gufo bench` does not support this model. Single session, 128 greedy decode
calls, three reset runs per prompt length. Prompts repeat the tokens from
“Write a detailed story about a lighthouse keeper.” to the stated length.

| Prompt tokens | Baseline decode tok/s | KDA graph prototype tok/s | Change |
| --- | ---: | ---: | ---: |
| 32 | 16.9931 | 17.2320 | +1.41% |
| 2048 | 12.7298 | 12.6863 | -0.34% |

Values are medians; all individual runs are in the
[measurement artifact](artifacts/kda-graph-20261019.json). Short-prompt ranges
overlap. The prototype was rejected; `98daaee` remains the implementation.

Baseline ran before candidate, each in a separate process under the same GPU
admission lock. Model load, session reset, argmax and logit-file writes are
outside decode timing. Loading took 25.11 s and 18.08 s respectively; the
ordering and cache history prevent attributing that difference to graphs.
Prefill includes allocation and warmup variation; raw per-run values are retained.
These direct timings are not interchangeable with prior stdout-chunk timings.
