# Laguna-S-2.1 in gufo: build plan (job 961)

Target: `Myric/Laguna-S-2.1-APEX-GGUF`, file `Laguna-S-2.1-APEX-i-quality-v2.gguf`
(814 tensors: F32 335, Q6_K 221, Q8_0 144, Q5_K 30, IQ4_XS 84 routed experts in
layers 10 to 37). Reference: upstream llama.cpp `src/models/laguna.cpp`
(`/server/programming/llama.cpp-laguna`, d854d98). Spec: `ARCH.md` here.

## Template

`src/models/laguna/` follows the `glm53_flash` package on
`perf/glm53-mla-splitk` (37f26b03), which already has what Laguna's MoE
needs: sigmoid routing with a selection-only bias, a weights scale, a shared
expert, leading dense layers, IQ MMQ kernels, an F32 CPU oracle, and
llama.cpp-named taps diffed against `llama-eval-callback` dumps.

| glm53_flash part | Laguna |
| --- | --- |
| config, weights, tokenizer, chat template | rewrite for `laguna.*` keys, tensor names, the Laguna pre-tokenizer, and Poolside's template |
| layer_ops contract | keep the shape; `MlaForward` becomes `AttnForward` |
| MLA + DSA indexer kernels | drop; new GQA attention (QK-norm, NeoX partial RoPE with YaRN on full layers, SWA 512, per-head softplus gate) |
| hyper-connections, KDA | drop; plain pre-norm residual |
| FfnForward (sigmoid MoE + shexp) | reuse; no SwiGLU clamps; add routed Q6_K and IQ4_XS experts: `qfn_mmq.hip.cpp:361` (`moe_vector_projection`) accepts only Q4_K/Q5_K/Q5_1/Q8_0 on main 5eedd099 (codexms #16077, #16129); the vendored mmq/vecdotq already have both types |
| reference (CPU oracle) | reuse the MoE half, write GQA attention |
| executor, generator, engine, serve runner | reuse with the simpler layer loop and a per-layer KV cache (full layers sized to context, SWA layers a 512 ring) |

## Phases and checks

1. Config + weights + tokenizer, CPU only. Check: config test against the GGUF
   header; token ids match `llama-tokenize` on a fixed corpus. Decide the BOS
   policy explicitly (the GGUF adds BOS 2) and use the same policy on both
   sides of every later comparison.
2. CPU oracle forward, a few tokens. Two oracles: F32 dequantized weights,
   and a Q8-activation-aware one, so that kernel arithmetic bugs aren't
   mistaken for expected quantization error (GLM #13058-13156). The MoE half
   comes from glm53_flash `reference_moe.cpp`, which already supports both
   denominator conventions. Tiny-model fixtures randomize every weight and use
   unequal dimensions (default init hid bugs, #12545). Check: final logits and
   per-layer taps vs `llama-eval-callback` on the same prompt.
3. GPU operators, one at a time against the oracle: attention on full and SWA
   layers separately (GQA groups 6/9; positions 511/512/513; chunk splits
   crossing the window; ring wrap; prefill vs single-row decode), FFN dense,
   MoE with each routed expert quant (Q6_K L1-4/43-47, Q5_K L5-9/38-42,
   IQ4_XS L10-37), plus the extreme-negative router-logit case for the
   `max(sum, 2^-14)` clamp.
4. GPU executor prefill + decode, single session. Check: teacher-forced KL /
   top-1 / tail errors vs llama.cpp over 64 x 256-token texts, with all
   positions scored and finite-logit checks. Measure Laguna's own floor
   (llama.cpp ubatch 8 vs 256). The GLM and Ornith floors are examples, not
   tolerances. Harness: `/server/staging/glm53-gufo/moe-compare/v6`,
   `/server/staging/glm53-quality/score.py` (fixed GLM paths, so adapt them),
   and the method in docs/models/ornith-1.5-122b/QUALITY.md. Never loosen a
   tolerance to hide an unexplained mismatch.
5. Serve runner, sessions, snapshots, functional suites, benchmarks. Run
   untapped resume, cancel, retry, multi-session and load/unload checks, since
   taps synchronize streams and hide races (Ornith #14879-15027). Keep
   per-stream scratch. If arena code is borrowed, it must include the
   510169bf pool-lifetime fix (#16022). Budget full-layer KV, SWA ring,
   snapshots, sessions and scratch together. Report passes, skips and
   failures separately. Decide perf changes by unprofiled, counterbalanced
   A/B on matched Release binaries.
6. DFlash drafter (`laguna-s-2.1-DFlash-Q4_K.gguf`, arch `dflash`), only after
   5, with its own feature-layer and mask verification.

Lessons source: codexms #965 review of #ai-dev-gufo 12015-16101,
`/server/staging/laguna/codexms-965-RESULT.txt`. Reference build:
`/server/programming/llama.cpp-laguna/BUILD-LAGUNA-REF.txt`.

Each phase lands as its own commit on `feat/laguna-s21`.
