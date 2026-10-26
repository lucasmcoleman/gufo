# Laguna-S-2.1 architecture (implementation spec)

Source: poolside/Laguna-S-2.1 `config.json` and `modeling_laguna.py` (copies in
`/server/ai/models/poolside/Laguna-S-2.1`), read 2026-10-26. Where the target
GGUF's metadata differs from the HF config (context length, YaRN factor), the
GGUF wins.

## Shapes

| Item | Value |
| --- | --- |
| Layers | 48 |
| Hidden | 3072 |
| Vocab | 100352, untied output |
| Head dim | 128 |
| KV heads | 8 on every layer |
| Query heads | 48 on full-attention layers, 72 on sliding-window layers |
| Layer pattern | full at `il % 4 == 0`, sliding otherwise (12 full, 36 sliding) |
| Sliding window | 512 |
| Layer 0 MLP | dense SwiGLU, intermediate 12288 |
| Layers 1 to 47 MLP | MoE: 256 routed experts, top 10, expert intermediate 1024, plus 1 shared expert (intermediate 1024) |
| RMSNorm eps | 1e-6 |
| BOS / EOS | 2 / {2, 24}, pad 9 |

## Layer

```
x  = h + Attn(RMSNorm_in(h))
h' = x + MLP(RMSNorm_post(x))
```

Final `RMSNorm` then `lm_head`.

## Attention

With `n` the input after `input_layernorm`:

1. `q = n Wq` (`heads*128`), `k = n Wk`, `v = n Wv` (`8*128` each). No biases.
2. Per-head RMSNorm on q and k (`q_norm`, `k_norm`, weight `[128]`), before RoPE.
3. RoPE, NeoX layout (`rotate_half`), on the leading `rotary_dim` channels only:
   - full layers: `rotary_dim = 64` (partial factor 0.5), theta 500000, YaRN
     (orig ctx 8192, beta_fast 32, beta_slow 1; factor and attention factor
     from GGUF metadata: factor 32). The GGUF's `yarn_attn_factor` 1.0 is
     not the net cos/sin scale: llama.cpp doesn't read that key for Laguna,
     and its YaRN path (`llama-context.cpp:269-306`) gives a net multiplier of
     1 + 0.1 ln(32) = 1.3466 on full layers (`Config::YarnMscale()`). That
     comes from reading the source, so confirm it against a phase 2 tap.
   - sliding layers: `rotary_dim = 128`, theta 10000, no scaling.
4. Causal softmax attention, scale `1/sqrt(128)`, GQA groups 6 (full) or 9
   (sliding). Sliding layers see keys at positions `p-511 .. p` (the
   transformers sliding mask keeps `window` tokens including the current one;
   confirm against the reference at a long prompt).
   Attention sinks exist in the code behind `swa_attention_sink_enabled`,
   which this config does not set. Check the GGUF for sink tensors.
5. Output gate, per head: `g = softplus(n Wg)`, `Wg` is `[3072, heads]`, F32 in
   the v2 GGUFs. Head `i`'s 128 outputs are multiplied by `g_i`.
6. `o = gated Wo`.

## MoE

```
logits = n Wr                      (F32, [256])
s      = sigmoid(logits)
sel    = topk(s + bias, 10)        (bias = e_score_correction_bias, selection only)
w      = s[sel] / sum(s[sel])      (norm_topk_prob)
out    = 2.5 * sum_j w_j * Expert_j(n) + Shared(n)
```

Experts and the shared expert are SwiGLU (`silu(gate) * up`, then down). The
shared expert has no sigmoid gate (unlike Qwen MoE). Router softcapping is 0,
and weights apply to expert outputs, not inputs.

Normalization contract: llama.cpp d854d98 (the logit reference) divides by
`max(sum, 2^-14)` in the generic `build_moe_ffn` (`llama-graph.cpp:2347`),
while Poolside's `modeling_laguna.py:182` divides by the raw sum. gufo follows
llama.cpp, because that is what the KL/top-1 gate measures against. The two
differ only when the ten selected sigmoids sum below 2^-14. The CPU oracle and
the MoE operator tests include an extreme-negative router-logit case (all
logits around -20) that exercises the clamp, the same fixture that caught this
difference on GLM (codexms #16129).

## KV cache

Full layers need the whole context. Sliding layers only ever read the last 512
positions, so a ring buffer of 512 (plus the verification/prefill chunk) is
enough for them. At 262144 context: full layers 12 x 2 x 8 x 128 x 2 B = 48 KiB
per token in F16, about 12 GiB at full context; sliding layers are bounded.

## llama.cpp reference

Upstream `src/models/laguna.cpp` (ggml-org master d854d98, 2026-10-26) matches
the above and fixes the GGUF names: `attn_q/k/v/output`, `attn_q_norm`,
`attn_k_norm`, `attn_gate` (`[3072, heads]` per head), `ffn_gate_inp`,
`exp_probs_b.bias` (llama.cpp's enum calls it `ffn_exp_probs_b`), `ffn_{gate,up,down}_exps`, `ffn_{gate,up,down}_shexp`,
dense `ffn_{gate,up,down}` on the `leading_dense_block_count` layers, plus per-layer `attn_norm` and `ffn_norm`. The GGUF sets `tokenizer.ggml.add_sep_token` true, but llama-tokenize adds only BOS, and gufo matches that. The gate
reads the attention-norm output (same input as q/k/v). SWA layers use
`rope.freq_base_swa` and `rope.dimension_count_swa` with no YaRN (ext factor 0,
attn factor 1, freq scale 1). Routing comes from `expert_gating_func`
(sigmoid), `expert_weights_norm` and `expert_weights_scale`. llama.cpp's KV
for SWA layers is `LLAMA_SWA_TYPE_STANDARD`: a key at position `j` is visible
from `p` when `p - j < n_swa`.

## Open questions

- Sliding-window boundary: follow llama.cpp's `LLAMA_SWA_TYPE_STANDARD`
  (`p - j < 512`) and confirm it with taps at positions 511/512/513.
- Resolved: tensor names and quant types (codexms #16074, header manifest in
  `outputs/laguna-gufo-assessment/`); upstream llama.cpp d854d98 has `laguna`,
  so the Poolside fork is only needed for DFlash.
